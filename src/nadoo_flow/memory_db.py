"""
Database-backed Chat History for Nadoo Flow
DB 기반 영구 저장소 + Redis 캐싱 이중 저장 패턴

Architecture:
    Redis (Fast Cache, TTL) → DB (Persistent Storage, Forever)

    Write: Both Redis + DB
    Read: Redis first (cache hit) → DB fallback (cache miss)
"""

import json
import logging
from typing import Any, Protocol

from .memory import BaseChatHistory, Message

logger = logging.getLogger(__name__)


class DatabaseAdapter(Protocol):
    """Database adapter protocol

    SQLAlchemy, Django ORM, or any DB ORM can implement this.
    """

    async def save_message(
        self,
        session_id: str,
        message: Message,
        workspace_id: str | None = None
    ) -> None:
        """Save a single message to DB"""
        ...

    async def save_messages(
        self,
        session_id: str,
        messages: list[Message],
        workspace_id: str | None = None
    ) -> None:
        """Save multiple messages to DB"""
        ...

    async def get_messages(
        self,
        session_id: str,
        workspace_id: str | None = None,
        limit: int | None = None
    ) -> list[Message]:
        """Get messages from DB"""
        ...

    async def clear_messages(
        self,
        session_id: str,
        workspace_id: str | None = None
    ) -> None:
        """Clear all messages for a session"""
        ...


class DatabaseChatHistory(BaseChatHistory):
    """Database-backed chat history with Redis caching

    이중 저장 패턴:
    - Redis: 빠른 액세스, TTL 기반 자동 만료 (캐시)
    - DB: 영구 저장, 검색 및 분석 가능 (persistent)

    Example:
        from sqlalchemy.ext.asyncio import AsyncSession

        # DB adapter 구현
        class SQLAlchemyAdapter(DatabaseAdapter):
            def __init__(self, db: AsyncSession):
                self.db = db

            async def save_message(self, session_id, message, workspace_id=None):
                chat_message = ChatMessage(
                    session_id=session_id,
                    workspace_id=workspace_id,
                    role=message.role,
                    content=message.content,
                    metadata=message.metadata,
                    timestamp=message.timestamp
                )
                self.db.add(chat_message)
                await self.db.commit()

            async def get_messages(self, session_id, workspace_id=None, limit=None):
                query = select(ChatMessage).where(
                    ChatMessage.session_id == session_id
                )
                if workspace_id:
                    query = query.where(ChatMessage.workspace_id == workspace_id)
                if limit:
                    query = query.limit(limit)

                result = await self.db.execute(query)
                messages = result.scalars().all()

                return [
                    Message(
                        role=msg.role,
                        content=msg.content,
                        metadata=msg.metadata,
                        timestamp=msg.timestamp
                    )
                    for msg in messages
                ]

        # 사용
        db_adapter = SQLAlchemyAdapter(db=db_session)
        history = DatabaseChatHistory(
            session_id="user_123",
            db_adapter=db_adapter,
            redis_client=redis_client,
            workspace_id="workspace_abc",
            redis_ttl=3600  # 1시간 캐시
        )
    """

    def __init__(
        self,
        session_id: str,
        db_adapter: DatabaseAdapter,
        redis_client: Any | None = None,
        workspace_id: str | None = None,
        redis_key_prefix: str = "chat_history:",
        redis_ttl: int | None = 3600
    ):
        """
        Args:
            session_id: 세션 ID
            db_adapter: DB 어댑터 (save/get messages 구현)
            redis_client: Redis 클라이언트 (optional, 없으면 DB only)
            workspace_id: 워크스페이스 ID (멀티테넌시)
            redis_key_prefix: Redis 키 접두사
            redis_ttl: Redis TTL (초), None이면 영구
        """
        self.session_id = session_id
        self.db_adapter = db_adapter
        self.redis = redis_client
        self.workspace_id = workspace_id
        self.redis_key_prefix = redis_key_prefix
        self.redis_ttl = redis_ttl

    def _make_redis_key(self) -> str:
        """Redis 키 생성"""
        if self.workspace_id:
            return f"{self.redis_key_prefix}{self.workspace_id}:{self.session_id}"
        return f"{self.redis_key_prefix}{self.session_id}"

    async def get_messages(self) -> list[Message]:
        """메시지 조회 (Redis → DB fallback)"""

        # ✅ 1. Redis에서 먼저 조회 (cache hit)
        if self.redis:
            try:
                key = self._make_redis_key()
                cached = self.redis.lrange(key, 0, -1)

                if cached:
                    logger.debug(f"📦 Redis cache HIT: {key}")
                    messages = []
                    for msg_json in cached:
                        msg_data = json.loads(msg_json)
                        messages.append(Message(**msg_data))
                    return messages

                logger.debug(f" Redis cache MISS: {key}")
            except Exception as e:
                logger.warning(f"Redis error, fallback to DB: {e}")

        # ✅ 2. DB에서 조회 (persistent storage)
        messages = await self.db_adapter.get_messages(
            session_id=self.session_id,
            workspace_id=self.workspace_id
        )

        # ✅ 3. Redis에 캐싱 (다음 조회 시 빠르게)
        if self.redis and messages:
            try:
                await self._cache_messages_to_redis(messages)
            except Exception as e:
                logger.warning(f"Failed to cache to Redis: {e}")

        return messages

    async def add_message(self, message: Message):
        """메시지 추가 (DB + Redis 동시 저장)"""

        # ✅ 1. DB에 영구 저장
        await self.db_adapter.save_message(
            session_id=self.session_id,
            message=message,
            workspace_id=self.workspace_id
        )
        logger.debug(f"💾 Saved to DB: session={self.session_id}")

        # ✅ 2. Redis에 캐싱 (선택)
        if self.redis:
            try:
                key = self._make_redis_key()
                msg_json = json.dumps(message.to_dict())
                self.redis.rpush(key, msg_json)

                if self.redis_ttl:
                    self.redis.expire(key, self.redis_ttl)

                logger.debug(f"📦 Cached to Redis: {key} (TTL={self.redis_ttl}s)")
            except Exception as e:
                logger.warning(f"Redis caching failed (non-critical): {e}")

    async def add_messages(self, messages: list[Message]):
        """여러 메시지 추가 (Batch)"""

        # ✅ 1. DB에 배치 저장
        await self.db_adapter.save_messages(
            session_id=self.session_id,
            messages=messages,
            workspace_id=self.workspace_id
        )
        logger.debug(f"💾 Batch saved to DB: {len(messages)} messages")

        # ✅ 2. Redis에 캐싱
        if self.redis:
            try:
                await self._cache_messages_to_redis(messages)
            except Exception as e:
                logger.warning(f"Redis batch caching failed: {e}")

    async def clear(self):
        """히스토리 초기화 (DB + Redis 모두)"""

        # ✅ 1. DB에서 삭제
        await self.db_adapter.clear_messages(
            session_id=self.session_id,
            workspace_id=self.workspace_id
        )

        # ✅ 2. Redis 캐시 삭제
        if self.redis:
            try:
                key = self._make_redis_key()
                self.redis.delete(key)
                logger.debug(f"🗑️ Cleared Redis cache: {key}")
            except Exception as e:
                logger.warning(f"Redis delete failed: {e}")

    async def _cache_messages_to_redis(self, messages: list[Message]):
        """메시지들을 Redis에 캐싱"""
        if not self.redis:
            return

        key = self._make_redis_key()

        # 기존 캐시 삭제
        self.redis.delete(key)

        # 새로 저장
        for message in messages:
            msg_json = json.dumps(message.to_dict())
            self.redis.rpush(key, msg_json)

        # TTL 설정
        if self.redis_ttl:
            self.redis.expire(key, self.redis_ttl)


def create_database_history_manager(
    db_adapter: DatabaseAdapter,
    redis_client: Any | None = None,
    redis_ttl: int = 3600,
    default_workspace_id: str | None = None
):
    """DatabaseChatHistory 매니저 팩토리

    Example:
        from src.flow.memory_db import create_database_history_manager, SQLAlchemyAdapter

        db_adapter = SQLAlchemyAdapter(db=db_session)

        history_manager = create_database_history_manager(
            db_adapter=db_adapter,
            redis_client=redis_client,
            redis_ttl=3600,
            default_workspace_id=workspace_id
        )

        # WorkflowContext에 주입
        workflow_context = BackendWorkflowContext(
            ...
            history_manager=history_manager
        )
    """
    from .memory import SessionHistoryManager

    def create_history(session_id: str, workspace_id: str | None = None) -> DatabaseChatHistory:
        return DatabaseChatHistory(
            session_id=session_id,
            db_adapter=db_adapter,
            redis_client=redis_client,
            workspace_id=workspace_id or default_workspace_id,
            redis_ttl=redis_ttl
        )

    return SessionHistoryManager(history_factory=create_history)


__all__ = [
    "DatabaseAdapter",
    "DatabaseChatHistory",
    "create_database_history_manager",
]
