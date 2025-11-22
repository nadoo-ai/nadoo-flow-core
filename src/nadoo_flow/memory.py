"""
Memory Management for Nadoo Flow
메모리 관리 - 채팅 히스토리, 세션 관리
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

from .prompts import Message

logger = logging.getLogger(__name__)


class BaseChatHistory(ABC):
    """채팅 히스토리 베이스 클래스

    모든 히스토리 구현은 이 인터페이스를 따라야 합니다.
    """

    @abstractmethod
    async def get_messages(self) -> list[Message]:
        """메시지 조회"""
        pass

    @abstractmethod
    async def add_message(self, message: Message):
        """메시지 추가"""
        pass

    @abstractmethod
    async def add_messages(self, messages: list[Message]):
        """여러 메시지 추가"""
        pass

    @abstractmethod
    async def clear(self):
        """히스토리 초기화"""
        pass


class InMemoryChatHistory(BaseChatHistory):
    """인메모리 채팅 히스토리

    메모리에 메시지를 저장합니다. 프로세스 재시작 시 초기화됩니다.

    Example:
        history = InMemoryChatHistory()
        await history.add_message(Message.user("Hello"))
        messages = await history.get_messages()
    """

    def __init__(self):
        self.messages: list[Message] = []

    async def get_messages(self) -> list[Message]:
        """메시지 조회"""
        return self.messages.copy()

    async def add_message(self, message: Message):
        """메시지 추가"""
        self.messages.append(message)

    async def add_messages(self, messages: list[Message]):
        """여러 메시지 추가"""
        self.messages.extend(messages)

    async def clear(self):
        """히스토리 초기화"""
        self.messages.clear()


class SlidingWindowChatHistory(BaseChatHistory):
    """슬라이딩 윈도우 채팅 히스토리

    최근 N개 메시지만 유지합니다 (토큰 제한 관리).

    Example:
        history = SlidingWindowChatHistory(
            base_history=InMemoryChatHistory(),
            window_size=10
        )
    """

    def __init__(
        self,
        base_history: BaseChatHistory,
        window_size: int = 10
    ):
        """
        Args:
            base_history: 기본 히스토리 저장소
            window_size: 윈도우 크기 (최근 N개 메시지)
        """
        self.base_history = base_history
        self.window_size = window_size

    async def get_messages(self) -> list[Message]:
        """최근 N개 메시지만 반환"""
        all_messages = await self.base_history.get_messages()
        return all_messages[-self.window_size:] if all_messages else []

    async def add_message(self, message: Message):
        """메시지 추가"""
        await self.base_history.add_message(message)

    async def add_messages(self, messages: list[Message]):
        """여러 메시지 추가"""
        await self.base_history.add_messages(messages)

    async def clear(self):
        """히스토리 초기화"""
        await self.base_history.clear()


# Redis 기반 채팅 히스토리 (optional)
try:
    import json
    import redis
    from redis import Redis

    class RedisChatHistory(BaseChatHistory):
        """Redis 기반 채팅 히스토리

        분산 환경에서 사용 가능한 Redis 히스토리입니다.

        Example:
            history = RedisChatHistory(
                session_id="user_123",
                redis_client=redis.Redis()
            )
        """

        def __init__(
            self,
            session_id: str,
            redis_client: Redis,
            key_prefix: str = "chat_history:",
            ttl: int | None = None
        ):
            """
            Args:
                session_id: 세션 ID
                redis_client: Redis 클라이언트
                key_prefix: Redis 키 접두사
                ttl: TTL (초), None이면 무제한
            """
            self.session_id = session_id
            self.redis = redis_client
            self.key_prefix = key_prefix
            self.ttl = ttl

        def _make_key(self) -> str:
            """Redis 키 생성"""
            return f"{self.key_prefix}{self.session_id}"

        async def get_messages(self) -> list[Message]:
            """메시지 조회"""
            key = self._make_key()
            messages_json = self.redis.lrange(key, 0, -1)

            messages = []
            for msg_json in messages_json:
                msg_data = json.loads(msg_json)
                messages.append(Message(**msg_data))

            return messages

        async def add_message(self, message: Message):
            """메시지 추가"""
            key = self._make_key()
            msg_json = json.dumps(message.to_dict())

            self.redis.rpush(key, msg_json)

            if self.ttl:
                self.redis.expire(key, self.ttl)

        async def add_messages(self, messages: list[Message]):
            """여러 메시지 추가"""
            if not messages:
                return

            key = self._make_key()
            messages_json = [json.dumps(msg.to_dict()) for msg in messages]

            self.redis.rpush(key, *messages_json)

            if self.ttl:
                self.redis.expire(key, self.ttl)

        async def clear(self):
            """히스토리 초기화"""
            key = self._make_key()
            self.redis.delete(key)

except ImportError:
    logger.debug("Redis package not available, RedisChatHistory disabled")
    RedisChatHistory = None  # type: ignore


class SessionHistoryManager:
    """세션별 히스토리 관리자

    여러 세션의 히스토리를 관리합니다.

    Example:
        manager = SessionHistoryManager()

        # 세션별 히스토리 가져오기
        history = await manager.get_history("user_123")
        await history.add_message(Message.user("Hello"))
    """

    def __init__(
        self,
        history_factory: Callable[[str], BaseChatHistory],
        window_size: int | None = None
    ):
        """
        Args:
            history_factory: 세션 ID를 받아 히스토리를 생성하는 함수
            window_size: 슬라이딩 윈도우 크기 (None이면 무제한)
        """
        self.history_factory = history_factory
        self.window_size = window_size
        self._histories: dict[str, BaseChatHistory] = {}

    async def get_history(self, session_id: str) -> BaseChatHistory:
        """세션 히스토리 가져오기

        Args:
            session_id: 세션 ID

        Returns:
            채팅 히스토리
        """
        if session_id not in self._histories:
            base_history = self.history_factory(session_id)

            if self.window_size:
                history = SlidingWindowChatHistory(
                    base_history=base_history,
                    window_size=self.window_size
                )
            else:
                history = base_history

            self._histories[session_id] = history

        return self._histories[session_id]

    async def clear_history(self, session_id: str):
        """세션 히스토리 초기화

        Args:
            session_id: 세션 ID
        """
        if session_id in self._histories:
            await self._histories[session_id].clear()
            del self._histories[session_id]


class ChatHistoryNode:
    """채팅 히스토리 믹스인

    노드에 채팅 히스토리 기능을 추가합니다.

    Example:
        class MyChatNode(BaseNode, ChatHistoryNode):
            def __init__(self):
                BaseNode.__init__(self, ...)
                ChatHistoryNode.__init__(
                    self,
                    history_manager=manager,
                    session_key="session_id"
                )

            async def execute(self, node_context, workflow_context):
                # 히스토리 자동 로딩
                history = await self.get_history(workflow_context)

                # 메시지 사용
                messages = await history.get_messages()

                # 새 메시지 추가
                await history.add_message(Message.assistant("Response"))
    """

    def __init__(
        self,
        history_manager: SessionHistoryManager,
        session_key: str = "session_id",
        auto_save: bool = True
    ):
        """
        Args:
            history_manager: 히스토리 관리자
            session_key: WorkflowContext에서 세션 ID를 가져올 키
            auto_save: 자동 저장 여부
        """
        self.history_manager = history_manager
        self.session_key = session_key
        self.auto_save = auto_save

    async def get_history(self, workflow_context: Any) -> BaseChatHistory:
        """워크플로우 컨텍스트에서 히스토리 가져오기

        Args:
            workflow_context: 워크플로우 컨텍스트

        Returns:
            채팅 히스토리
        """
        # 세션 ID 추출
        session_id = getattr(workflow_context, self.session_key, None)

        if not session_id:
            # 기본 세션 ID
            session_id = workflow_context.workflow_id

        return await self.history_manager.get_history(session_id)

    async def save_interaction(
        self,
        workflow_context: Any,
        user_message: str,
        assistant_message: str
    ):
        """대화 상호작용 저장

        Args:
            workflow_context: 워크플로우 컨텍스트
            user_message: 사용자 메시지
            assistant_message: 어시스턴트 응답
        """
        history = await self.get_history(workflow_context)

        await history.add_messages([
            Message.user(user_message),
            Message.assistant(assistant_message)
        ])


# 편의 함수들
def create_inmemory_history_manager(window_size: int | None = None) -> SessionHistoryManager:
    """인메모리 히스토리 매니저 생성

    Args:
        window_size: 슬라이딩 윈도우 크기

    Returns:
        SessionHistoryManager
    """
    def factory(session_id: str) -> BaseChatHistory:
        return InMemoryChatHistory()

    return SessionHistoryManager(
        history_factory=factory,
        window_size=window_size
    )


def create_redis_history_manager(
    redis_client: Any,
    window_size: int | None = None,
    ttl: int | None = None
) -> SessionHistoryManager:
    """Redis 히스토리 매니저 생성

    Args:
        redis_client: Redis 클라이언트
        window_size: 슬라이딩 윈도우 크기
        ttl: TTL (초)

    Returns:
        SessionHistoryManager
    """
    if RedisChatHistory is None:
        raise ImportError("Redis package is required for RedisChatHistory")

    def factory(session_id: str) -> BaseChatHistory:
        return RedisChatHistory(
            session_id=session_id,
            redis_client=redis_client,
            ttl=ttl
        )

    return SessionHistoryManager(
        history_factory=factory,
        window_size=window_size
    )
