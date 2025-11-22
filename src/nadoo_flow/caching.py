"""
Caching mechanisms for Nadoo Flow
캐싱 메커니즘 - LLM 응답 캐싱, 비용 절감
"""

import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """캐시 엔트리

    캐시에 저장되는 데이터 구조
    """

    key: str
    """캐시 키"""

    value: Any
    """캐시된 값"""

    created_at: float
    """생성 시간 (Unix timestamp)"""

    ttl: Optional[float] = None
    """TTL (초 단위, None이면 무제한)"""

    metadata: dict[str, Any] | None = None
    """메타데이터"""

    def is_expired(self) -> bool:
        """만료 여부 확인"""
        if self.ttl is None:
            return False

        return (time.time() - self.created_at) > self.ttl


class BaseCache(ABC):
    """캐시 베이스 클래스

    모든 캐시 구현은 이 인터페이스를 따라야 합니다.
    """

    @abstractmethod
    def get(self, key: str) -> Any | None:
        """캐시에서 값 조회"""
        pass

    @abstractmethod
    def set(self, key: str, value: Any, ttl: float | None = None):
        """캐시에 값 저장"""
        pass

    @abstractmethod
    def delete(self, key: str):
        """캐시에서 값 삭제"""
        pass

    @abstractmethod
    def clear(self):
        """캐시 전체 삭제"""
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """키 존재 여부"""
        pass


class InMemoryCache(BaseCache):
    """인메모리 캐시

    메모리에 캐시를 저장합니다. 프로세스 재시작 시 초기화됩니다.

    Example:
        cache = InMemoryCache(default_ttl=3600)
        cache.set("key", "value")
        value = cache.get("key")
    """

    def __init__(self, default_ttl: float | None = None):
        """
        Args:
            default_ttl: 기본 TTL (초), None이면 무제한
        """
        self.default_ttl = default_ttl
        self._cache: dict[str, CacheEntry] = {}

    def get(self, key: str) -> Any | None:
        """캐시에서 값 조회"""
        entry = self._cache.get(key)

        if entry is None:
            return None

        # 만료 확인
        if entry.is_expired():
            self.delete(key)
            return None

        return entry.value

    def set(self, key: str, value: Any, ttl: float | None = None):
        """캐시에 값 저장"""
        if ttl is None:
            ttl = self.default_ttl

        entry = CacheEntry(
            key=key,
            value=value,
            created_at=time.time(),
            ttl=ttl
        )

        self._cache[key] = entry

    def delete(self, key: str):
        """캐시에서 값 삭제"""
        self._cache.pop(key, None)

    def clear(self):
        """캐시 전체 삭제"""
        self._cache.clear()

    def exists(self, key: str) -> bool:
        """키 존재 여부 (만료 체크 포함)"""
        return self.get(key) is not None

    def cleanup_expired(self):
        """만료된 엔트리 정리"""
        expired_keys = [
            key for key, entry in self._cache.items()
            if entry.is_expired()
        ]

        for key in expired_keys:
            self.delete(key)

        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")


class ResponseCache:
    """LLM 응답 캐싱

    LLM 호출 결과를 캐싱하여 동일한 입력에 대해 API 호출을 생략합니다.

    Example:
        cache = ResponseCache(InMemoryCache(default_ttl=3600))

        # 캐시 키 생성
        cache_key = cache.make_key(
            prompt="What is AI?",
            model="gpt-4",
            temperature=0.7
        )

        # 캐시 조회
        cached = cache.get(cache_key)
        if cached:
            return cached

        # LLM 호출
        response = call_llm(...)

        # 캐시 저장
        cache.set(cache_key, response)
    """

    def __init__(
        self,
        cache: BaseCache,
        namespace: str = "nadoo_llm",
        include_model_params: bool = True
    ):
        """
        Args:
            cache: 사용할 캐시 백엔드
            namespace: 캐시 키 네임스페이스
            include_model_params: 캐시 키에 모델 파라미터 포함 여부
        """
        self.cache = cache
        self.namespace = namespace
        self.include_model_params = include_model_params

    def make_key(
        self,
        prompt: str | list[dict[str, Any]],
        model: str | None = None,
        **kwargs
    ) -> str:
        """캐시 키 생성

        Args:
            prompt: 프롬프트 (문자열 또는 메시지 리스트)
            model: 모델 이름
            **kwargs: 추가 파라미터 (temperature, max_tokens 등)

        Returns:
            캐시 키 문자열
        """
        # 키 재료 수집
        key_parts = [self.namespace]

        # 프롬프트 해시
        if isinstance(prompt, str):
            prompt_hash = self._hash_string(prompt)
        else:
            prompt_hash = self._hash_object(prompt)

        key_parts.append(prompt_hash)

        # 모델 이름
        if model:
            key_parts.append(model)

        # 모델 파라미터 (옵션)
        if self.include_model_params and kwargs:
            # temperature, max_tokens 등만 포함 (재현 가능한 파라미터)
            relevant_params = {
                k: v for k, v in kwargs.items()
                if k in ("temperature", "max_tokens", "top_p", "frequency_penalty", "presence_penalty")
            }
            if relevant_params:
                params_hash = self._hash_object(relevant_params)
                key_parts.append(params_hash)

        return ":".join(key_parts)

    def get(self, key: str) -> Any | None:
        """캐시 조회"""
        value = self.cache.get(key)

        if value is not None:
            logger.debug(f"Cache hit: {key}")
        else:
            logger.debug(f"Cache miss: {key}")

        return value

    def set(self, key: str, value: Any, ttl: float | None = None):
        """캐시 저장"""
        self.cache.set(key, value, ttl)
        logger.debug(f"Cache stored: {key}")

    def delete(self, key: str):
        """캐시 삭제"""
        self.cache.delete(key)

    def clear(self):
        """네임스페이스의 모든 캐시 삭제"""
        # InMemoryCache의 경우 prefix 필터링
        if isinstance(self.cache, InMemoryCache):
            keys_to_delete = [
                key for key in self.cache._cache.keys()
                if key.startswith(self.namespace + ":")
            ]
            for key in keys_to_delete:
                self.cache.delete(key)
        else:
            # 다른 캐시는 전체 삭제
            self.cache.clear()

    def _hash_string(self, text: str) -> str:
        """문자열 해시"""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def _hash_object(self, obj: Any) -> str:
        """객체 해시 (JSON 직렬화 후)"""
        json_str = json.dumps(obj, sort_keys=True, ensure_ascii=False)
        return self._hash_string(json_str)


class CachedNode:
    """캐싱 기능이 있는 노드 믹스인

    노드에 캐싱 기능을 추가합니다.

    Example:
        class MyLLMNode(BaseNode, CachedNode):
            def __init__(self):
                BaseNode.__init__(self, ...)
                CachedNode.__init__(
                    self,
                    cache=ResponseCache(InMemoryCache())
                )

            async def execute(self, node_context, workflow_context):
                # 캐시 키 생성
                cache_key = self.response_cache.make_key(
                    prompt=node_context.get_input("prompt"),
                    model="gpt-4"
                )

                # 캐시 조회
                cached = self.response_cache.get(cache_key)
                if cached:
                    return NodeResult(success=True, output=cached)

                # LLM 호출
                result = await self._call_llm(...)

                # 캐시 저장
                self.response_cache.set(cache_key, result.output)

                return result
    """

    def __init__(self, response_cache: ResponseCache):
        """
        Args:
            response_cache: 사용할 응답 캐시
        """
        self.response_cache = response_cache
        self._cache_enabled = True

    def enable_cache(self):
        """캐시 활성화"""
        self._cache_enabled = True

    def disable_cache(self):
        """캐시 비활성화"""
        self._cache_enabled = False

    def is_cache_enabled(self) -> bool:
        """캐시 활성화 여부"""
        return self._cache_enabled

    def clear_cache(self):
        """이 노드의 캐시 삭제"""
        self.response_cache.clear()


# Redis 캐시 (optional, redis 패키지 필요)
try:
    import redis
    from redis import Redis

    class RedisCache(BaseCache):
        """Redis 기반 캐시

        분산 환경에서 사용 가능한 Redis 캐시입니다.

        Example:
            cache = RedisCache(
                host="localhost",
                port=6379,
                db=0,
                default_ttl=3600
            )

            cache.set("key", "value")
            value = cache.get("key")
        """

        def __init__(
            self,
            host: str = "localhost",
            port: int = 6379,
            db: int = 0,
            password: str | None = None,
            default_ttl: float | None = None,
            prefix: str = "nadoo:"
        ):
            """
            Args:
                host: Redis 호스트
                port: Redis 포트
                db: Redis DB 번호
                password: Redis 비밀번호
                default_ttl: 기본 TTL (초)
                prefix: 키 접두사
            """
            self.redis = Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=False
            )
            self.default_ttl = default_ttl
            self.prefix = prefix

        def _make_key(self, key: str) -> str:
            """접두사가 포함된 키 생성"""
            return f"{self.prefix}{key}"

        def get(self, key: str) -> Any | None:
            """캐시 조회"""
            full_key = self._make_key(key)
            value = self.redis.get(full_key)

            if value is None:
                return None

            # JSON 역직렬화
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                # JSON이 아니면 원본 반환
                return value

        def set(self, key: str, value: Any, ttl: float | None = None):
            """캐시 저장"""
            full_key = self._make_key(key)

            if ttl is None:
                ttl = self.default_ttl

            # JSON 직렬화
            try:
                serialized = json.dumps(value, ensure_ascii=False)
            except (TypeError, ValueError):
                # JSON 직렬화 불가능하면 문자열로
                serialized = str(value)

            if ttl:
                self.redis.setex(full_key, int(ttl), serialized)
            else:
                self.redis.set(full_key, serialized)

        def delete(self, key: str):
            """캐시 삭제"""
            full_key = self._make_key(key)
            self.redis.delete(full_key)

        def clear(self):
            """접두사로 시작하는 모든 키 삭제"""
            pattern = f"{self.prefix}*"
            keys = self.redis.keys(pattern)

            if keys:
                self.redis.delete(*keys)

        def exists(self, key: str) -> bool:
            """키 존재 여부"""
            full_key = self._make_key(key)
            return self.redis.exists(full_key) > 0

except ImportError:
    # Redis 패키지가 없으면 RedisCache 미제공
    logger.debug("Redis package not available, RedisCache disabled")
    RedisCache = None  # type: ignore
