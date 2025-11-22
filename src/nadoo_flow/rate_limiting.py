"""
Rate Limiting for Nadoo Flow
속도 제한 - Token Bucket 알고리즘
"""

import asyncio
import time
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TokenBucket:
    """Token Bucket 알고리즘 구현

    시간 기반 속도 제한을 위한 토큰 버킷 알고리즘입니다.

    Example:
        bucket = TokenBucket(
            requests_per_second=0.5,  # 2초에 1회
            max_bucket_size=10  # 버스트 허용
        )

        if await bucket.acquire():
            # API 호출
            pass
    """

    requests_per_second: float
    """초당 요청 수 (0.5 = 2초에 1회)"""

    max_bucket_size: int = 10
    """최대 버킷 크기 (버스트 허용)"""

    check_every_n_seconds: float = 0.1
    """체크 간격 (초)"""

    _tokens: float = field(default=0.0, init=False)
    """현재 토큰 수"""

    _last_update: float = field(default_factory=time.time, init=False)
    """마지막 업데이트 시간"""

    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)
    """동시성 제어를 위한 락"""

    def __post_init__(self):
        """초기화"""
        self._tokens = float(self.max_bucket_size)

    def _refill_tokens(self):
        """토큰 리필"""
        now = time.time()
        time_passed = now - self._last_update

        # 시간에 비례하여 토큰 추가
        tokens_to_add = time_passed * self.requests_per_second
        self._tokens = min(self._tokens + tokens_to_add, self.max_bucket_size)

        self._last_update = now

    async def acquire(self, tokens: int = 1, timeout: float | None = None) -> bool:
        """토큰 획득 (비동기)

        Args:
            tokens: 필요한 토큰 수
            timeout: 타임아웃 (초), None이면 무제한 대기

        Returns:
            성공 여부
        """
        start_time = time.time()

        async with self._lock:
            while True:
                self._refill_tokens()

                # 토큰이 충분하면 획득
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return True

                # 타임아웃 체크
                if timeout is not None:
                    elapsed = time.time() - start_time
                    if elapsed >= timeout:
                        return False

                # 대기 시간 계산
                tokens_needed = tokens - self._tokens
                wait_time = tokens_needed / self.requests_per_second
                wait_time = min(wait_time, self.check_every_n_seconds)

                # 짧은 시간 대기
                await asyncio.sleep(wait_time)

    def try_acquire(self, tokens: int = 1) -> bool:
        """토큰 획득 시도 (대기 없음)

        Args:
            tokens: 필요한 토큰 수

        Returns:
            성공 여부
        """
        self._refill_tokens()

        if self._tokens >= tokens:
            self._tokens -= tokens
            return True

        return False

    def get_available_tokens(self) -> float:
        """현재 사용 가능한 토큰 수"""
        self._refill_tokens()
        return self._tokens

    def get_wait_time(self, tokens: int = 1) -> float:
        """토큰 획득까지 대기 시간 (초)"""
        self._refill_tokens()

        if self._tokens >= tokens:
            return 0.0

        tokens_needed = tokens - self._tokens
        return tokens_needed / self.requests_per_second


class RateLimiter:
    """Rate Limiter

    여러 토큰 버킷을 관리하는 Rate Limiter입니다.

    Example:
        limiter = RateLimiter(
            requests_per_second=0.5,
            max_bucket_size=10
        )

        async with limiter:
            # API 호출
            pass
    """

    def __init__(
        self,
        requests_per_second: float,
        max_bucket_size: int = 10,
        check_every_n_seconds: float = 0.1
    ):
        """
        Args:
            requests_per_second: 초당 요청 수
            max_bucket_size: 최대 버킷 크기
            check_every_n_seconds: 체크 간격
        """
        self.bucket = TokenBucket(
            requests_per_second=requests_per_second,
            max_bucket_size=max_bucket_size,
            check_every_n_seconds=check_every_n_seconds
        )

    async def acquire(self, tokens: int = 1, timeout: float | None = None) -> bool:
        """토큰 획득"""
        return await self.bucket.acquire(tokens, timeout)

    async def __aenter__(self):
        """Async context manager 진입"""
        await self.bucket.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager 종료"""
        pass


class MultiTenantRateLimiter:
    """멀티 테넌트 Rate Limiter

    여러 테넌트(사용자, 그룹, 워크스페이스 등)별로 속도 제한을 적용합니다.

    Example:
        limiter = MultiTenantRateLimiter(
            default_requests_per_second=1.0,
            tenant_limits={
                "premium_user": 10.0,
                "free_user": 0.1
            }
        )

        async with limiter.limit("user_123"):
            # API 호출
            pass
    """

    def __init__(
        self,
        default_requests_per_second: float = 1.0,
        default_max_bucket_size: int = 10,
        tenant_limits: dict[str, float] | None = None
    ):
        """
        Args:
            default_requests_per_second: 기본 초당 요청 수
            default_max_bucket_size: 기본 버킷 크기
            tenant_limits: 테넌트별 제한 설정 (key: tenant_id, value: requests_per_second)
        """
        self.default_requests_per_second = default_requests_per_second
        self.default_max_bucket_size = default_max_bucket_size
        self.tenant_limits = tenant_limits or {}
        self._buckets: dict[str, TokenBucket] = {}
        self._lock = asyncio.Lock()

    async def _get_bucket(self, tenant_id: str) -> TokenBucket:
        """테넌트에 대한 토큰 버킷 가져오기"""
        if tenant_id not in self._buckets:
            async with self._lock:
                # Double-check locking
                if tenant_id not in self._buckets:
                    requests_per_second = self.tenant_limits.get(
                        tenant_id,
                        self.default_requests_per_second
                    )

                    self._buckets[tenant_id] = TokenBucket(
                        requests_per_second=requests_per_second,
                        max_bucket_size=self.default_max_bucket_size
                    )

        return self._buckets[tenant_id]

    async def acquire(
        self,
        tenant_id: str,
        tokens: int = 1,
        timeout: float | None = None
    ) -> bool:
        """테넌트에 대한 토큰 획득

        Args:
            tenant_id: 테넌트 식별자 (사용자 ID, 워크스페이스 ID 등)
            tokens: 필요한 토큰 수
            timeout: 타임아웃 (초)

        Returns:
            성공 여부
        """
        bucket = await self._get_bucket(tenant_id)
        success = await bucket.acquire(tokens, timeout)

        if not success:
            logger.warning(
                f"Rate limit exceeded for tenant {tenant_id}"
            )

        return success

    def limit(self, tenant_id: str):
        """Async context manager를 반환

        Args:
            tenant_id: 테넌트 식별자

        Returns:
            Context manager
        """
        return _TenantLimitContext(self, tenant_id)

    def set_tenant_limit(self, tenant_id: str, requests_per_second: float):
        """테넌트 제한 설정

        Args:
            tenant_id: 테넌트 식별자
            requests_per_second: 초당 요청 수
        """
        self.tenant_limits[tenant_id] = requests_per_second

        # 기존 버킷 제거 (다음 요청 시 새로 생성)
        if tenant_id in self._buckets:
            del self._buckets[tenant_id]


class _TenantLimitContext:
    """Multi-Tenant Rate Limiter Context Manager"""

    def __init__(self, limiter: MultiTenantRateLimiter, tenant_id: str):
        self.limiter = limiter
        self.tenant_id = tenant_id

    async def __aenter__(self):
        await self.limiter.acquire(self.tenant_id)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


class RateLimitedNode:
    """Rate Limiting이 적용된 노드 믹스인

    노드에 속도 제한 기능을 추가합니다.

    Example:
        class MyLLMNode(BaseNode, RateLimitedNode):
            def __init__(self):
                BaseNode.__init__(self, ...)
                RateLimitedNode.__init__(
                    self,
                    rate_limiter=RateLimiter(requests_per_second=0.5)
                )

            async def execute(self, node_context, workflow_context):
                # 자동으로 속도 제한 적용
                async with self.rate_limiter:
                    # API 호출
                    pass
    """

    def __init__(
        self,
        rate_limiter: RateLimiter | None = None,
        requests_per_second: float = 1.0
    ):
        """
        Args:
            rate_limiter: 사용할 Rate Limiter (None이면 기본 생성)
            requests_per_second: 초당 요청 수 (rate_limiter가 None일 때)
        """
        if rate_limiter is None:
            rate_limiter = RateLimiter(requests_per_second=requests_per_second)

        self.rate_limiter = rate_limiter
        self._rate_limiting_enabled = True

    def enable_rate_limiting(self):
        """속도 제한 활성화"""
        self._rate_limiting_enabled = True

    def disable_rate_limiting(self):
        """속도 제한 비활성화"""
        self._rate_limiting_enabled = False

    def is_rate_limiting_enabled(self) -> bool:
        """속도 제한 활성화 여부"""
        return self._rate_limiting_enabled


# Redis 기반 분산 Rate Limiter (optional)
try:
    import redis
    from redis import Redis

    class RedisRateLimiter:
        """Redis 기반 분산 Rate Limiter

        여러 인스턴스에서 공유되는 속도 제한을 구현합니다.

        Example:
            limiter = RedisRateLimiter(
                redis_client=redis.Redis(),
                key_prefix="rate_limit:",
                requests_per_window=100,
                window_seconds=60
            )

            if await limiter.check_and_increment("user_123"):
                # API 호출
                pass
        """

        def __init__(
            self,
            redis_client: Redis,
            key_prefix: str = "rate_limit:",
            requests_per_window: int = 100,
            window_seconds: int = 60
        ):
            """
            Args:
                redis_client: Redis 클라이언트
                key_prefix: Redis 키 접두사
                requests_per_window: 시간 윈도우당 요청 수
                window_seconds: 시간 윈도우 (초)
            """
            self.redis = redis_client
            self.key_prefix = key_prefix
            self.requests_per_window = requests_per_window
            self.window_seconds = window_seconds

        def _make_key(self, identifier: str) -> str:
            """Rate limit 키 생성"""
            return f"{self.key_prefix}{identifier}"

        async def check_and_increment(self, identifier: str) -> bool:
            """속도 제한 체크 및 카운터 증가

            Args:
                identifier: 사용자/워크스페이스 ID

            Returns:
                허용 여부 (True = 허용, False = 제한 초과)
            """
            key = self._make_key(identifier)

            # Lua 스크립트로 원자적 실행
            lua_script = """
            local current = redis.call('GET', KEYS[1])
            if current and tonumber(current) >= tonumber(ARGV[1]) then
                return 0
            end
            redis.call('INCR', KEYS[1])
            redis.call('EXPIRE', KEYS[1], ARGV[2])
            return 1
            """

            result = self.redis.eval(
                lua_script,
                1,
                key,
                self.requests_per_window,
                self.window_seconds
            )

            return result == 1

        def get_remaining(self, identifier: str) -> int:
            """남은 요청 수"""
            key = self._make_key(identifier)
            current = self.redis.get(key)

            if current is None:
                return self.requests_per_window

            return max(0, self.requests_per_window - int(current))

        def reset(self, identifier: str):
            """카운터 리셋"""
            key = self._make_key(identifier)
            self.redis.delete(key)

except ImportError:
    logger.debug("Redis package not available, RedisRateLimiter disabled")
    RedisRateLimiter = None  # type: ignore
