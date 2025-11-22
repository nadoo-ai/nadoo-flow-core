"""
Resilience mechanisms for Nadoo Flow
복원력 메커니즘 - Retry, Fallback 등
"""

import asyncio
import random
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Type

from .base import BaseNode, NodeContext, NodeResult, WorkflowContext

logger = logging.getLogger(__name__)


@dataclass
class RetryPolicy:
    """재시도 정책 설정

    Example:
        policy = RetryPolicy(
            max_attempts=5,
            initial_delay=1.0,
            max_delay=60.0,
            exponential_base=2.0,
            jitter=1.0,
            retry_on_exceptions=(TimeoutError, ConnectionError)
        )
    """

    max_attempts: int = 3
    """최대 재시도 횟수"""

    initial_delay: float = 1.0
    """초기 대기 시간 (초)"""

    max_delay: float = 60.0
    """최대 대기 시간 (초)"""

    exponential_base: float = 2.0
    """지수 백오프 베이스 (2.0 = 매번 2배)"""

    jitter: float = 1.0
    """지터 범위 (0~jitter초 랜덤 추가)"""

    retry_on_exceptions: tuple[Type[Exception], ...] = (Exception,)
    """재시도할 예외 타입들"""

    retry_on_status: tuple[str, ...] = ()
    """재시도할 노드 상태들 (예: "timeout", "rate_limit")"""


class RetryableNode(BaseNode):
    """재시도 기능이 있는 노드

    지수 백오프와 지터를 사용한 자동 재시도 기능을 제공합니다.

    Example:
        class MyLLMNode(RetryableNode):
            def __init__(self):
                super().__init__(
                    node_id="llm",
                    node_type="llm",
                    name="LLM Node",
                    config={},
                    retry_policy=RetryPolicy(
                        max_attempts=5,
                        retry_on_exceptions=(TimeoutError, ConnectionError)
                    )
                )

            async def _execute_with_retry(self, node_context, workflow_context):
                # LLM 호출 로직
                return NodeResult(success=True, output={...})
    """

    def __init__(
        self,
        node_id: str,
        node_type: str,
        name: str,
        config: dict[str, Any],
        retry_policy: RetryPolicy | None = None
    ):
        super().__init__(node_id, node_type, name, config)
        self.retry_policy = retry_policy or RetryPolicy()

    async def execute(self, node_context: NodeContext, workflow_context: WorkflowContext) -> NodeResult:
        """재시도 로직이 적용된 실행"""
        last_error = None
        retry_metadata = {
            "total_attempts": 0,
            "retry_delays": [],
            "errors": []
        }

        for attempt in range(self.retry_policy.max_attempts):
            retry_metadata["total_attempts"] = attempt + 1

            try:
                # 실제 노드 로직 실행
                result = await self._execute_with_retry(node_context, workflow_context)

                # 성공 시 메타데이터 추가
                if result.success:
                    if attempt > 0:
                        result.metadata["retry_info"] = retry_metadata
                        logger.info(
                            f"Node {self.node_id} succeeded after {attempt + 1} attempts"
                        )
                    return result

                # 실패했지만 특정 상태에서만 재시도
                if self.retry_policy.retry_on_status:
                    status_code = result.metadata.get("status_code")
                    if status_code not in self.retry_policy.retry_on_status:
                        logger.info(
                            f"Node {self.node_id} failed with non-retryable status: {status_code}"
                        )
                        return result

                # 재시도 가능한 실패
                last_error = result.error
                retry_metadata["errors"].append(result.error or "Unknown error")

            except self.retry_policy.retry_on_exceptions as e:
                last_error = str(e)
                retry_metadata["errors"].append(str(e))
                logger.warning(
                    f"Node {self.node_id} attempt {attempt + 1} failed: {e}"
                )
            except Exception as e:
                # 재시도하지 않을 예외는 즉시 실패
                logger.error(f"Node {self.node_id} failed with non-retryable error: {e}")
                return NodeResult(
                    success=False,
                    error=str(e),
                    metadata={"retry_info": retry_metadata}
                )

            # 마지막 시도였으면 실패 반환
            if attempt == self.retry_policy.max_attempts - 1:
                logger.error(
                    f"Node {self.node_id} failed after {self.retry_policy.max_attempts} attempts"
                )
                return NodeResult(
                    success=False,
                    error=f"Max retries exceeded. Last error: {last_error}",
                    metadata={"retry_info": retry_metadata}
                )

            # 지수 백오프 계산
            delay = self._calculate_delay(attempt)
            retry_metadata["retry_delays"].append(delay)

            logger.info(
                f"Node {self.node_id} retrying in {delay:.2f}s "
                f"(attempt {attempt + 1}/{self.retry_policy.max_attempts})"
            )

            await asyncio.sleep(delay)

        # 여기 도달하면 안 되지만 안전장치
        return NodeResult(
            success=False,
            error=f"Unexpected retry loop exit: {last_error}",
            metadata={"retry_info": retry_metadata}
        )

    async def _execute_with_retry(
        self,
        node_context: NodeContext,
        workflow_context: WorkflowContext
    ) -> NodeResult:
        """하위 클래스에서 구현할 실제 실행 로직

        이 메서드는 재시도 로직 없이 순수한 노드 로직만 구현하면 됩니다.
        """
        raise NotImplementedError("Subclasses must implement _execute_with_retry")

    def _calculate_delay(self, attempt: int) -> float:
        """지수 백오프 + 지터로 대기 시간 계산"""
        # 지수 백오프
        delay = self.retry_policy.initial_delay * (
            self.retry_policy.exponential_base ** attempt
        )

        # 최대 대기 시간 제한
        delay = min(delay, self.retry_policy.max_delay)

        # 지터 추가 (0 ~ jitter 사이 랜덤)
        if self.retry_policy.jitter > 0:
            delay += random.uniform(0, self.retry_policy.jitter)

        return delay


class FallbackNode(BaseNode):
    """여러 노드를 순차 시도하는 폴백 노드

    첫 번째 노드부터 시도하여 성공할 때까지 다음 노드로 넘어갑니다.
    모든 노드가 실패하면 마지막 에러를 반환합니다.

    Example:
        fallback = FallbackNode(
            node_id="llm_fallback",
            nodes=[
                GPT4Node(),      # 1순위: GPT-4
                ClaudeNode(),    # 2순위: Claude
                LlamaNode()      # 3순위: Local Llama
            ],
            handle_exceptions=(TimeoutError, ConnectionError)
        )
    """

    def __init__(
        self,
        node_id: str,
        nodes: list[BaseNode],
        handle_exceptions: tuple[Type[Exception], ...] = (Exception,),
        pass_error_context: bool = True,
        name: str = "Fallback",
        config: dict[str, Any] | None = None
    ):
        """
        Args:
            node_id: 노드 ID
            nodes: 시도할 노드 리스트 (순서대로)
            handle_exceptions: 폴백할 예외 타입들
            pass_error_context: 이전 에러를 다음 노드에 전달할지 여부
            name: 노드 이름
            config: 노드 설정
        """
        super().__init__(
            node_id=node_id,
            node_type="fallback",
            name=name,
            config=config or {}
        )
        self.fallback_nodes = nodes
        self.handle_exceptions = handle_exceptions
        self.pass_error_context = pass_error_context

    async def execute(self, node_context: NodeContext, workflow_context: WorkflowContext) -> NodeResult:
        """폴백 로직으로 노드들 순차 실행"""
        if not self.fallback_nodes:
            return NodeResult(
                success=False,
                error="No fallback nodes configured"
            )

        errors = []
        fallback_metadata = {
            "attempted_nodes": [],
            "errors": []
        }

        for idx, node in enumerate(self.fallback_nodes):
            fallback_metadata["attempted_nodes"].append({
                "node_id": node.node_id,
                "node_type": node.node_type,
                "attempt_index": idx
            })

            try:
                logger.info(
                    f"Fallback node {self.node_id}: trying {node.node_id} "
                    f"(option {idx + 1}/{len(self.fallback_nodes)})"
                )

                # 이전 에러 컨텍스트 전달
                if self.pass_error_context and errors:
                    node_context.metadata["previous_errors"] = errors

                # 노드 실행
                result = await node.execute(node_context, workflow_context)

                # 성공 시 메타데이터 추가하고 반환
                if result.success:
                    result.metadata["fallback_info"] = {
                        **fallback_metadata,
                        "successful_node": node.node_id,
                        "fallback_index": idx,
                        "total_attempts": idx + 1
                    }

                    if idx > 0:
                        logger.info(
                            f"Fallback node {self.node_id} succeeded with "
                            f"{node.node_id} (fallback option {idx + 1})"
                        )

                    return result

                # 실패한 경우 에러 기록
                error_msg = result.error or "Unknown error"
                errors.append({
                    "node_id": node.node_id,
                    "error": error_msg,
                    "metadata": result.metadata
                })
                fallback_metadata["errors"].append(error_msg)

                logger.warning(
                    f"Fallback option {idx + 1} ({node.node_id}) failed: {error_msg}"
                )

            except self.handle_exceptions as e:
                error_msg = str(e)
                errors.append({
                    "node_id": node.node_id,
                    "error": error_msg,
                    "exception_type": type(e).__name__
                })
                fallback_metadata["errors"].append(error_msg)

                logger.warning(
                    f"Fallback option {idx + 1} ({node.node_id}) raised exception: {e}"
                )
                continue

            except Exception as e:
                # 처리하지 않는 예외는 즉시 실패
                logger.error(
                    f"Fallback option {idx + 1} ({node.node_id}) "
                    f"raised non-retryable exception: {e}"
                )
                return NodeResult(
                    success=False,
                    error=f"Non-retryable exception in {node.node_id}: {e}",
                    metadata={"fallback_info": fallback_metadata}
                )

        # 모든 폴백이 실패
        logger.error(
            f"All {len(self.fallback_nodes)} fallback options exhausted for {self.node_id}"
        )

        return NodeResult(
            success=False,
            error=f"All fallback options exhausted. Tried {len(self.fallback_nodes)} nodes.",
            metadata={
                "fallback_info": fallback_metadata,
                "all_errors": errors
            }
        )
