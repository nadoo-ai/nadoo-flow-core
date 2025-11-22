"""
Batch Processing for Nadoo Flow
배치 처리 - 대량 데이터 병렬 처리
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable

from .base import BaseNode, NodeContext, NodeResult, WorkflowContext

logger = logging.getLogger(__name__)


@dataclass
class BatchResult:
    """배치 처리 결과"""

    index: int
    """배치 인덱스"""

    input_data: dict[str, Any]
    """입력 데이터"""

    success: bool
    """성공 여부"""

    output: dict[str, Any] | None = None
    """출력 데이터"""

    error: str | None = None
    """에러 메시지"""

    execution_time: float | None = None
    """실행 시간 (초)"""


class BatchProcessor:
    """배치 프로세서

    대량의 데이터를 병렬로 처리합니다.

    Example:
        processor = BatchProcessor(
            node=my_llm_node,
            max_concurrency=5
        )

        results = await processor.batch([
            {"prompt": "Question 1"},
            {"prompt": "Question 2"},
            ...
        ])
    """

    def __init__(
        self,
        node: BaseNode,
        max_concurrency: int = 10,
        continue_on_error: bool = True
    ):
        """
        Args:
            node: 처리할 노드
            max_concurrency: 최대 동시 실행 수
            continue_on_error: 에러 발생 시 계속 진행 여부
        """
        self.node = node
        self.max_concurrency = max_concurrency
        self.continue_on_error = continue_on_error

    async def batch(
        self,
        inputs: list[dict[str, Any]],
        workflow_context: WorkflowContext | None = None
    ) -> list[BatchResult]:
        """배치 처리 실행

        Args:
            inputs: 입력 데이터 리스트
            workflow_context: 워크플로우 컨텍스트 (None이면 새로 생성)

        Returns:
            BatchResult 리스트
        """
        if workflow_context is None:
            workflow_context = WorkflowContext()

        # 세마포어로 동시성 제어
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def process_one(index: int, input_data: dict[str, Any]) -> BatchResult:
            """단일 입력 처리"""
            import time

            async with semaphore:
                start_time = time.time()

                try:
                    # 노드 컨텍스트 생성
                    node_context = NodeContext(
                        node_id=f"{self.node.node_id}_batch_{index}",
                        node_type=self.node.node_type,
                        input_data=input_data
                    )

                    # 노드 실행
                    result = await self.node.execute(node_context, workflow_context)

                    execution_time = time.time() - start_time

                    return BatchResult(
                        index=index,
                        input_data=input_data,
                        success=result.success,
                        output=result.output,
                        error=result.error,
                        execution_time=execution_time
                    )

                except Exception as e:
                    execution_time = time.time() - start_time

                    if not self.continue_on_error:
                        raise

                    logger.error(f"Batch item {index} failed: {e}")

                    return BatchResult(
                        index=index,
                        input_data=input_data,
                        success=False,
                        error=str(e),
                        execution_time=execution_time
                    )

        # 모든 입력 병렬 처리
        tasks = [
            process_one(i, input_data)
            for i, input_data in enumerate(inputs)
        ]

        results = await asyncio.gather(*tasks)

        return list(results)

    async def batch_as_completed(
        self,
        inputs: list[dict[str, Any]],
        workflow_context: WorkflowContext | None = None
    ) -> AsyncIterator[BatchResult]:
        """배치 처리 (완료되는 대로 yield)

        Args:
            inputs: 입력 데이터 리스트
            workflow_context: 워크플로우 컨텍스트

        Yields:
            완료된 BatchResult
        """
        if workflow_context is None:
            workflow_context = WorkflowContext()

        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def process_one(index: int, input_data: dict[str, Any]) -> BatchResult:
            """단일 입력 처리"""
            import time

            async with semaphore:
                start_time = time.time()

                try:
                    node_context = NodeContext(
                        node_id=f"{self.node.node_id}_batch_{index}",
                        node_type=self.node.node_type,
                        input_data=input_data
                    )

                    result = await self.node.execute(node_context, workflow_context)
                    execution_time = time.time() - start_time

                    return BatchResult(
                        index=index,
                        input_data=input_data,
                        success=result.success,
                        output=result.output,
                        error=result.error,
                        execution_time=execution_time
                    )

                except Exception as e:
                    execution_time = time.time() - start_time

                    if not self.continue_on_error:
                        raise

                    return BatchResult(
                        index=index,
                        input_data=input_data,
                        success=False,
                        error=str(e),
                        execution_time=execution_time
                    )

        # 태스크 생성
        tasks = [
            asyncio.create_task(process_one(i, input_data))
            for i, input_data in enumerate(inputs)
        ]

        # 완료되는 대로 yield
        for coro in asyncio.as_completed(tasks):
            result = await coro
            yield result


class MapNode(BaseNode):
    """Map 노드

    리스트의 각 항목에 대해 하위 노드를 실행합니다.

    Example:
        map_node = MapNode(
            node_id="map_translate",
            child_node=translate_node,
            input_key="texts",
            max_concurrency=5
        )
    """

    def __init__(
        self,
        node_id: str,
        child_node: BaseNode,
        input_key: str = "items",
        output_key: str = "results",
        max_concurrency: int = 10,
        name: str = "Map",
        config: dict[str, Any] | None = None
    ):
        """
        Args:
            node_id: 노드 ID
            child_node: 각 항목에 적용할 노드
            input_key: 입력 리스트 키
            output_key: 출력 리스트 키
            max_concurrency: 최대 동시 실행 수
            name: 노드 이름
            config: 노드 설정
        """
        super().__init__(
            node_id=node_id,
            node_type="map",
            name=name,
            config=config or {}
        )
        self.child_node = child_node
        self.input_key = input_key
        self.output_key = output_key
        self.max_concurrency = max_concurrency

    async def execute(
        self,
        node_context: NodeContext,
        workflow_context: WorkflowContext
    ) -> NodeResult:
        """Map 실행"""
        # 입력 리스트 가져오기
        items = node_context.get_input(self.input_key)

        if items is None:
            return NodeResult(
                success=False,
                error=f"Input key '{self.input_key}' not found"
            )

        if not isinstance(items, list):
            return NodeResult(
                success=False,
                error=f"Input '{self.input_key}' must be a list"
            )

        # 배치 프로세서로 처리
        processor = BatchProcessor(
            node=self.child_node,
            max_concurrency=self.max_concurrency
        )

        # 각 항목을 딕셔너리로 래핑
        batch_inputs = [{"item": item} for item in items]

        batch_results = await processor.batch(batch_inputs, workflow_context)

        # 결과 수집
        results = []
        errors = []

        for batch_result in batch_results:
            if batch_result.success and batch_result.output:
                results.append(batch_result.output)
            else:
                errors.append({
                    "index": batch_result.index,
                    "error": batch_result.error
                })

        # 에러가 있으면 실패
        if errors and not all(r.success for r in batch_results):
            return NodeResult(
                success=False,
                error=f"Some items failed: {len(errors)} errors",
                metadata={"errors": errors}
            )

        return NodeResult(
            success=True,
            output={
                self.output_key: results,
                "total": len(items),
                "succeeded": len(results),
                "failed": len(errors)
            }
        )


class FilterNode(BaseNode):
    """Filter 노드

    조건에 맞는 항목만 필터링합니다.

    Example:
        filter_node = FilterNode(
            node_id="filter_positive",
            filter_fn=lambda x: x["score"] > 0.5,
            input_key="items"
        )
    """

    def __init__(
        self,
        node_id: str,
        filter_fn: Callable[[Any], bool],
        input_key: str = "items",
        output_key: str = "filtered",
        name: str = "Filter",
        config: dict[str, Any] | None = None
    ):
        """
        Args:
            node_id: 노드 ID
            filter_fn: 필터 함수 (True이면 포함)
            input_key: 입력 리스트 키
            output_key: 출력 리스트 키
            name: 노드 이름
            config: 노드 설정
        """
        super().__init__(
            node_id=node_id,
            node_type="filter",
            name=name,
            config=config or {}
        )
        self.filter_fn = filter_fn
        self.input_key = input_key
        self.output_key = output_key

    async def execute(
        self,
        node_context: NodeContext,
        workflow_context: WorkflowContext
    ) -> NodeResult:
        """Filter 실행"""
        items = node_context.get_input(self.input_key)

        if items is None:
            return NodeResult(
                success=False,
                error=f"Input key '{self.input_key}' not found"
            )

        if not isinstance(items, list):
            return NodeResult(
                success=False,
                error=f"Input '{self.input_key}' must be a list"
            )

        try:
            # 필터 적용
            filtered = [item for item in items if self.filter_fn(item)]

            return NodeResult(
                success=True,
                output={
                    self.output_key: filtered,
                    "original_count": len(items),
                    "filtered_count": len(filtered)
                }
            )

        except Exception as e:
            return NodeResult(
                success=False,
                error=f"Filter function failed: {e}"
            )


class ReduceNode(BaseNode):
    """Reduce 노드

    리스트를 단일 값으로 축약합니다.

    Example:
        reduce_node = ReduceNode(
            node_id="sum",
            reduce_fn=lambda acc, x: acc + x["value"],
            initial_value=0,
            input_key="numbers"
        )
    """

    def __init__(
        self,
        node_id: str,
        reduce_fn: Callable[[Any, Any], Any],
        initial_value: Any,
        input_key: str = "items",
        output_key: str = "result",
        name: str = "Reduce",
        config: dict[str, Any] | None = None
    ):
        """
        Args:
            node_id: 노드 ID
            reduce_fn: Reduce 함수 (accumulator, item) -> accumulator
            initial_value: 초기값
            input_key: 입력 리스트 키
            output_key: 출력 값 키
            name: 노드 이름
            config: 노드 설정
        """
        super().__init__(
            node_id=node_id,
            node_type="reduce",
            name=name,
            config=config or {}
        )
        self.reduce_fn = reduce_fn
        self.initial_value = initial_value
        self.input_key = input_key
        self.output_key = output_key

    async def execute(
        self,
        node_context: NodeContext,
        workflow_context: WorkflowContext
    ) -> NodeResult:
        """Reduce 실행"""
        items = node_context.get_input(self.input_key)

        if items is None:
            return NodeResult(
                success=False,
                error=f"Input key '{self.input_key}' not found"
            )

        if not isinstance(items, list):
            return NodeResult(
                success=False,
                error=f"Input '{self.input_key}' must be a list"
            )

        try:
            # Reduce 적용
            result = self.initial_value
            for item in items:
                result = self.reduce_fn(result, item)

            return NodeResult(
                success=True,
                output={
                    self.output_key: result,
                    "processed_count": len(items)
                }
            )

        except Exception as e:
            return NodeResult(
                success=False,
                error=f"Reduce function failed: {e}"
            )
