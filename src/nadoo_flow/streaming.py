"""
Advanced Streaming for Nadoo Flow
고급 스트리밍 - Fine-grained 이벤트 스트리밍
"""

import asyncio
import time
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, AsyncIterator, Literal
from enum import Enum

from .base import NodeContext, WorkflowContext

logger = logging.getLogger(__name__)


class StreamEventType(str, Enum):
    """스트리밍 이벤트 타입"""

    # Workflow events
    WORKFLOW_START = "on_workflow_start"
    WORKFLOW_END = "on_workflow_end"
    WORKFLOW_ERROR = "on_workflow_error"

    # Node events
    NODE_START = "on_node_start"
    NODE_END = "on_node_end"
    NODE_ERROR = "on_node_error"

    # LLM events
    LLM_START = "on_llm_start"
    LLM_END = "on_llm_end"
    LLM_TOKEN = "on_llm_token"
    LLM_ERROR = "on_llm_error"

    # Tool events
    TOOL_START = "on_tool_start"
    TOOL_END = "on_tool_end"
    TOOL_ERROR = "on_tool_error"

    # Parser events
    PARSER_START = "on_parser_start"
    PARSER_END = "on_parser_end"
    PARSER_ERROR = "on_parser_error"

    # Custom events
    CUSTOM = "on_custom"


@dataclass
class StreamEvent:
    """스트리밍 이벤트

    모든 스트리밍 이벤트의 표준 형식입니다.
    """

    event_type: StreamEventType
    """이벤트 타입"""

    name: str
    """이벤트 이름 (노드 ID, 도구 이름 등)"""

    timestamp: float = field(default_factory=time.time)
    """이벤트 발생 시간"""

    data: dict[str, Any] = field(default_factory=dict)
    """이벤트 데이터"""

    run_id: str | None = None
    """실행 ID"""

    parent_run_id: str | None = None
    """부모 실행 ID"""

    tags: list[str] = field(default_factory=list)
    """태그 목록"""

    metadata: dict[str, Any] = field(default_factory=dict)
    """메타데이터"""

    def to_dict(self) -> dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "event": self.event_type.value,
            "name": self.name,
            "timestamp": self.timestamp,
            "data": self.data,
            "run_id": self.run_id,
            "parent_run_id": self.parent_run_id,
            "tags": self.tags,
            "metadata": self.metadata
        }


class StreamingContext:
    """스트리밍 컨텍스트

    스트리밍 이벤트를 수집하고 전파합니다.

    Example:
        async with StreamingContext() as ctx:
            async for event in ctx.stream():
                print(event.event_type, event.data)
    """

    def __init__(self, buffer_size: int = 100):
        """
        Args:
            buffer_size: 이벤트 버퍼 크기
        """
        self.buffer_size = buffer_size
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=buffer_size)
        self._active = False

    async def __aenter__(self):
        """Context manager 진입"""
        self._active = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager 종료"""
        self._active = False
        # Sentinel 값 전송 (스트림 종료 신호)
        await self._queue.put(None)

    async def emit(self, event: StreamEvent):
        """이벤트 발송

        Args:
            event: 스트리밍 이벤트
        """
        if not self._active:
            return

        try:
            await self._queue.put(event)
        except asyncio.QueueFull:
            logger.warning("Stream event queue is full, dropping event")

    async def stream(self) -> AsyncIterator[StreamEvent]:
        """이벤트 스트리밍

        Yields:
            StreamEvent
        """
        while True:
            event = await self._queue.get()

            if event is None:
                # Sentinel 값 받음 (종료)
                break

            yield event


class StreamingNode:
    """스트리밍 노드 믹스인

    노드에 스트리밍 기능을 추가합니다.

    Example:
        class MyLLMNode(BaseNode, StreamingNode):
            def __init__(self):
                BaseNode.__init__(self, ...)
                StreamingNode.__init__(self)

            async def execute(self, node_context, workflow_context):
                # 스트리밍 컨텍스트 가져오기
                stream_ctx = self.get_streaming_context(workflow_context)

                # 시작 이벤트
                await self.emit_start(stream_ctx, node_context)

                # LLM 호출 with 토큰 스트리밍
                async for token in llm.stream(...):
                    await self.emit_token(stream_ctx, token)

                # 종료 이벤트
                await self.emit_end(stream_ctx, node_context, result)
    """

    def __init__(self):
        """StreamingNode 초기화"""
        pass

    def get_streaming_context(self, workflow_context: WorkflowContext) -> StreamingContext | None:
        """워크플로우 컨텍스트에서 스트리밍 컨텍스트 가져오기

        Args:
            workflow_context: 워크플로우 컨텍스트

        Returns:
            스트리밍 컨텍스트 (없으면 None)
        """
        return getattr(workflow_context, "streaming_context", None)

    async def emit_event(
        self,
        stream_ctx: StreamingContext | None,
        event_type: StreamEventType,
        name: str,
        data: dict[str, Any] | None = None,
        **kwargs
    ):
        """이벤트 발송

        Args:
            stream_ctx: 스트리밍 컨텍스트
            event_type: 이벤트 타입
            name: 이벤트 이름
            data: 이벤트 데이터
            **kwargs: 추가 이벤트 속성
        """
        if stream_ctx is None:
            return

        event = StreamEvent(
            event_type=event_type,
            name=name,
            data=data or {},
            **kwargs
        )

        await stream_ctx.emit(event)

    async def emit_start(
        self,
        stream_ctx: StreamingContext | None,
        node_context: NodeContext
    ):
        """노드 시작 이벤트

        Args:
            stream_ctx: 스트리밍 컨텍스트
            node_context: 노드 컨텍스트
        """
        await self.emit_event(
            stream_ctx,
            StreamEventType.NODE_START,
            node_context.node_id,
            data={
                "node_type": node_context.node_type,
                "input_data": node_context.input_data
            }
        )

    async def emit_end(
        self,
        stream_ctx: StreamingContext | None,
        node_context: NodeContext,
        result: Any
    ):
        """노드 종료 이벤트

        Args:
            stream_ctx: 스트리밍 컨텍스트
            node_context: 노드 컨텍스트
            result: 실행 결과
        """
        await self.emit_event(
            stream_ctx,
            StreamEventType.NODE_END,
            node_context.node_id,
            data={
                "node_type": node_context.node_type,
                "output_data": node_context.output_data,
                "execution_time": node_context.execution_time,
                "success": getattr(result, "success", True)
            }
        )

    async def emit_token(
        self,
        stream_ctx: StreamingContext | None,
        token: str,
        node_id: str | None = None
    ):
        """LLM 토큰 이벤트

        Args:
            stream_ctx: 스트리밍 컨텍스트
            token: 토큰 문자열
            node_id: 노드 ID
        """
        await self.emit_event(
            stream_ctx,
            StreamEventType.LLM_TOKEN,
            node_id or "llm",
            data={"token": token}
        )


class StreamEventFilter:
    """스트리밍 이벤트 필터

    특정 이벤트만 필터링합니다.

    Example:
        filter = StreamEventFilter(
            event_types=[StreamEventType.LLM_TOKEN],
            node_ids=["llm_node_1"]
        )

        async for event in filter.filter(stream):
            print(event.data["token"])
    """

    def __init__(
        self,
        event_types: list[StreamEventType] | None = None,
        node_ids: list[str] | None = None,
        tags: list[str] | None = None
    ):
        """
        Args:
            event_types: 허용할 이벤트 타입 (None이면 전체)
            node_ids: 허용할 노드 ID (None이면 전체)
            tags: 허용할 태그 (None이면 전체)
        """
        self.event_types = set(event_types) if event_types else None
        self.node_ids = set(node_ids) if node_ids else None
        self.tags = set(tags) if tags else None

    def matches(self, event: StreamEvent) -> bool:
        """이벤트가 필터와 일치하는지 확인

        Args:
            event: 스트리밍 이벤트

        Returns:
            일치 여부
        """
        # 이벤트 타입 필터
        if self.event_types and event.event_type not in self.event_types:
            return False

        # 노드 ID 필터
        if self.node_ids and event.name not in self.node_ids:
            return False

        # 태그 필터
        if self.tags and not any(tag in self.tags for tag in event.tags):
            return False

        return True

    async def filter(
        self,
        stream: AsyncIterator[StreamEvent]
    ) -> AsyncIterator[StreamEvent]:
        """스트림 필터링

        Args:
            stream: 원본 스트림

        Yields:
            필터링된 이벤트
        """
        async for event in stream:
            if self.matches(event):
                yield event


class TokenCollector:
    """토큰 수집기

    LLM 토큰을 수집하여 텍스트로 변환합니다.

    Example:
        collector = TokenCollector()

        async for event in stream:
            if event.event_type == StreamEventType.LLM_TOKEN:
                collector.add_token(event.data["token"])

        full_text = collector.get_text()
    """

    def __init__(self):
        self.tokens: list[str] = []

    def add_token(self, token: str):
        """토큰 추가

        Args:
            token: 토큰 문자열
        """
        self.tokens.append(token)

    def get_text(self) -> str:
        """수집된 전체 텍스트 반환

        Returns:
            전체 텍스트
        """
        return "".join(self.tokens)

    def clear(self):
        """토큰 초기화"""
        self.tokens.clear()


# 편의 함수들
async def collect_tokens(stream: AsyncIterator[StreamEvent]) -> str:
    """스트림에서 LLM 토큰 수집

    Args:
        stream: 스트리밍 이벤트

    Returns:
        전체 텍스트
    """
    collector = TokenCollector()

    async for event in stream:
        if event.event_type == StreamEventType.LLM_TOKEN:
            collector.add_token(event.data["token"])

    return collector.get_text()


async def collect_node_outputs(
    stream: AsyncIterator[StreamEvent]
) -> dict[str, Any]:
    """스트림에서 노드 출력 수집

    Args:
        stream: 스트리밍 이벤트

    Returns:
        노드별 출력 딕셔너리
    """
    outputs = {}

    async for event in stream:
        if event.event_type == StreamEventType.NODE_END:
            outputs[event.name] = event.data.get("output_data", {})

    return outputs
