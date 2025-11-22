"""
Tests for Streaming Module
스트리밍 모듈 테스트
"""

import pytest
import asyncio
from nadoo_flow.streaming import (
    StreamEventType,
    StreamEvent,
    StreamingContext,
    StreamingNode,
    StreamEventFilter,
    TokenCollector,
    collect_tokens,
    collect_node_outputs,
)
from nadoo_flow import BaseNode, NodeContext, NodeResult, WorkflowContext


@pytest.mark.asyncio
async def test_streaming_context():
    """스트리밍 컨텍스트 테스트"""

    async with StreamingContext() as ctx:
        # 이벤트 전송 태스크
        async def send_events():
            for i in range(5):
                await ctx.emit(StreamEvent(
                    event_type=StreamEventType.LLM_TOKEN,
                    name="test_node",
                    data={"token": f"token_{i}"}
                ))
                await asyncio.sleep(0.01)

        # 이벤트 수신 태스크
        async def receive_events():
            tokens = []
            async for event in ctx.stream():
                tokens.append(event.data["token"])
            return tokens

        send_task = asyncio.create_task(send_events())
        receive_task = asyncio.create_task(receive_events())

        await send_task
        # Context manager 종료 시 sentinel 전송됨

    tokens = await receive_task
    assert len(tokens) == 5
    assert tokens[0] == "token_0"
    assert tokens[4] == "token_4"


@pytest.mark.asyncio
async def test_stream_event_filter():
    """스트리밍 이벤트 필터 테스트"""

    events = [
        StreamEvent(
            event_type=StreamEventType.NODE_START,
            name="node_1",
            data={}
        ),
        StreamEvent(
            event_type=StreamEventType.LLM_TOKEN,
            name="node_1",
            data={"token": "hello"}
        ),
        StreamEvent(
            event_type=StreamEventType.LLM_TOKEN,
            name="node_2",
            data={"token": "world"}
        ),
        StreamEvent(
            event_type=StreamEventType.NODE_END,
            name="node_1",
            data={}
        ),
    ]

    async def event_stream():
        for event in events:
            yield event

    # 이벤트 타입 필터
    token_filter = StreamEventFilter(
        event_types=[StreamEventType.LLM_TOKEN]
    )

    filtered = []
    async for event in token_filter.filter(event_stream()):
        filtered.append(event)

    assert len(filtered) == 2
    assert all(e.event_type == StreamEventType.LLM_TOKEN for e in filtered)

    # 노드 ID 필터
    node_filter = StreamEventFilter(
        node_ids=["node_1"]
    )

    filtered = []
    async for event in node_filter.filter(event_stream()):
        filtered.append(event)

    assert len(filtered) == 3
    assert all(e.name == "node_1" for e in filtered)


@pytest.mark.asyncio
async def test_streaming_node():
    """스트리밍 노드 믹스인 테스트"""

    class TestStreamingNode(BaseNode, StreamingNode):
        def __init__(self):
            BaseNode.__init__(
                self,
                node_id="test_streaming",
                node_type="test",
                name="test_streaming_node",
                config={}
            )
            StreamingNode.__init__(self)

        async def execute(
            self,
            node_context: NodeContext,
            workflow_context: WorkflowContext
        ) -> NodeResult:
            stream_ctx = self.get_streaming_context(workflow_context)

            # 시작 이벤트
            await self.emit_start(stream_ctx, node_context)

            # 토큰 스트리밍
            tokens = ["Hello", ", ", "World", "!"]
            for token in tokens:
                await self.emit_token(stream_ctx, token, self.node_id)

            result = NodeResult(success=True, output={"text": "".join(tokens)})

            # 종료 이벤트
            await self.emit_end(stream_ctx, node_context, result)

            return result

    node = TestStreamingNode()
    workflow_context = WorkflowContext(workflow_id="test")

    # 스트리밍 컨텍스트 추가
    streaming_context = StreamingContext()
    workflow_context.streaming_context = streaming_context

    async with streaming_context:
        # 이벤트 수집
        async def collect_events():
            events = []
            async for event in streaming_context.stream():
                events.append(event)
            return events

        collect_task = asyncio.create_task(collect_events())

        # 노드 실행
        node_context = NodeContext(
            node_id="test_streaming",
            node_type="test",
            input_data={}
        )

        result = await node.execute(node_context, workflow_context)
        assert result.success

    events = await collect_task

    # 이벤트 확인: start + 4 tokens + end = 6
    assert len(events) == 6

    # 시작 이벤트
    assert events[0].event_type == StreamEventType.NODE_START

    # 토큰 이벤트들
    tokens = [e.data["token"] for e in events[1:5]]
    assert tokens == ["Hello", ", ", "World", "!"]

    # 종료 이벤트
    assert events[5].event_type == StreamEventType.NODE_END


@pytest.mark.asyncio
async def test_token_collector():
    """토큰 수집기 테스트"""

    collector = TokenCollector()

    # 토큰 추가
    tokens = ["Hello", ", ", "World", "!"]
    for token in tokens:
        collector.add_token(token)

    # 전체 텍스트 확인
    text = collector.get_text()
    assert text == "Hello, World!"

    # 초기화
    collector.clear()
    assert collector.get_text() == ""


@pytest.mark.asyncio
async def test_collect_tokens():
    """토큰 수집 편의 함수 테스트"""

    events = [
        StreamEvent(
            event_type=StreamEventType.NODE_START,
            name="node",
            data={}
        ),
        StreamEvent(
            event_type=StreamEventType.LLM_TOKEN,
            name="node",
            data={"token": "Hello"}
        ),
        StreamEvent(
            event_type=StreamEventType.LLM_TOKEN,
            name="node",
            data={"token": " World"}
        ),
        StreamEvent(
            event_type=StreamEventType.NODE_END,
            name="node",
            data={}
        ),
    ]

    async def event_stream():
        for event in events:
            yield event

    text = await collect_tokens(event_stream())
    assert text == "Hello World"


@pytest.mark.asyncio
async def test_collect_node_outputs():
    """노드 출력 수집 테스트"""

    events = [
        StreamEvent(
            event_type=StreamEventType.NODE_END,
            name="node_1",
            data={"output_data": {"result": "A"}}
        ),
        StreamEvent(
            event_type=StreamEventType.NODE_END,
            name="node_2",
            data={"output_data": {"result": "B"}}
        ),
    ]

    async def event_stream():
        for event in events:
            yield event

    outputs = await collect_node_outputs(event_stream())
    assert len(outputs) == 2
    assert outputs["node_1"]["result"] == "A"
    assert outputs["node_2"]["result"] == "B"


@pytest.mark.asyncio
async def test_stream_event_to_dict():
    """스트림 이벤트 직렬화 테스트"""

    event = StreamEvent(
        event_type=StreamEventType.LLM_TOKEN,
        name="test_node",
        data={"token": "hello"},
        run_id="run_123",
        tags=["tag1", "tag2"],
        metadata={"key": "value"}
    )

    event_dict = event.to_dict()

    assert event_dict["event"] == StreamEventType.LLM_TOKEN.value
    assert event_dict["name"] == "test_node"
    assert event_dict["data"]["token"] == "hello"
    assert event_dict["run_id"] == "run_123"
    assert event_dict["tags"] == ["tag1", "tag2"]
    assert event_dict["metadata"]["key"] == "value"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
