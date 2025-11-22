"""
Example 2: Streaming Chatbot
예제 2: 스트리밍 챗봇
"""

import asyncio
from nadoo_flow import (
    BaseNode,
    NodeContext,
    NodeResult,
    WorkflowContext,
    StreamingContext,
    StreamingNode,
    StreamEventType,
    StreamEventFilter,
)


class StreamingLLMNode(BaseNode, StreamingNode):
    """스트리밍 LLM 노드"""

    def __init__(self, node_id: str, model: str = "gpt-4"):
        BaseNode.__init__(self, node_id=node_id, node_type="llm")
        StreamingNode.__init__(self)
        self.model = model

    async def execute(
        self,
        node_context: NodeContext,
        workflow_context: WorkflowContext
    ) -> NodeResult:
        """LLM 스트리밍 실행"""

        stream_ctx = self.get_streaming_context(workflow_context)
        prompt = node_context.input_data.get("prompt", "")

        # LLM 시작 이벤트
        await self.emit_event(
            stream_ctx,
            StreamEventType.LLM_START,
            self.node_id,
            data={"model": self.model, "prompt_length": len(prompt)}
        )

        # 응답 생성 (토큰 스트리밍)
        response = await self._stream_response(stream_ctx, prompt)

        # LLM 종료 이벤트
        await self.emit_event(
            stream_ctx,
            StreamEventType.LLM_END,
            self.node_id,
            data={"response_length": len(response), "model": self.model}
        )

        return NodeResult(
            success=True,
            output={"response": response}
        )

    async def _stream_response(
        self,
        stream_ctx: StreamingContext | None,
        prompt: str
    ) -> str:
        """응답 스트리밍 (Mock)"""

        # 실제로는 OpenAI API 스트리밍
        response_text = (
            "안녕하세요! 저는 AI 어시스턴트입니다. "
            "무엇을 도와드릴까요? "
            "질문이 있으시면 언제든지 말씀해주세요."
        )

        full_response = ""

        # 단어 단위로 스트리밍
        words = response_text.split()
        for word in words:
            token = word + " "
            full_response += token

            # 토큰 이벤트 전송
            await self.emit_token(stream_ctx, token, self.node_id)

            # 시뮬레이션 딜레이
            await asyncio.sleep(0.05)

        return full_response.strip()


async def main():
    """스트리밍 챗봇 데모"""

    print("=== Streaming Chatbot Demo ===\n")

    # 스트리밍 LLM 노드
    llm = StreamingLLMNode("streaming_llm", model="gpt-4")

    # 워크플로우 컨텍스트
    workflow_context = WorkflowContext(workflow_id="streaming_demo")

    # 스트리밍 컨텍스트 추가
    streaming_context = StreamingContext(buffer_size=100)
    workflow_context.streaming_context = streaming_context

    async with streaming_context:
        # 토큰 스트리밍 이벤트 처리
        async def display_tokens():
            """토큰을 실시간으로 출력"""

            # 토큰만 필터링
            token_filter = StreamEventFilter(
                event_types=[StreamEventType.LLM_TOKEN]
            )

            print("Assistant: ", end="", flush=True)

            async for event in token_filter.filter(streaming_context.stream()):
                token = event.data.get("token", "")
                print(token, end="", flush=True)

            print("\n")  # 줄바꿈

        # 토큰 처리 태스크 시작
        display_task = asyncio.create_task(display_tokens())

        # LLM 실행
        node_context = NodeContext(
            node_id="streaming_llm",
            node_type="llm",
            input_data={"prompt": "안녕하세요"}
        )

        result = await llm.execute(node_context, workflow_context)

        # 토큰 처리 완료 대기
        await display_task

    print(f"\nFinal result: {result.success}")
    print(f"Response length: {len(result.output.get('response', ''))}")


async def demo_with_progress():
    """진행 상황 표시 포함"""

    print("\n\n=== Streaming with Progress ===\n")

    llm = StreamingLLMNode("streaming_llm", model="claude")
    workflow_context = WorkflowContext(workflow_id="progress_demo")
    streaming_context = StreamingContext()
    workflow_context.streaming_context = streaming_context

    async with streaming_context:
        # 모든 이벤트 처리
        async def display_with_progress():
            """진행 상황과 토큰 함께 표시"""

            token_count = 0

            async for event in streaming_context.stream():
                if event.event_type == StreamEventType.LLM_START:
                    print(f"[Starting {event.data.get('model')}...]\n")
                    print("Response: ", end="", flush=True)

                elif event.event_type == StreamEventType.LLM_TOKEN:
                    token = event.data.get("token", "")
                    print(token, end="", flush=True)
                    token_count += 1

                elif event.event_type == StreamEventType.LLM_END:
                    print(f"\n\n[Completed - {token_count} tokens generated]")

        display_task = asyncio.create_task(display_with_progress())

        node_context = NodeContext(
            node_id="streaming_llm",
            node_type="llm",
            input_data={"prompt": "Tell me about AI"}
        )

        await llm.execute(node_context, workflow_context)
        await display_task


if __name__ == "__main__":
    asyncio.run(main())
    asyncio.run(demo_with_progress())
