"""
Final Features Example - Memory, Streaming, Parallel Execution
최종 기능 예시 - 메모리, 스트리밍, 병렬 실행
"""

import asyncio
import time
from nadoo_flow import (
    BaseNode,
    NodeContext,
    NodeResult,
    WorkflowContext,
    NodeStatus,
    # Memory
    InMemoryChatHistory,
    SessionHistoryManager,
    ChatHistoryNode,
    Message,
    # Streaming
    StreamingContext,
    StreamingNode,
    StreamEventType,
    StreamEventFilter,
    TokenCollector,
    # Parallel
    ParallelNode,
    ParallelStrategy,
    FanOutFanInNode,
    race,
)


# ========================================
# 1. Memory + Streaming 통합 예시
# ========================================

class ConversationalLLMNode(BaseNode, ChatHistoryNode, StreamingNode):
    """대화형 LLM 노드 (메모리 + 스트리밍)

    채팅 히스토리를 유지하면서 토큰을 스트리밍합니다.
    """

    def __init__(
        self,
        node_id: str,
        history_manager: SessionHistoryManager,
        model: str = "gpt-4"
    ):
        BaseNode.__init__(self, node_id=node_id, node_type="llm")
        ChatHistoryNode.__init__(self, history_manager=history_manager)
        StreamingNode.__init__(self)
        self.model = model

    async def execute(
        self,
        node_context: NodeContext,
        workflow_context: WorkflowContext
    ) -> NodeResult:
        """대화형 LLM 실행"""

        # 1. 히스토리 로드
        history = await self.get_history(workflow_context)
        messages = await history.get_messages()

        # 2. 사용자 메시지 추가
        user_message = node_context.input_data.get("user_message", "")
        await history.add_message(Message.user(user_message))

        # 3. 스트리밍 컨텍스트 가져오기
        stream_ctx = self.get_streaming_context(workflow_context)

        # 4. LLM 호출 시작 이벤트
        await self.emit_event(
            stream_ctx,
            StreamEventType.LLM_START,
            self.node_id,
            data={"model": self.model, "message_count": len(messages) + 1}
        )

        # 5. LLM 응답 생성 (시뮬레이션)
        response_text = await self._generate_response(
            messages + [Message.user(user_message)],
            stream_ctx
        )

        # 6. 어시스턴트 메시지 저장
        await history.add_message(Message.assistant(response_text))

        # 7. LLM 종료 이벤트
        await self.emit_event(
            stream_ctx,
            StreamEventType.LLM_END,
            self.node_id,
            data={"response_length": len(response_text)}
        )

        return NodeResult(
            success=True,
            output={"response": response_text}
        )

    async def _generate_response(
        self,
        messages: list[Message],
        stream_ctx: StreamingContext | None
    ) -> str:
        """응답 생성 (스트리밍)"""

        # 실제로는 OpenAI/Claude API 호출
        # 여기서는 시뮬레이션
        tokens = [
            "안녕하세요", "! ", "어떻게", " 도와", "드릴", "까요", "?"
        ]

        full_text = ""
        for token in tokens:
            full_text += token

            # 토큰 스트리밍
            await self.emit_token(stream_ctx, token, self.node_id)

            # 시뮬레이션 딜레이
            await asyncio.sleep(0.1)

        return full_text


async def demo_memory_streaming():
    """메모리 + 스트리밍 데모"""

    print("=== Memory + Streaming Demo ===\n")

    # 히스토리 매니저 생성
    history_manager = SessionHistoryManager(
        history_factory=lambda sid: InMemoryChatHistory(),
        window_size=10
    )

    # 대화형 노드 생성
    llm_node = ConversationalLLMNode(
        node_id="chat_llm",
        history_manager=history_manager
    )

    # 워크플로우 컨텍스트
    workflow_context = WorkflowContext(
        workflow_id="chat_session_1",
        session_id="user_123"
    )

    # 스트리밍 컨텍스트 추가
    streaming_context = StreamingContext()
    workflow_context.streaming_context = streaming_context

    async with streaming_context:
        # 스트리밍 이벤트 처리 태스크
        async def process_events():
            # 토큰만 필터링
            token_filter = StreamEventFilter(
                event_types=[StreamEventType.LLM_TOKEN]
            )

            print("Bot: ", end="", flush=True)
            async for event in token_filter.filter(streaming_context.stream()):
                token = event.data.get("token", "")
                print(token, end="", flush=True)
            print("\n")

        # 이벤트 처리 시작
        event_task = asyncio.create_task(process_events())

        # 첫 번째 대화
        node_context = NodeContext(
            node_id="chat_llm",
            node_type="llm",
            input_data={"user_message": "안녕하세요"}
        )

        result = await llm_node.execute(node_context, workflow_context)

        # 이벤트 처리 완료 대기
        await event_task

    # 히스토리 확인
    history = await history_manager.get_history("user_123")
    messages = await history.get_messages()
    print(f"\nHistory: {len(messages)} messages")
    for msg in messages:
        print(f"  - {msg.role}: {msg.content[:50]}...")


# ========================================
# 2. Parallel Execution 예시
# ========================================

class SearchNode(BaseNode):
    """검색 노드 (시뮬레이션)"""

    def __init__(self, node_id: str, search_engine: str, delay: float = 1.0):
        super().__init__(node_id=node_id, node_type="search")
        self.search_engine = search_engine
        self.delay = delay

    async def execute(
        self,
        node_context: NodeContext,
        workflow_context: WorkflowContext
    ) -> NodeResult:
        """검색 실행"""

        query = node_context.input_data.get("query", "")

        print(f"[{self.search_engine}] Searching for: {query}")

        # 시뮬레이션 딜레이
        await asyncio.sleep(self.delay)

        # 결과 반환
        results = [
            f"{self.search_engine} result 1 for {query}",
            f"{self.search_engine} result 2 for {query}",
        ]

        print(f"[{self.search_engine}] Done!")

        return NodeResult(
            success=True,
            output={
                "engine": self.search_engine,
                "results": results,
                "count": len(results)
            }
        )


async def demo_parallel_all():
    """병렬 실행 (ALL 전략) - 모두 성공해야 함"""

    print("\n=== Parallel Execution (ALL) Demo ===\n")

    # 여러 검색 엔진 동시 실행
    parallel_search = ParallelNode(
        node_id="parallel_search",
        nodes=[
            SearchNode("google", "Google", delay=1.0),
            SearchNode("bing", "Bing", delay=1.5),
            SearchNode("duckduckgo", "DuckDuckGo", delay=0.8),
        ],
        strategy=ParallelStrategy.ALL,
        aggregate_outputs=True
    )

    workflow_context = WorkflowContext(workflow_id="search_workflow")
    node_context = NodeContext(
        node_id="parallel_search",
        node_type="parallel",
        input_data={"query": "Python async programming"}
    )

    start_time = time.time()
    result = await parallel_search.execute(node_context, workflow_context)
    elapsed = time.time() - start_time

    print(f"\nAll searches completed in {elapsed:.2f}s")
    print(f"Success: {result.success}")
    print(f"Results: {len(result.output)} engines")

    for engine_id, data in result.output.items():
        print(f"  - {data.get('engine')}: {data.get('count')} results")


async def demo_parallel_race():
    """병렬 실행 (RACE 전략) - 가장 빠른 것만"""

    print("\n=== Parallel Execution (RACE) Demo ===\n")

    # 가장 빠른 검색 엔진만 사용
    race_search = ParallelNode(
        node_id="race_search",
        nodes=[
            SearchNode("google", "Google", delay=1.2),
            SearchNode("bing", "Bing", delay=1.5),
            SearchNode("duckduckgo", "DuckDuckGo", delay=0.5),  # 가장 빠름
        ],
        strategy=ParallelStrategy.RACE
    )

    workflow_context = WorkflowContext(workflow_id="race_workflow")
    node_context = NodeContext(
        node_id="race_search",
        node_type="parallel",
        input_data={"query": "Machine learning"}
    )

    start_time = time.time()
    result = await race_search.execute(node_context, workflow_context)
    elapsed = time.time() - start_time

    print(f"\nFastest search completed in {elapsed:.2f}s")
    print(f"Winner: {result.metadata.get('fastest_node')}")
    print(f"Results: {result.output}")


async def demo_fan_out_fan_in():
    """Fan-out/Fan-in 패턴"""

    print("\n=== Fan-out/Fan-in Demo ===\n")

    # 여러 LLM에게 질문하고 consensus 찾기
    def find_consensus(results: dict) -> dict:
        """결과 집계 - consensus 찾기"""

        all_responses = []
        for node_id, data in results.items():
            if data and "results" in data:
                all_responses.extend(data["results"])

        return {
            "consensus": f"Found {len(all_responses)} total results",
            "engines_used": len(results),
            "all_results": all_responses
        }

    fan_out_fan_in = FanOutFanInNode(
        node_id="multi_search",
        parallel_nodes=[
            SearchNode("google", "Google", delay=0.8),
            SearchNode("bing", "Bing", delay=1.0),
        ],
        aggregator=find_consensus,
        parallel_strategy=ParallelStrategy.ALL_SETTLED
    )

    workflow_context = WorkflowContext(workflow_id="fan_workflow")
    node_context = NodeContext(
        node_id="multi_search",
        node_type="fan_out_fan_in",
        input_data={"query": "AI agents"}
    )

    start_time = time.time()
    result = await fan_out_fan_in.execute(node_context, workflow_context)
    elapsed = time.time() - start_time

    print(f"\nFan-out/Fan-in completed in {elapsed:.2f}s")
    print(f"Consensus: {result.output.get('consensus')}")
    print(f"Engines: {result.output.get('engines_used')}")


# ========================================
# 3. 통합 예시: Memory + Streaming + Parallel
# ========================================

class MultiLLMChatNode(BaseNode, ChatHistoryNode, StreamingNode):
    """여러 LLM을 병렬로 호출하는 채팅 노드"""

    def __init__(
        self,
        node_id: str,
        history_manager: SessionHistoryManager,
        models: list[str]
    ):
        BaseNode.__init__(self, node_id=node_id, node_type="multi_llm")
        ChatHistoryNode.__init__(self, history_manager=history_manager)
        StreamingNode.__init__(self)
        self.models = models

    async def execute(
        self,
        node_context: NodeContext,
        workflow_context: WorkflowContext
    ) -> NodeResult:
        """여러 LLM 병렬 실행"""

        # 히스토리 로드
        history = await self.get_history(workflow_context)
        messages = await history.get_messages()

        user_message = node_context.input_data.get("user_message", "")
        await history.add_message(Message.user(user_message))

        # 스트리밍 컨텍스트
        stream_ctx = self.get_streaming_context(workflow_context)

        # 병렬 LLM 호출 노드들 생성
        llm_nodes = [
            MockLLMNode(f"llm_{model}", model, stream_ctx)
            for model in self.models
        ]

        parallel_llm = ParallelNode(
            node_id=f"{self.node_id}_parallel",
            nodes=llm_nodes,
            strategy=ParallelStrategy.ALL_SETTLED,
            aggregate_outputs=True
        )

        # 병렬 실행
        parallel_result = await parallel_llm.execute(node_context, workflow_context)

        # 결과 집계
        all_responses = []
        for model_id, data in parallel_result.output.items():
            if data and "response" in data:
                all_responses.append({
                    "model": data.get("model"),
                    "response": data.get("response")
                })

        # 최종 응답 선택 (여기서는 첫 번째)
        final_response = all_responses[0]["response"] if all_responses else "No response"

        await history.add_message(Message.assistant(final_response))

        return NodeResult(
            success=True,
            output={
                "response": final_response,
                "all_responses": all_responses
            }
        )


class MockLLMNode(BaseNode, StreamingNode):
    """Mock LLM 노드"""

    def __init__(self, node_id: str, model: str, stream_ctx: StreamingContext | None):
        BaseNode.__init__(self, node_id=node_id, node_type="llm")
        StreamingNode.__init__(self)
        self.model = model
        self._stream_ctx = stream_ctx

    async def execute(
        self,
        node_context: NodeContext,
        workflow_context: WorkflowContext
    ) -> NodeResult:
        """Mock LLM 실행"""

        # 시뮬레이션
        await asyncio.sleep(0.5)

        response = f"Response from {self.model}"

        # 토큰 스트리밍
        for token in response.split():
            await self.emit_token(self._stream_ctx, token + " ", self.node_id)
            await asyncio.sleep(0.05)

        return NodeResult(
            success=True,
            output={"model": self.model, "response": response}
        )


async def demo_integrated():
    """통합 데모: Memory + Streaming + Parallel"""

    print("\n=== Integrated Demo (Memory + Streaming + Parallel) ===\n")

    history_manager = SessionHistoryManager(
        history_factory=lambda sid: InMemoryChatHistory()
    )

    multi_llm = MultiLLMChatNode(
        node_id="multi_llm_chat",
        history_manager=history_manager,
        models=["GPT-4", "Claude", "Gemini"]
    )

    workflow_context = WorkflowContext(
        workflow_id="integrated_workflow",
        session_id="user_456"
    )

    streaming_context = StreamingContext()
    workflow_context.streaming_context = streaming_context

    async with streaming_context:
        # 스트리밍 처리
        async def process_events():
            token_filter = StreamEventFilter(
                event_types=[StreamEventType.LLM_TOKEN]
            )

            print("Multi-LLM Response: ", end="", flush=True)
            async for event in token_filter.filter(streaming_context.stream()):
                token = event.data.get("token", "")
                print(token, end="", flush=True)
            print("\n")

        event_task = asyncio.create_task(process_events())

        node_context = NodeContext(
            node_id="multi_llm_chat",
            node_type="multi_llm",
            input_data={"user_message": "Explain async programming"}
        )

        result = await multi_llm.execute(node_context, workflow_context)

        await event_task

    print(f"All responses collected: {len(result.output.get('all_responses', []))}")


# ========================================
# Main
# ========================================

async def main():
    """모든 데모 실행"""

    # 1. Memory + Streaming
    await demo_memory_streaming()

    # 2. Parallel Execution
    await demo_parallel_all()
    await demo_parallel_race()
    await demo_fan_out_fan_in()

    # 3. 통합
    await demo_integrated()


if __name__ == "__main__":
    asyncio.run(main())
