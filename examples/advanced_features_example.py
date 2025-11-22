"""
Nadoo Flow - Advanced Features Example
새로 추가된 기능들(Retry, Fallback, Parser, Caching, Callbacks) 사용 예제
"""

import asyncio
from typing import Any, Literal
from pydantic import BaseModel, Field

from nadoo_flow import (
    # Core
    BaseNode,
    NodeContext,
    NodeResult,
    WorkflowContext,
    WorkflowExecutor,
    # Resilience
    RetryableNode,
    RetryPolicy,
    FallbackNode,
    # Parsers
    StructuredOutputParser,
    ParserNode,
    RetryableParserNode,
    # Callbacks
    CallbackManager,
    ConsoleCallbackHandler,
    LoggingCallbackHandler,
    # Caching
    InMemoryCache,
    ResponseCache,
    CachedNode,
)


# ============================================================================
# 1. Retry 메커니즘 예제
# ============================================================================

class UnreliableLLMNode(RetryableNode):
    """재시도 기능이 있는 LLM 노드 예제

    네트워크 오류나 일시적인 장애 시 자동으로 재시도합니다.
    """

    def __init__(self):
        super().__init__(
            node_id="unreliable_llm",
            node_type="llm",
            name="Unreliable LLM with Retry",
            config={},
            retry_policy=RetryPolicy(
                max_attempts=5,
                initial_delay=1.0,
                max_delay=30.0,
                exponential_base=2.0,
                jitter=1.0,
                retry_on_exceptions=(TimeoutError, ConnectionError)
            )
        )
        self.call_count = 0

    async def _execute_with_retry(
        self,
        node_context: NodeContext,
        workflow_context: WorkflowContext
    ) -> NodeResult:
        """실제 LLM 호출 로직 (재시도 없이)"""
        self.call_count += 1

        # 시뮬레이션: 처음 2번은 실패
        if self.call_count <= 2:
            print(f"   Attempt {self.call_count} failed (simulated)")
            raise TimeoutError("Simulated timeout")

        # 3번째 시도에서 성공
        print(f"  ✅ Attempt {self.call_count} succeeded!")
        return NodeResult(
            success=True,
            output={"text": "LLM response after retries"}
        )


async def demo_retry():
    """Retry 메커니즘 데모"""
    print("\n" + "=" * 60)
    print("1. RETRY MECHANISM DEMO")
    print("=" * 60)

    node = UnreliableLLMNode()
    context = WorkflowContext()
    node_context = NodeContext(
        node_id=node.node_id,
        node_type=node.node_type
    )

    result = await node.execute(node_context, context)

    print(f"\n✅ Final Result: {result.success}")
    print(f"   Total attempts: {result.metadata.get('retry_info', {}).get('total_attempts')}")


# ============================================================================
# 2. Fallback 노드 예제
# ============================================================================

class GPT4Node(BaseNode):
    """GPT-4 시뮬레이션 (비싸지만 좋음)"""

    def __init__(self):
        super().__init__(
            node_id="gpt4",
            node_type="llm",
            name="GPT-4",
            config={}
        )

    async def execute(self, node_context: NodeContext, workflow_context: WorkflowContext) -> NodeResult:
        # 시뮬레이션: Rate limit 에러
        print("  🔴 GPT-4: Rate limit exceeded!")
        raise Exception("Rate limit exceeded")


class ClaudeNode(BaseNode):
    """Claude 시뮬레이션 (중간)"""

    def __init__(self):
        super().__init__(
            node_id="claude",
            node_type="llm",
            name="Claude",
            config={}
        )

    async def execute(self, node_context: NodeContext, workflow_context: WorkflowContext) -> NodeResult:
        # 시뮬레이션: Timeout
        print("  🟡 Claude: Timeout!")
        raise TimeoutError("Request timeout")


class LocalLlamaNode(BaseNode):
    """Local Llama 시뮬레이션 (저렴하고 안정적)"""

    def __init__(self):
        super().__init__(
            node_id="llama",
            node_type="llm",
            name="Local Llama",
            config={}
        )

    async def execute(self, node_context: NodeContext, workflow_context: WorkflowContext) -> NodeResult:
        # 항상 성공
        print("  🟢 Local Llama: Success!")
        return NodeResult(
            success=True,
            output={"text": "Response from local Llama model"}
        )


async def demo_fallback():
    """Fallback 노드 데모"""
    print("\n" + "=" * 60)
    print("2. FALLBACK NODE DEMO")
    print("=" * 60)

    # Fallback 체인: GPT-4 → Claude → Local Llama
    fallback = FallbackNode(
        node_id="llm_fallback",
        nodes=[
            GPT4Node(),
            ClaudeNode(),
            LocalLlamaNode()
        ],
        handle_exceptions=(Exception,)
    )

    context = WorkflowContext()
    node_context = NodeContext(
        node_id=fallback.node_id,
        node_type=fallback.node_type
    )

    result = await fallback.execute(node_context, context)

    print(f"\n✅ Final Result: {result.success}")
    print(f"   Successful node: {result.metadata.get('fallback_info', {}).get('successful_node')}")
    print(f"   Fallback index: {result.metadata.get('fallback_info', {}).get('fallback_index')}")


# ============================================================================
# 3. Structured Output Parser 예제
# ============================================================================

class AgentAction(BaseModel):
    """에이전트 행동 모델"""

    action: Literal["search", "calculate", "answer"]
    reasoning: str = Field(description="Why this action was chosen")
    parameters: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)

    class Config:
        json_schema_extra = {
            "example": {
                "action": "search",
                "reasoning": "User wants to know about AI",
                "parameters": {"query": "What is AI?"},
                "confidence": 0.9
            }
        }


class MockLLMNode(BaseNode):
    """LLM 출력을 시뮬레이션하는 노드"""

    def __init__(self, response: str):
        super().__init__(
            node_id="mock_llm",
            node_type="llm",
            name="Mock LLM",
            config={}
        )
        self.response = response

    async def execute(self, node_context: NodeContext, workflow_context: WorkflowContext) -> NodeResult:
        return NodeResult(
            success=True,
            output={"text": self.response}
        )


async def demo_parser():
    """Structured Output Parser 데모"""
    print("\n" + "=" * 60)
    print("3. STRUCTURED OUTPUT PARSER DEMO")
    print("=" * 60)

    # LLM 응답 시뮬레이션 (JSON 형식)
    llm_output = """
    Here's my decision:
    ```json
    {
        "action": "search",
        "reasoning": "The user wants to find information about quantum computing",
        "parameters": {
            "query": "quantum computing basics",
            "max_results": 5
        },
        "confidence": 0.85
    }
    ```
    """

    # 파서 생성
    parser = StructuredOutputParser(pydantic_model=AgentAction)

    # 파서 노드 생성
    parser_node = ParserNode(
        node_id="action_parser",
        parser=parser,
        input_key="text"
    )

    # 워크플로우 실행
    llm_node = MockLLMNode(llm_output)
    executor = WorkflowExecutor()
    executor.add_node(llm_node)

    context = WorkflowContext()
    node_context = NodeContext(
        node_id=llm_node.node_id,
        node_type=llm_node.node_type
    )

    # LLM 실행
    llm_result = await llm_node.execute(node_context, context)
    print(f"\n📝 LLM Output:\n{llm_result.output['text'][:200]}...")

    # 파싱
    parser_context = NodeContext(
        node_id=parser_node.node_id,
        node_type=parser_node.node_type,
        input_data=llm_result.output
    )
    parse_result = await parser_node.execute(parser_context, context)

    if parse_result.success:
        parsed = parse_result.output["parsed"]
        print(f"\n✅ Parsed Action:")
        print(f"   Action: {parsed['action']}")
        print(f"   Reasoning: {parsed['reasoning']}")
        print(f"   Confidence: {parsed['confidence']}")
        print(f"   Parameters: {parsed['parameters']}")


# ============================================================================
# 4. LLM Response Caching 예제
# ============================================================================

class CachedLLMNode(BaseNode, CachedNode):
    """캐싱 기능이 있는 LLM 노드"""

    def __init__(self, cache: ResponseCache):
        BaseNode.__init__(
            self,
            node_id="cached_llm",
            node_type="llm",
            name="Cached LLM",
            config={}
        )
        CachedNode.__init__(self, response_cache=cache)
        self.call_count = 0

    async def execute(self, node_context: NodeContext, workflow_context: WorkflowContext) -> NodeResult:
        prompt = node_context.get_input("prompt", "")

        # 캐시 키 생성
        cache_key = self.response_cache.make_key(
            prompt=prompt,
            model="gpt-4",
            temperature=0.7
        )

        # 캐시 조회
        if self.is_cache_enabled():
            cached = self.response_cache.get(cache_key)
            if cached:
                print(f"  💾 Cache HIT for: {prompt[:50]}...")
                return NodeResult(success=True, output=cached)

        # 캐시 미스 - LLM 호출
        print(f"  🔄 Cache MISS - Calling LLM for: {prompt[:50]}...")
        self.call_count += 1

        # 시뮬레이션: LLM 호출 (1초 소요)
        await asyncio.sleep(1)
        response = f"LLM response to: {prompt}"

        output = {"text": response, "call_count": self.call_count}

        # 캐시 저장
        if self.is_cache_enabled():
            self.response_cache.set(cache_key, output, ttl=3600)

        return NodeResult(success=True, output=output)


async def demo_caching():
    """LLM Response Caching 데모"""
    print("\n" + "=" * 60)
    print("4. LLM RESPONSE CACHING DEMO")
    print("=" * 60)

    # 캐시 설정
    cache = ResponseCache(
        cache=InMemoryCache(default_ttl=3600),
        namespace="demo"
    )

    node = CachedLLMNode(cache)
    context = WorkflowContext()

    # 첫 번째 호출 (캐시 미스)
    print("\n📤 First call:")
    node_context1 = NodeContext(
        node_id=node.node_id,
        node_type=node.node_type,
        input_data={"prompt": "What is artificial intelligence?"}
    )
    result1 = await node.execute(node_context1, context)
    print(f"   Response: {result1.output['text']}")

    # 두 번째 호출 - 동일한 프롬프트 (캐시 히트)
    print("\n📥 Second call (same prompt):")
    node_context2 = NodeContext(
        node_id=node.node_id,
        node_type=node.node_type,
        input_data={"prompt": "What is artificial intelligence?"}
    )
    result2 = await node.execute(node_context2, context)
    print(f"   Response: {result2.output['text']}")

    # 세 번째 호출 - 다른 프롬프트 (캐시 미스)
    print("\n📤 Third call (different prompt):")
    node_context3 = NodeContext(
        node_id=node.node_id,
        node_type=node.node_type,
        input_data={"prompt": "Explain machine learning"}
    )
    result3 = await node.execute(node_context3, context)
    print(f"   Response: {result3.output['text']}")

    print(f"\n📊 Total LLM API calls: {node.call_count} (saved 1 call via cache)")


# ============================================================================
# 5. Callback System 예제
# ============================================================================

async def demo_callbacks():
    """Callback System 데모"""
    print("\n" + "=" * 60)
    print("5. CALLBACK SYSTEM DEMO")
    print("=" * 60)

    # 콜백 매니저 설정
    callback_manager = CallbackManager()
    callback_manager.add_handler(ConsoleCallbackHandler(verbose=True, colors=True))

    # 간단한 워크플로우 생성
    class SimpleNode(BaseNode):
        async def execute(self, node_context: NodeContext, workflow_context: WorkflowContext) -> NodeResult:
            await asyncio.sleep(0.5)  # 작업 시뮬레이션
            return NodeResult(success=True, output={"result": "completed"})

    node = SimpleNode(
        node_id="simple_node",
        node_type="custom",
        name="Simple Node",
        config={}
    )

    context = WorkflowContext()
    node_context = NodeContext(
        node_id=node.node_id,
        node_type=node.node_type
    )

    # 워크플로우 시작 이벤트
    callback_manager.on_workflow_start(context, inputs={"test": "data"})

    # 노드 시작 이벤트
    callback_manager.on_node_start(node_context, context)

    # 노드 실행
    result = await node.execute(node_context, context)

    # 노드 종료 이벤트
    from nadoo_flow import NodeStatus
    node_context.end_time = node_context.start_time + 0.5
    node_context.status = NodeStatus.SUCCESS
    callback_manager.on_node_end(node_context, context, result)

    # 워크플로우 종료 이벤트
    context.status = NodeStatus.SUCCESS
    context.end_time = context.start_time + 1.0
    callback_manager.on_workflow_end(context)


# ============================================================================
# Main
# ============================================================================

async def main():
    """모든 데모 실행"""
    print("\n" + "=" * 60)
    print("NADOO FLOW - ADVANCED FEATURES DEMO")
    print("=" * 60)

    await demo_retry()
    await demo_fallback()
    await demo_parser()
    await demo_caching()
    await demo_callbacks()

    print("\n" + "=" * 60)
    print("✅ ALL DEMOS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
