"""
Tests for Nadoo Flow chaining API
"""

import pytest
from nadoo_flow import (
    ChainableNode,
    NodeChain,
    FunctionNode,
    PassthroughNode,
    NodeContext,
    WorkflowContext,
    NodeResult,
)


class UppercaseNode(ChainableNode):
    """Node that uppercases text"""

    def __init__(self):
        super().__init__(
            node_id="uppercase",
            node_type="transform",
            name="Uppercase",
            config={}
        )

    async def execute(self, node_context: NodeContext, workflow_context: WorkflowContext) -> NodeResult:
        text = node_context.input_data.get("text", "")
        return NodeResult(
            success=True,
            output={"text": text.upper()}
        )


class ReverseNode(ChainableNode):
    """Node that reverses text"""

    def __init__(self):
        super().__init__(
            node_id="reverse",
            node_type="transform",
            name="Reverse",
            config={}
        )

    async def execute(self, node_context: NodeContext, workflow_context: WorkflowContext) -> NodeResult:
        text = node_context.input_data.get("text", "")
        return NodeResult(
            success=True,
            output={"text": text[::-1]}
        )


class AddPrefixNode(ChainableNode):
    """Node that adds a prefix"""

    def __init__(self, prefix: str = ">>>"):
        self.prefix = prefix
        super().__init__(
            node_id="add_prefix",
            node_type="transform",
            name="Add Prefix",
            config={"prefix": prefix}
        )

    async def execute(self, node_context: NodeContext, workflow_context: WorkflowContext) -> NodeResult:
        text = node_context.input_data.get("text", "")
        return NodeResult(
            success=True,
            output={"text": f"{self.prefix} {text}"}
        )


@pytest.mark.asyncio
async def test_pipe_operator():
    """Test pipe operator for chaining nodes"""
    # Create chain using | operator
    chain = UppercaseNode() | ReverseNode()

    result = await chain.run({"text": "hello"})

    assert result["text"] == "OLLEH"  # "hello" -> "HELLO" -> "OLLEH"


@pytest.mark.asyncio
async def test_multiple_pipe_operators():
    """Test multiple pipe operators"""
    chain = UppercaseNode() | ReverseNode() | AddPrefixNode(">>>")

    result = await chain.run({"text": "world"})

    assert result["text"] == ">>> DLROW"  # "world" -> "WORLD" -> "DLROW" -> ">>> DLROW"


@pytest.mark.asyncio
async def test_runnable_lambda():
    """Test FunctionNode with pipe operator"""
    uppercase = FunctionNode(lambda x: {"text": x["text"].upper()})
    add_exclamation = FunctionNode(lambda x: {"text": f"{x['text']}!"})

    chain = uppercase | add_exclamation

    result = await chain.run({"text": "hello"})

    assert result["text"] == "HELLO!"


@pytest.mark.asyncio
async def test_runnable_passthrough():
    """Test PassthroughNode"""
    passthrough = PassthroughNode()
    uppercase = UppercaseNode()

    chain = passthrough | uppercase

    result = await chain.run({"text": "test"})

    assert result["text"] == "TEST"


@pytest.mark.asyncio
async def test_astream():
    """Test astream method"""
    chain = UppercaseNode() | ReverseNode()

    chunks = []
    async for chunk in chain.stream({"text": "stream"}):
        chunks.append(chunk)

    # Should get two chunks (one from each node)
    assert len(chunks) == 2
    assert chunks[-1]["text"] == "MAERTS"  # "stream" -> "STREAM" -> "MAERTS"


@pytest.mark.asyncio
async def test_async_lambda():
    """Test FunctionNode with async function"""
    async def async_uppercase(x):
        return {"text": x["text"].upper()}

    chain = FunctionNode(async_uppercase) | ReverseNode()

    result = await chain.run({"text": "async"})

    assert result["text"] == "CNYSA"


@pytest.mark.asyncio
async def test_complex_chain():
    """Test complex chain with multiple operations"""
    # Chain: uppercase -> reverse -> add prefix -> lambda
    chain = (
        UppercaseNode()
        | ReverseNode()
        | AddPrefixNode("Result:")
        | FunctionNode(lambda x: {"text": f"[{x['text']}]"})
    )

    result = await chain.run({"text": "nadoo"})

    # "nadoo" -> "NADOO" -> "OODAN" -> "Result: OODAN" -> "[Result: OODAN]"
    assert result["text"] == "[Result: OODAN]"


@pytest.mark.asyncio
async def test_single_node_ainvoke():
    """Test ainvoke on a single node"""
    node = UppercaseNode()

    result = await node.run({"text": "single"})

    assert result["text"] == "SINGLE"


@pytest.mark.asyncio
async def test_chainable_node_pipe_with_chain():
    """Test piping ChainableNode with existing NodeChain"""
    node1 = UppercaseNode()
    node2 = ReverseNode()

    # Create chain first
    chain = NodeChain(nodes=[node2])

    # Pipe node with chain
    combined = node1 | chain

    assert isinstance(combined, NodeChain)
    assert len(combined.nodes) == 2


@pytest.mark.asyncio
async def test_chainable_node_run_failure():
    """Test ChainableNode.run() when execution fails"""

    class FailingNode(ChainableNode):
        def __init__(self):
            super().__init__("fail", "fail", "Fail", {})

        async def execute(self, node_context, workflow_context):
            # Return failure
            return NodeResult(success=False, error="Intentional failure")

    node = FailingNode()

    # Should raise RuntimeError on failure
    with pytest.raises(RuntimeError) as exc_info:
        await node.run({"input": "data"})

    assert "Node execution failed" in str(exc_info.value)


@pytest.mark.asyncio
async def test_function_node_with_none_return():
    """Test FunctionNode when function returns None"""

    def no_return_func(data):
        # Function returns None
        _ = data.get("x", 0) + 1
        return {}  # Return empty dict instead of None

    node = FunctionNode(no_return_func)
    chain = NodeChain(nodes=[node])

    result = await chain.run({"x": 5})

    # Should handle empty return gracefully
    assert result == {}


@pytest.mark.asyncio
async def test_passthrough_node_execute():
    """Test PassthroughNode.execute() directly"""

    node = PassthroughNode()
    # PassthroughNode copies input_data + global_variables
    input_data = {"data": "test"}
    node_context = NodeContext(node_id="pass", node_type="passthrough", input_data=input_data)
    workflow_context = WorkflowContext()

    # Test execute directly
    result = await node.execute(node_context, workflow_context)

    assert result.success
    assert result.output == input_data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
