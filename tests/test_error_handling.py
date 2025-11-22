"""
Tests for error handling and edge cases in Nadoo Flow
"""

import pytest
from nadoo_flow import (
    WorkflowExecutor,
    WorkflowContext,
    BaseNode,
    ChainableNode,
    NodeResult,
    CommonNodeTypes,
    NodeContext,
    FunctionNode,
)


class FailingNode(ChainableNode):
    """Node that always fails"""

    def __init__(self):
        super().__init__(
            node_id="failing",
            node_type="test",
            name="Failing Node",
            config={}
        )

    async def execute(self, node_context: NodeContext, workflow_context: WorkflowContext) -> NodeResult:
        return NodeResult(
            success=False,
            error="Intentional failure for testing"
        )


class ExceptionNode(ChainableNode):
    """Node that raises an exception"""

    def __init__(self):
        super().__init__(
            node_id="exception",
            node_type="test",
            name="Exception Node",
            config={}
        )

    async def execute(self, node_context: NodeContext, workflow_context: WorkflowContext) -> NodeResult:
        raise ValueError("Test exception")


@pytest.mark.asyncio
async def test_chain_node_failure():
    """Test that chain properly handles node failures"""
    class SuccessNode(ChainableNode):
        async def execute(self, node_context: NodeContext, workflow_context: WorkflowContext) -> NodeResult:
            return NodeResult(success=True, output={"status": "ok"})

    chain = SuccessNode(node_id="success", node_type="test", name="Success", config={}) | FailingNode()

    with pytest.raises(RuntimeError) as exc_info:
        await chain.run({"input": "test"})

    assert "Node execution failed" in str(exc_info.value)
    assert "Intentional failure" in str(exc_info.value)


@pytest.mark.asyncio
async def test_function_node_exception():
    """Test FunctionNode error handling"""

    def failing_function(x):
        raise ValueError("Function failed")

    node = FunctionNode(failing_function)

    # FunctionNode catches exceptions and returns NodeResult with error
    context = WorkflowContext()
    node_context = NodeContext(node_id=node.node_id, node_type=node.node_type)
    node_context.input_data = {"test": "data"}

    result = await node.execute(node_context, context)

    assert not result.success
    assert "Function failed" in result.error


@pytest.mark.asyncio
async def test_workflow_executor_no_start_node():
    """Test workflow execution without a start node"""
    executor = WorkflowExecutor()

    # Add a node that is NOT a start node
    node = BaseNode(
        node_id="regular",
        node_type=CommonNodeTypes.CUSTOM,
        name="Regular Node",
        config={}
    )
    executor.add_node(node)

    context = WorkflowContext()

    # Should fail because there's no start node
    result = await executor.execute(context)

    assert result.status.value == "failed"
    assert "No start node defined" in result.error


@pytest.mark.asyncio
async def test_workflow_executor_node_not_found():
    """Test workflow execution with missing node in chain"""
    executor = WorkflowExecutor()

    # Create start node that points to non-existent node
    start_node = BaseNode(
        node_id="start",
        node_type=CommonNodeTypes.START,
        name="Start",
        config={}
    )
    start_node.add_next_node("nonexistent_node_id")
    executor.add_node(start_node)

    context = WorkflowContext()

    result = await executor.execute(context)

    assert result.status.value == "failed"
    assert "Node nonexistent_node_id not found" in result.error


@pytest.mark.asyncio
async def test_workflow_executor_node_failure():
    """Test workflow execution with node failure"""
    executor = WorkflowExecutor()

    start_node = BaseNode(
        node_id="start",
        node_type=CommonNodeTypes.START,
        name="Start",
        config={}
    )

    failing_node = FailingNode()

    start_node.add_next_node(failing_node.node_id)
    executor.add_node(start_node)
    executor.add_node(failing_node)

    context = WorkflowContext()

    result = await executor.execute(context)

    assert result.status.value == "failed"
    assert "Intentional failure" in result.error


def test_run_sync():
    """Test synchronous run() helper

    Note: This test is NOT marked as async because run_sync() creates its own event loop.
    It's designed for use in synchronous/non-async contexts.
    """
    class SimpleNode(ChainableNode):
        async def execute(self, node_context: NodeContext, workflow_context: WorkflowContext) -> NodeResult:
            return NodeResult(success=True, output={"text": "sync test"})

    node = SimpleNode(node_id="simple", node_type="test", name="Simple", config={})

    # Test run_sync (synchronous wrapper)
    result = node.run_sync({"input": "test"})

    assert result["text"] == "sync test"


@pytest.mark.asyncio
async def test_chain_extend_with_nodechain():
    """Test chaining when other is also a NodeChain"""
    class Node1(ChainableNode):
        async def execute(self, node_context: NodeContext, workflow_context: WorkflowContext) -> NodeResult:
            return NodeResult(success=True, output={"step": 1})

    class Node2(ChainableNode):
        async def execute(self, node_context: NodeContext, workflow_context: WorkflowContext) -> NodeResult:
            return NodeResult(success=True, output={"step": 2})

    class Node3(ChainableNode):
        async def execute(self, node_context: NodeContext, workflow_context: WorkflowContext) -> NodeResult:
            return NodeResult(success=True, output={"step": 3})

    node1 = Node1(node_id="n1", node_type="test", name="Node1", config={})
    node2 = Node2(node_id="n2", node_type="test", name="Node2", config={})
    node3 = Node3(node_id="n3", node_type="test", name="Node3", config={})

    # Create two chains
    chain1 = node1 | node2
    chain2 = chain1 | node3

    result = await chain2.run({})

    assert result["step"] == 3
    assert len(chain2.nodes) == 3


@pytest.mark.asyncio
async def test_async_function_node():
    """Test FunctionNode with async function"""
    async def async_transform(x):
        # Simulate async operation
        import asyncio
        await asyncio.sleep(0.001)
        return {"result": "async"}

    node = FunctionNode(async_transform)
    result = await node.run({"input": "test"})

    assert result["result"] == "async"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
