"""
Basic tests for Nadoo Flow
"""

import pytest
from nadoo_flow import (
    WorkflowExecutor,
    WorkflowContext,
    BaseNode,
    NodeResult,
    CommonNodeTypes,
    NodeContext,
)


class GreetingNode(BaseNode):
    """Simple test node that returns a greeting"""

    async def execute(self, node_context: NodeContext, workflow_context: WorkflowContext) -> NodeResult:
        name = workflow_context.get_global_variable("name", "World")
        return NodeResult(
            success=True,
            output={"message": f"Hello, {name}!"}
        )


class AddNumbersNode(BaseNode):
    """Node that adds two numbers"""

    async def execute(self, node_context: NodeContext, workflow_context: WorkflowContext) -> NodeResult:
        a = workflow_context.get_global_variable("a", 0)
        b = workflow_context.get_global_variable("b", 0)
        return NodeResult(
            success=True,
            output={"result": a + b}
        )


@pytest.mark.asyncio
async def test_basic_workflow():
    """Test basic workflow execution"""
    executor = WorkflowExecutor()

    # Need a START node for workflow to execute
    start_node = BaseNode(
        node_id="start",
        node_type=CommonNodeTypes.START,
        name="Start",
        config={}
    )

    greeting_node = GreetingNode(
        node_id="greeting",
        node_type=CommonNodeTypes.UNIVERSAL,
        name="Greeting Node",
        config={}
    )

    start_node.add_next_node("greeting")

    executor.add_node(start_node)
    executor.add_node(greeting_node)

    context = WorkflowContext()
    context.set_global_variable("name", "Nadoo")

    result = await executor.execute(context, {"input": "test"})

    assert result.status.value == "success"
    assert "greeting" in result.node_contexts
    assert result.node_contexts["greeting"].output_data["message"] == "Hello, Nadoo!"


@pytest.mark.asyncio
async def test_node_context():
    """Test node context functionality"""
    context = NodeContext(
        node_id="test",
        node_type=CommonNodeTypes.UNIVERSAL,
    )

    context.set_output("key1", "value1")
    context.set_variable("var1", 100)

    assert context.get_input("key1", None) is None
    assert context.output_data["key1"] == "value1"
    assert context.get_variable("var1") == 100


@pytest.mark.asyncio
async def test_workflow_context():
    """Test workflow context functionality"""
    context = WorkflowContext()

    context.set_global_variable("x", 10)
    context.set_global_variable("y", 20)

    assert context.get_global_variable("x") == 10
    assert context.get_global_variable("y") == 20
    assert context.get_global_variable("z", "default") == "default"


@pytest.mark.asyncio
async def test_multi_node_workflow():
    """Test workflow with multiple nodes"""
    executor = WorkflowExecutor()

    # Create nodes
    start_node = BaseNode(
        node_id="start",
        node_type=CommonNodeTypes.START,
        name="Start",
        config={}
    )

    add_node = AddNumbersNode(
        node_id="add",
        node_type=CommonNodeTypes.UNIVERSAL,
        name="Add Numbers",
        config={}
    )

    # Connect nodes
    start_node.add_next_node("add")

    executor.add_node(start_node)
    executor.add_node(add_node)

    context = WorkflowContext()
    context.set_global_variable("a", 5)
    context.set_global_variable("b", 3)

    result = await executor.execute(context)

    assert result.status.value == "success"
    assert len(result.execution_path) == 2
    assert result.node_contexts["add"].output_data["result"] == 8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
