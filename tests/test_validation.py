"""
Tests for Workflow Validation (Cycle Detection, Reachability)
"""

import pytest

from nadoo_flow.base import BaseNode, CommonNodeTypes, NodeResult, WorkflowExecutor


class SimpleNode(BaseNode):
    """Simple test node"""

    def __init__(self, node_id: str, node_type: str = "test", name: str = None):
        if name is None:
            name = node_id
        super().__init__(
            node_id=node_id,
            node_type=node_type,
            name=name,
            config={}
        )

    async def execute(self, node_context, workflow_context) -> NodeResult:
        return NodeResult(
            success=True,
            output={"result": f"Node {self.node_id} executed"},
        )


@pytest.mark.asyncio
async def test_valid_linear_workflow():
    """Test validation of a simple linear workflow (no cycles)"""
    executor = WorkflowExecutor()

    # Create linear workflow: START -> A -> B -> END
    start = SimpleNode("start", CommonNodeTypes.START)
    node_a = SimpleNode("node_a")
    node_b = SimpleNode("node_b")
    end = SimpleNode("end", CommonNodeTypes.END)

    start.add_next_node("node_a")
    node_a.add_next_node("node_b")
    node_b.add_next_node("end")

    executor.add_node(start)
    executor.add_node(node_a)
    executor.add_node(node_b)
    executor.add_node(end)

    # Should pass validation
    result = await executor.validate()
    assert result is True


@pytest.mark.asyncio
async def test_cycle_detection():
    """Test that cycles are detected and rejected"""
    executor = WorkflowExecutor()

    # Create workflow with cycle: START -> A -> B -> A (cycle!)
    start = SimpleNode("start", CommonNodeTypes.START)
    node_a = SimpleNode("node_a")
    node_b = SimpleNode("node_b")

    start.add_next_node("node_a")
    node_a.add_next_node("node_b")
    node_b.add_next_node("node_a")  # Cycle: B -> A

    executor.add_node(start)
    executor.add_node(node_a)
    executor.add_node(node_b)

    # Should fail validation due to cycle
    result = await executor.validate()
    assert result is False


@pytest.mark.asyncio
async def test_self_loop_detection():
    """Test that self-loops are detected"""
    executor = WorkflowExecutor()

    # Create workflow with self-loop: START -> A -> A (self-loop!)
    start = SimpleNode("start", CommonNodeTypes.START)
    node_a = SimpleNode("node_a")

    start.add_next_node("node_a")
    node_a.add_next_node("node_a")  # Self-loop

    executor.add_node(start)
    executor.add_node(node_a)

    # Should fail validation due to self-loop
    result = await executor.validate()
    assert result is False


@pytest.mark.asyncio
async def test_unreachable_nodes_warning(caplog):
    """Test that unreachable nodes are detected (but don't fail validation)"""
    executor = WorkflowExecutor()

    # Create workflow with isolated node: START -> A, but B is isolated
    start = SimpleNode("start", CommonNodeTypes.START)
    node_a = SimpleNode("node_a")
    node_b = SimpleNode("node_b")  # Isolated node

    start.add_next_node("node_a")
    # node_b has no incoming edges

    executor.add_node(start)
    executor.add_node(node_a)
    executor.add_node(node_b)

    # Should pass validation but log warning
    result = await executor.validate()
    assert result is True

    # Check that warning was logged
    assert "Unreachable nodes" in caplog.text
    assert "node_b" in caplog.text


@pytest.mark.asyncio
async def test_no_end_node_reachable():
    """Test that validation fails if END node is not reachable"""
    executor = WorkflowExecutor()

    # Create workflow: START -> A, END (isolated)
    start = SimpleNode("start", CommonNodeTypes.START)
    node_a = SimpleNode("node_a")
    end = SimpleNode("end", CommonNodeTypes.END)  # Isolated END

    start.add_next_node("node_a")
    # END node has no incoming edges

    executor.add_node(start)
    executor.add_node(node_a)
    executor.add_node(end)

    # Should fail validation (END not reachable)
    result = await executor.validate()
    assert result is False


@pytest.mark.asyncio
async def test_branching_workflow_no_cycle():
    """Test validation of branching workflow (no cycles)"""
    executor = WorkflowExecutor()

    # Create branching workflow:
    #   START -> A -> B -> END
    #            A -> C -> END
    start = SimpleNode("start", CommonNodeTypes.START)
    node_a = SimpleNode("node_a")
    node_b = SimpleNode("node_b")
    node_c = SimpleNode("node_c")
    end = SimpleNode("end", CommonNodeTypes.END)

    start.add_next_node("node_a")
    node_a.add_next_node("node_b")
    node_a.add_next_node("node_c")  # Branch
    node_b.add_next_node("end")
    node_c.add_next_node("end")

    executor.add_node(start)
    executor.add_node(node_a)
    executor.add_node(node_b)
    executor.add_node(node_c)
    executor.add_node(end)

    # Should pass validation
    result = await executor.validate()
    assert result is True


@pytest.mark.asyncio
async def test_complex_cycle_detection():
    """Test cycle detection in complex graph"""
    executor = WorkflowExecutor()

    # Create complex workflow with cycle:
    #   START -> A -> B -> C -> D -> B (cycle!)
    start = SimpleNode("start", CommonNodeTypes.START)
    node_a = SimpleNode("node_a")
    node_b = SimpleNode("node_b")
    node_c = SimpleNode("node_c")
    node_d = SimpleNode("node_d")

    start.add_next_node("node_a")
    node_a.add_next_node("node_b")
    node_b.add_next_node("node_c")
    node_c.add_next_node("node_d")
    node_d.add_next_node("node_b")  # Cycle: D -> B

    executor.add_node(start)
    executor.add_node(node_a)
    executor.add_node(node_b)
    executor.add_node(node_c)
    executor.add_node(node_d)

    # Should fail validation due to cycle
    result = await executor.validate()
    assert result is False


@pytest.mark.asyncio
async def test_no_start_node():
    """Test that validation fails if no START node is defined"""
    executor = WorkflowExecutor()

    # Create workflow without START node
    node_a = SimpleNode("node_a")
    node_b = SimpleNode("node_b")

    node_a.add_next_node("node_b")

    executor.add_node(node_a)
    executor.add_node(node_b)

    # Should fail validation (no START node)
    result = await executor.validate()
    assert result is False


@pytest.mark.asyncio
async def test_multiple_end_nodes():
    """Test workflow with multiple END nodes (all reachable)"""
    executor = WorkflowExecutor()

    # Create workflow: START -> A -> END1
    #                  START -> B -> END2
    start = SimpleNode("start", CommonNodeTypes.START)
    node_a = SimpleNode("node_a")
    node_b = SimpleNode("node_b")
    end1 = SimpleNode("end1", CommonNodeTypes.END)
    end2 = SimpleNode("end2", CommonNodeTypes.END)

    start.add_next_node("node_a")
    start.add_next_node("node_b")
    node_a.add_next_node("end1")
    node_b.add_next_node("end2")

    executor.add_node(start)
    executor.add_node(node_a)
    executor.add_node(node_b)
    executor.add_node(end1)
    executor.add_node(end2)

    # Should pass validation (both END nodes reachable)
    result = await executor.validate()
    assert result is True


@pytest.mark.asyncio
async def test_workflow_without_end_node():
    """Test workflow without END node (should still validate)"""
    executor = WorkflowExecutor()

    # Create workflow: START -> A -> B (no END node)
    start = SimpleNode("start", CommonNodeTypes.START)
    node_a = SimpleNode("node_a")
    node_b = SimpleNode("node_b")

    start.add_next_node("node_a")
    node_a.add_next_node("node_b")

    executor.add_node(start)
    executor.add_node(node_a)
    executor.add_node(node_b)

    # Should pass validation (END node is optional)
    result = await executor.validate()
    assert result is True
