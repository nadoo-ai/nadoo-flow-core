"""
Tests for Nadoo Flow multi-backend architecture
"""

import pytest
from nadoo_flow import (
    BackendRegistry,
    NadooBackend,
    IWorkflowBackend,
    WorkflowContext,
    BaseNode,
    NodeResult,
    CommonNodeTypes,
)


class MockBackend:
    """Mock backend for testing"""

    def __init__(self):
        self.executed = False

    async def execute(self, workflow_context, initial_input=None):
        """Mock execute method"""
        self.executed = True
        return workflow_context

    async def validate(self):
        """Mock validate method"""
        return True


@pytest.mark.asyncio
async def test_backend_registry_create_default():
    """Test creating default native backend"""
    backend = BackendRegistry.create()

    assert isinstance(backend, NadooBackend)
    assert backend is not None


@pytest.mark.asyncio
async def test_backend_registry_create_native():
    """Test creating native backend explicitly"""
    backend = BackendRegistry.create("native")

    assert isinstance(backend, NadooBackend)


@pytest.mark.asyncio
async def test_backend_registry_register():
    """Test registering custom backend"""
    # Register mock backend
    BackendRegistry.register("mock", MockBackend)

    # Verify it's registered
    assert "mock" in BackendRegistry.list_backends()

    # Create instance
    backend = BackendRegistry.create("mock")
    assert isinstance(backend, MockBackend)

    # Cleanup
    BackendRegistry.unregister("mock")


@pytest.mark.asyncio
async def test_backend_registry_unregister():
    """Test unregistering custom backend"""
    # Register and then unregister
    BackendRegistry.register("temp", MockBackend)
    assert "temp" in BackendRegistry.list_backends()

    BackendRegistry.unregister("temp")
    assert "temp" not in BackendRegistry.list_backends()


@pytest.mark.asyncio
async def test_backend_registry_cannot_unregister_native():
    """Test that native backend cannot be unregistered"""
    with pytest.raises(ValueError, match="Cannot unregister the default 'native' backend"):
        BackendRegistry.unregister("native")


@pytest.mark.asyncio
async def test_backend_registry_create_nonexistent():
    """Test creating non-existent backend raises error"""
    with pytest.raises(ValueError, match="Backend 'nonexistent' not registered"):
        BackendRegistry.create("nonexistent")


@pytest.mark.asyncio
async def test_backend_registry_set_default():
    """Test setting default backend"""
    # Register custom backend
    BackendRegistry.register("custom", MockBackend)

    # Save original default
    original_default = BackendRegistry.get_default()

    try:
        # Set new default
        BackendRegistry.set_default("custom")
        assert BackendRegistry.get_default() == "custom"

        # Create without args should use new default
        backend = BackendRegistry.create()
        assert isinstance(backend, MockBackend)
    finally:
        # Restore original default
        BackendRegistry.set_default(original_default)
        BackendRegistry.unregister("custom")


@pytest.mark.asyncio
async def test_backend_registry_set_default_nonexistent():
    """Test setting non-existent backend as default raises error"""
    with pytest.raises(ValueError, match="Backend 'nonexistent' not registered"):
        BackendRegistry.set_default("nonexistent")


@pytest.mark.asyncio
async def test_backend_registry_list_backends():
    """Test listing all registered backends"""
    backends = BackendRegistry.list_backends()

    assert "native" in backends
    assert isinstance(backends, list)


@pytest.mark.asyncio
async def test_nadoo_backend_execute():
    """Test NadooBackend execute method"""
    backend = NadooBackend()

    # Create simple workflow
    start_node = BaseNode(
        node_id="start",
        node_type=CommonNodeTypes.START,
        name="Start",
        config={}
    )

    class TestNode(BaseNode):
        async def execute(self, node_context, workflow_context):
            return NodeResult(
                success=True,
                output={"message": "test"}
            )

    test_node = TestNode(
        node_id="test",
        node_type=CommonNodeTypes.CUSTOM,
        name="Test",
        config={}
    )

    start_node.add_next_node("test")
    backend.add_node(start_node)
    backend.add_node(test_node)

    # Execute
    context = WorkflowContext()
    result = await backend.execute(context)

    assert result is not None
    assert "test" in result.node_contexts
    assert result.node_contexts["test"].output_data["message"] == "test"


@pytest.mark.asyncio
async def test_nadoo_backend_validate():
    """Test NadooBackend validate method"""
    backend = NadooBackend()

    # Create simple workflow
    start_node = BaseNode(
        node_id="start",
        node_type=CommonNodeTypes.START,
        name="Start",
        config={}
    )

    backend.add_node(start_node)

    # Validate
    is_valid = await backend.validate()
    assert is_valid is True


@pytest.mark.asyncio
async def test_nadoo_backend_get_node():
    """Test NadooBackend get_node method"""
    backend = NadooBackend()

    start_node = BaseNode(
        node_id="start",
        node_type=CommonNodeTypes.START,
        name="Start",
        config={}
    )

    backend.add_node(start_node)

    # Get node
    retrieved = backend.get_node("start")
    assert retrieved is not None
    assert retrieved.node_id == "start"

    # Get non-existent node
    none_node = backend.get_node("nonexistent")
    assert none_node is None


@pytest.mark.asyncio
async def test_nadoo_backend_start_node_id():
    """Test NadooBackend start_node_id property"""
    backend = NadooBackend()

    start_node = BaseNode(
        node_id="start",
        node_type=CommonNodeTypes.START,
        name="Start",
        config={}
    )

    backend.add_node(start_node)

    # Check start node ID
    assert backend.start_node_id == "start"


@pytest.mark.asyncio
async def test_iworkflow_backend_protocol():
    """Test that MockBackend satisfies IWorkflowBackend protocol"""
    backend = MockBackend()

    # Runtime check
    assert isinstance(backend, IWorkflowBackend)

    # Verify methods exist
    context = WorkflowContext()
    result = await backend.execute(context)
    assert backend.executed is True

    is_valid = await backend.validate()
    assert is_valid is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
