"""
Nadoo Native Backend Implementation
Default workflow execution backend for Nadoo Flow
"""

from typing import Any, Dict
from ..base import WorkflowContext, WorkflowExecutor, IStepNode


class NadooBackend:
    """Nadoo's native workflow execution backend

    This is the default backend that uses WorkflowExecutor directly.
    It provides full control and flexibility without any abstraction overhead.

    Example:
        backend = NadooBackend()
        backend.add_node(my_node)

        context = WorkflowContext()
        result = await backend.execute(context)
    """

    def __init__(self):
        self.executor = WorkflowExecutor()

    def add_node(self, node: IStepNode):
        """Add a node to the workflow

        Args:
            node: Node to add to the workflow
        """
        self.executor.add_node(node)

    def get_node(self, node_id: str) -> IStepNode | None:
        """Get a node by ID

        Args:
            node_id: ID of the node to retrieve

        Returns:
            Node if found, None otherwise
        """
        return self.executor.get_node(node_id)

    async def execute(
        self,
        workflow_context: WorkflowContext,
        initial_input: Dict[str, Any] | None = None
    ) -> WorkflowContext:
        """Execute the workflow using Nadoo's native executor

        Args:
            workflow_context: Workflow execution context
            initial_input: Optional initial input data

        Returns:
            Updated workflow context with execution results
        """
        return await self.executor.execute(workflow_context, initial_input)

    async def validate(self) -> bool:
        """Validate the workflow configuration

        Returns:
            True if workflow is valid, False otherwise
        """
        return await self.executor.validate()

    @property
    def start_node_id(self) -> str | None:
        """Get the start node ID

        Returns:
            Start node ID if set, None otherwise
        """
        return self.executor.start_node_id
