"""
Workflow Backend Protocol
Defines the interface that all workflow backends must implement
"""

from typing import Protocol, Any, Dict, runtime_checkable
from ..base import WorkflowContext


@runtime_checkable
class IWorkflowBackend(Protocol):
    """Protocol for workflow execution backends

    This protocol defines the interface that all backends must implement.
    Backends can be Nadoo's native implementation, LangGraph, CrewAI, etc.

    Example:
        class MyBackend:
            async def execute(self, workflow_context, initial_input):
                # Implementation here
                return workflow_context

            async def validate(self):
                return True
    """

    async def execute(
        self,
        workflow_context: WorkflowContext,
        initial_input: Dict[str, Any] | None = None
    ) -> WorkflowContext:
        """Execute the workflow

        Args:
            workflow_context: Workflow execution context
            initial_input: Optional initial input data

        Returns:
            Updated workflow context with execution results
        """
        ...

    async def validate(self) -> bool:
        """Validate the workflow configuration

        Returns:
            True if workflow is valid, False otherwise
        """
        ...
