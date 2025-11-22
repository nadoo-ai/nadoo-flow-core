"""
Chaining API for Nadoo Flow - fluent interface for workflow composition
"""

from typing import Any, AsyncGenerator, Dict, Optional
from .base import BaseNode, NodeContext, NodeResult, WorkflowContext, WorkflowExecutor, CommonNodeTypes


class ChainableNode(BaseNode):
    """Base class for chainable nodes in Nadoo Flow

    Nadoo Flow nodes can be chained using the pipe operator for fluent composition.

    Example:
        chain = PromptNode() | AIAgentNode() | ParserNode()
        result = await chain.run({"input": "hello"})
    """

    def __or__(self, other: "ChainableNode") -> "NodeChain":
        """Enable pipe operator for chaining: node1 | node2"""
        if isinstance(other, NodeChain):
            return NodeChain(nodes=[self] + other.nodes)
        return NodeChain(nodes=[self, other])

    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run this node with input data

        Args:
            input_data: Input dictionary

        Returns:
            Output dictionary from node execution
        """
        context = WorkflowContext()
        for key, value in input_data.items():
            context.set_global_variable(key, value)

        node_context = NodeContext(
            node_id=self.node_id,
            node_type=self.node_type,
            input_data=input_data
        )

        result = await self.execute(node_context, context)

        if not result.success:
            raise RuntimeError(f"Node execution failed: {result.error}")

        return result.output

    async def stream(self, input_data: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        """Run and stream outputs from this node

        Args:
            input_data: Input dictionary

        Yields:
            Output chunks from node execution
        """
        result = await self.run(input_data)
        yield result

    def run_sync(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronous version of run() for convenience

        Note: Use run() for async code. This is just a helper.
        """
        import asyncio
        return asyncio.run(self.run(input_data))


class NodeChain:
    """Chain of nodes that execute in sequence

    NodeChain executes multiple nodes in order, passing output from one node as input to the next.

    Example:
        chain = NodeChain([transform_node, process_node, output_node])
        result = await chain.run({"input": "data"})
    """

    def __init__(self, nodes: list[ChainableNode]):
        self.nodes = nodes

    def __or__(self, other: ChainableNode) -> "NodeChain":
        """Enable pipe operator for extending the chain"""
        if isinstance(other, NodeChain):
            return NodeChain(nodes=self.nodes + other.nodes)
        return NodeChain(nodes=self.nodes + [other])

    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run all nodes in sequence

        Args:
            input_data: Input to the first node

        Returns:
            Output from the last node
        """
        current_data = input_data

        for node in self.nodes:
            current_data = await node.run(current_data)

        return current_data

    async def stream(self, input_data: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream outputs from all nodes in the chain

        Args:
            input_data: Input to the first node

        Yields:
            Output chunks from each node
        """
        current_data = input_data

        for node in self.nodes:
            async for chunk in node.stream(current_data):
                current_data = chunk
                yield chunk

    def run_sync(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronous version of run()"""
        import asyncio
        return asyncio.run(self.run(input_data))


class FunctionNode(ChainableNode):
    """Node that wraps a function for easy chaining

    FunctionNode allows you to use any function (sync or async) as a node in a chain.

    Example:
        uppercase = FunctionNode(lambda x: {"text": x["text"].upper()})
        chain = input_node | uppercase | process_node
    """

    def __init__(self, func, node_id: Optional[str] = None):
        """
        Args:
            func: Function to execute (can be sync or async)
            node_id: Optional node ID
        """
        import inspect
        import uuid

        self.func = func
        self.is_async = inspect.iscoroutinefunction(func)

        super().__init__(
            node_id=node_id or f"fn_{uuid.uuid4().hex[:8]}",
            node_type="function",
            name="Function",
            config={}
        )

    async def execute(self, node_context: NodeContext, workflow_context: WorkflowContext) -> NodeResult:
        """Execute the function"""
        try:
            input_data = {**node_context.input_data, **workflow_context.global_variables}

            if self.is_async:
                output = await self.func(input_data)
            else:
                output = self.func(input_data)

            return NodeResult(success=True, output=output)
        except Exception as e:
            return NodeResult(success=False, error=str(e))


class PassthroughNode(ChainableNode):
    """Node that passes input through unchanged

    Useful for debugging chains or copying data between pipeline stages.

    Example:
        chain = PassthroughNode() | process_node | PassthroughNode()
    """

    def __init__(self, node_id: Optional[str] = None):
        import uuid
        super().__init__(
            node_id=node_id or f"pass_{uuid.uuid4().hex[:8]}",
            node_type="passthrough",
            name="Passthrough",
            config={}
        )

    async def execute(self, node_context: NodeContext, workflow_context: WorkflowContext) -> NodeResult:
        """Pass through input unchanged"""
        return NodeResult(
            success=True,
            output={**node_context.input_data, **workflow_context.global_variables}
        )
