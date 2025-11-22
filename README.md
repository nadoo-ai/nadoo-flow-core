# Nadoo Flow Core

[![PyPI version](https://badge.fury.io/py/nadoo-flow-core.svg)](https://pypi.org/project/nadoo-flow-core/)
[![Python Versions](https://img.shields.io/pypi/pyversions/nadoo-flow-core.svg)](https://pypi.org/project/nadoo-flow-core/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/nadoo-ai/nadoo-flow-core/workflows/CI/badge.svg)](https://github.com/nadoo-ai/nadoo-flow-core/actions)
[![codecov](https://codecov.io/gh/nadoo-ai/nadoo-flow-core/branch/main/graph/badge.svg)](https://codecov.io/gh/nadoo-ai/nadoo-flow-core)

A flexible workflow orchestration framework with multi-backend support, inspired by LangChain's LCEL but designed for production use.

## Features

- **Type-Safe**: Full Pydantic support with type validation
- **Async-First**: Built on asyncio for high performance
- **Modular**: Composable nodes with clear interfaces
- **Streaming**: Native SSE support for real-time updates
- **Flexible**: Conditional branching, loops, parallel execution
- **Observable**: Built-in execution tracking and metadata

## Installation

```bash
pip install nadoo-flow-core
```

With CEL expression support:
```bash
pip install nadoo-flow-core[cel]
```

Future backend extensions:
```bash
pip install nadoo-flow-langgraph  # LangGraph backend (future)
pip install nadoo-flow-crewai     # CrewAI backend (future)
pip install nadoo-flow-a2a        # Google A2A backend (future)
```

## Quick Start

### Simple Workflow

```python
from nadoo_flow import WorkflowExecutor, WorkflowContext, BaseNode, NodeResult, CommonNodeTypes

# Define a custom node
class GreetingNode(BaseNode):
    async def execute(self, node_context, workflow_context):
        name = workflow_context.get_global_variable("name", "World")
        return NodeResult(
            success=True,
            output={"message": f"Hello, {name}!"}
        )

# Create workflow with START node
executor = WorkflowExecutor()
start_node = BaseNode(
    node_id="start",
    node_type=CommonNodeTypes.START,
    name="Start",
    config={}
)
greeting_node = GreetingNode(
    node_id="greet",
    node_type=CommonNodeTypes.CUSTOM,
    name="Greeting",
    config={}
)

# Connect nodes
start_node.add_next_node("greet")
executor.add_node(start_node)
executor.add_node(greeting_node)

# Execute
context = WorkflowContext()
context.set_global_variable("name", "Alice")

result = await executor.execute(context)
print(result.node_contexts["greet"].output_data)
# Output: {'message': 'Hello, Alice!'}
```

### Chaining API (Fluent Interface)

For simpler workflows, use the chaining API with the pipe operator:

```python
from nadoo_flow import ChainableNode, FunctionNode, NodeResult

# Define chainable nodes
class UppercaseNode(ChainableNode):
    def __init__(self):
        super().__init__(
            node_id="uppercase",
            node_type="transform",
            name="Uppercase",
            config={}
        )

    async def execute(self, node_context, workflow_context):
        text = node_context.input_data.get("text", "")
        return NodeResult(
            success=True,
            output={"text": text.upper()}
        )

class ReverseNode(ChainableNode):
    def __init__(self):
        super().__init__(
            node_id="reverse",
            node_type="transform",
            name="Reverse",
            config={}
        )

    async def execute(self, node_context, workflow_context):
        text = node_context.input_data.get("text", "")
        return NodeResult(
            success=True,
            output={"text": text[::-1]}
        )

# Chain nodes with pipe operator
chain = UppercaseNode() | ReverseNode()
result = await chain.run({"text": "hello"})
print(result)
# Output: {'text': 'OLLEH'}

# Use FunctionNode for quick transformations
chain = (
    UppercaseNode()
    | ReverseNode()
    | FunctionNode(lambda x: {"text": f"Result: {x['text']}"})
)
result = await chain.run({"text": "nadoo"})
print(result)
# Output: {'text': 'Result: OODAN'}
```

### Streaming

```python
# Stream outputs from each node
async for chunk in chain.stream({"text": "stream"}):
    print(chunk)
# Output:
# {'text': 'STREAM'}
# {'text': 'MAERTS'}
```

## Core Concepts

### Nodes
Nodes are the building blocks of workflows. Each node implements the `IStepNode` interface:

```python
class IStepNode(ABC):
    @abstractmethod
    async def execute(self, node_context: NodeContext, workflow_context: WorkflowContext) -> NodeResult:
        pass
```

### Chainable Nodes

For fluent workflow composition, use `ChainableNode`:

```python
from nadoo_flow import ChainableNode, NodeResult

class MyNode(ChainableNode):
    async def execute(self, node_context, workflow_context):
        # Your logic here
        return NodeResult(success=True, output={...})

# Chain with pipe operator
chain = node1 | node2 | node3
result = await chain.run(input_data)
```

### Helper Nodes

#### FunctionNode
Wrap any function (sync or async) as a node:

```python
from nadoo_flow import FunctionNode

# Sync function
uppercase = FunctionNode(lambda x: {"text": x["text"].upper()})

# Async function
async def fetch_data(x):
    # await some async operation
    return {"data": result}

fetch_node = FunctionNode(fetch_data)

# Use in chain
chain = uppercase | fetch_node | process
```

#### PassthroughNode
Pass input through unchanged (useful for debugging):

```python
from nadoo_flow import PassthroughNode

# Debug chain by inserting passthrough nodes
chain = (
    input_node
    | PassthroughNode()  # Check input here
    | transform_node
    | PassthroughNode()  # Check output here
    | output_node
)
```

### Workflow Context
Shared state across all nodes in a workflow:

- `global_variables`: Shared data between nodes
- `node_contexts`: Execution history of each node
- `execution_path`: Order of node execution

### Node Result
Return value from node execution:

- `success`: Whether the node succeeded
- `output`: Data to pass to next nodes
- `next_node_id`: Which node to execute next (optional)
- `conditional_next`: Conditional routing based on output

## Architecture

Nadoo Flow provides both explicit graph-based workflows and fluent chaining API:

| Feature | LangChain LCEL | Nadoo Flow |
|---------|----------------|------------|
| **Chaining** | Pipe operator (`\|`) | ✅ Pipe operator (`\|`) + Graph |
| **State** | LangGraph StateGraph | WorkflowContext |
| **Streaming** | `astream()` | `stream()` + SSE |
| **Methods** | `ainvoke()`, `astream()` | `run()`, `stream()` |
| **Abstraction** | High-level | Direct control |

### Chaining API

Nadoo Flow supports LangChain-style chaining with Nadoo-specific naming:

```python
# Nadoo Flow
from nadoo_flow import ChainableNode, FunctionNode, PassthroughNode

chain = transform_node | process_node | output_node
result = await chain.run(input_data)

# LangChain LCEL (for comparison)
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

chain = transform | process | output
result = await chain.ainvoke(input_data)
```

### Multi-Backend Architecture

Nadoo Flow is a **multi-backend orchestration framework**. The core API is an interface/protocol, and you can choose or extend different backend implementations:

```
┌──────────────────────────────────────────┐
│   Nadoo Flow Core API (Protocol)        │
│   - IWorkflowBackend (interface)         │
│   - ChainableNode (abstract)             │
│   - run() / stream() (stable API)        │
└──────────────┬───────────────────────────┘
               │
    ┌──────────┴──────────┐
    │  Backend Registry   │
    │  (Factory Pattern)  │
    └──────────┬──────────┘
               │
┌──────────────┴────────────────────────────────┐
│              │              │                  │
▼              ▼              ▼                  ▼
NadooBackend  LangGraphBackend CrewAIBackend  A2ABackend
("native")    (future)        (future)       (future)
```

**Key insight**: `WorkflowExecutor` is wrapped as the "native" backend implementation. Users interact with the stable API (`ChainableNode.run()`), and the backend is swappable through `BackendRegistry`.

#### Using Backends

```python
from nadoo_flow import BackendRegistry, WorkflowContext

# Use default native backend
backend = BackendRegistry.create()  # or create("native")
backend.add_node(my_node)

context = WorkflowContext()
result = await backend.execute(context)

# Register custom backend
class MyCustomBackend:
    async def execute(self, workflow_context, initial_input=None):
        # Your implementation
        return workflow_context

    async def validate(self):
        return True

BackendRegistry.register("custom", MyCustomBackend)
backend = BackendRegistry.create("custom")

# List all available backends
backends = BackendRegistry.list_backends()  # ["native", "custom"]
```

Benefits:
- **Future-proof**: Switch backends without code changes
- **Best-of-breed**: Use LangGraph for prototyping, CrewAI for agents, A2A for Google ecosystem
- **Gradual migration**: Start with NadooBackend, migrate incrementally
- **Vendor independence**: Not locked into any single framework
- **Same API, different engines**: Like SQLAlchemy (one API, many databases)

## Comparison with Other Frameworks

### Nadoo Flow vs LangChain/LangGraph

| Aspect | LangChain/LangGraph | Nadoo Flow |
|--------|---------------------|------------|
| **API Style** | `ainvoke()`, `astream()` | `run()`, `stream()` |
| **Chaining** | `RunnableSequence` | `NodeChain` |
| **Functions** | `RunnableLambda` | `FunctionNode` |
| **Passthrough** | `RunnablePassthrough` | `PassthroughNode` |
| **Dependencies** | Heavy (many packages) | Minimal (pydantic only) |
| **Use Case** | General AI workflows | No-code/low-code platforms |
| **Customization** | Pre-built integrations | Custom nodes |
| **Backend** | Fixed | Multi-backend (native + extensible) |

### When to use Nadoo Flow?

✅ Building no-code/low-code platforms
✅ Need fine-grained control over workflow execution
✅ Require custom node types
✅ Want minimal dependencies
✅ Need SSE streaming for UI integration
✅ Multi-backend flexibility with extensibility

### When to use LangChain?

✅ Rapid prototyping with LLMs
✅ Need pre-built integrations (200+ tools)
✅ Working with standard AI workflows
✅ Prefer LCEL ecosystem

### Migration Path

Start with Nadoo Flow native backend, migrate to LangGraph/CrewAI/A2A if needed:

```python
from nadoo_flow import BackendRegistry

# Start with native backend
backend = BackendRegistry.create("native")

# Later, register and switch to LangGraph/CrewAI/A2A
BackendRegistry.register("langgraph", LangGraphBackend)
BackendRegistry.set_default("langgraph")

# Your workflow code stays the same
backend = BackendRegistry.create()
result = await backend.execute(context)
```

## Advanced Features

### Conditional Branching

```python
result = NodeResult(
    success=True,
    conditional_next={
        "approved": "approval_node",
        "rejected": "rejection_node"
    }
)
```

### Parallel Execution

Use `ParallelNode` to execute multiple nodes concurrently.

### Streaming Updates

```python
# In a node
await workflow_context.emit_sse({
    "type": "TEXT_DELTA",
    "text": "Processing...",
    "timestamp": datetime.now(UTC).isoformat()
})
```

## License

MIT License - see LICENSE file for details.

## Links

- [Documentation](https://docs.nadoo.ai)
- [GitHub](https://github.com/nadoo-ai/nadoo-flow-core)
- [Nadoo AI Platform](https://nadoo.ai)
