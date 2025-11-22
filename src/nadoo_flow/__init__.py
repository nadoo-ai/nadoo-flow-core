"""
Nadoo Flow - A flexible workflow orchestration framework
"""

from .base import (
    # Core Types
    NodeType,
    CommonNodeTypes,
    NodeStatus,
    # Context Classes
    NodeContext,
    WorkflowContext,
    # Node Interfaces
    IStepNode,
    BaseNode,
    # Result Classes
    NodeResult,
    Answer,
    NodeChunk,
    # Executor
    WorkflowExecutor,
)

from .chain import (
    # Chaining API - Fluent interface for workflow composition
    ChainableNode,
    NodeChain,
    FunctionNode,
    PassthroughNode,
)

from .backends import (
    # Multi-backend support
    IWorkflowBackend,
    BackendRegistry,
    NadooBackend,
)

from .resilience import (
    # Resilience mechanisms - Retry and Fallback
    RetryPolicy,
    RetryableNode,
    FallbackNode,
)

from .parsers import (
    # Output Parsers - Structured output extraction
    OutputParser,
    StructuredOutputParser,
    JsonOutputParser,
    StringOutputParser,
    ParserNode,
    RetryableParserNode,
)

from .callbacks import (
    # Callback system - Observability and monitoring
    CallbackEvent,
    BaseCallbackHandler,
    CallbackManager,
    ConsoleCallbackHandler,
    LoggingCallbackHandler,
)

from .caching import (
    # Caching mechanisms - LLM response caching
    CacheEntry,
    BaseCache,
    InMemoryCache,
    ResponseCache,
    CachedNode,
)

from .rate_limiting import (
    # Rate limiting - Token bucket algorithm
    TokenBucket,
    RateLimiter,
    MultiTenantRateLimiter,
    RateLimitedNode,
)

from .prompts import (
    # Prompt templates - Reusable prompts
    Message,
    PromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
    FewShotPromptTemplate,
    PromptLibrary,
    get_default_library,
)

from .batch import (
    # Batch processing - Parallel execution
    BatchResult,
    BatchProcessor,
    MapNode,
    FilterNode,
    ReduceNode,
)

from .tools import (
    # Tools - Auto schema inference
    StructuredTool,
    ToolRegistry,
    infer_schema_from_function,
    parse_docstring,
)

from .memory import (
    # Memory - Session-based chat history
    BaseChatHistory,
    InMemoryChatHistory,
    SlidingWindowChatHistory,
    RedisChatHistory,
    SessionHistoryManager,
    ChatHistoryNode,
    create_inmemory_history_manager,
    create_redis_history_manager,
)

from .memory_db import (
    # Memory DB - Database-backed persistent storage
    DatabaseAdapter,
    DatabaseChatHistory,
    create_database_history_manager,
)

from .streaming import (
    # Streaming - Fine-grained event streaming
    StreamEventType,
    StreamEvent,
    StreamingContext,
    StreamingNode,
    StreamEventFilter,
    TokenCollector,
    collect_tokens,
    collect_node_outputs,
)

__version__ = "0.1.0"
__all__ = [
    # Core Types
    "NodeType",
    "CommonNodeTypes",
    "NodeStatus",
    # Context Classes
    "NodeContext",
    "WorkflowContext",
    # Node Interfaces
    "IStepNode",
    "BaseNode",
    # Result Classes
    "NodeResult",
    "Answer",
    "NodeChunk",
    # Executor
    "WorkflowExecutor",
    # Chaining API
    "ChainableNode",
    "NodeChain",
    "FunctionNode",
    "PassthroughNode",
    # Multi-backend
    "IWorkflowBackend",
    "BackendRegistry",
    "NadooBackend",
    # Resilience
    "RetryPolicy",
    "RetryableNode",
    "FallbackNode",
    # Parsers
    "OutputParser",
    "StructuredOutputParser",
    "JsonOutputParser",
    "StringOutputParser",
    "ParserNode",
    "RetryableParserNode",
    # Callbacks
    "CallbackEvent",
    "BaseCallbackHandler",
    "CallbackManager",
    "ConsoleCallbackHandler",
    "LoggingCallbackHandler",
    # Caching
    "CacheEntry",
    "BaseCache",
    "InMemoryCache",
    "ResponseCache",
    "CachedNode",
    # Rate Limiting
    "TokenBucket",
    "RateLimiter",
    "MultiTenantRateLimiter",
    "RateLimitedNode",
    # Prompts
    "Message",
    "PromptTemplate",
    "ChatPromptTemplate",
    "MessagesPlaceholder",
    "FewShotPromptTemplate",
    "PromptLibrary",
    "get_default_library",
    # Batch
    "BatchResult",
    "BatchProcessor",
    "MapNode",
    "FilterNode",
    "ReduceNode",
    # Tools
    "StructuredTool",
    "ToolRegistry",
    "infer_schema_from_function",
    "parse_docstring",
    # Memory
    "BaseChatHistory",
    "InMemoryChatHistory",
    "SlidingWindowChatHistory",
    "RedisChatHistory",
    "SessionHistoryManager",
    "ChatHistoryNode",
    "create_inmemory_history_manager",
    "create_redis_history_manager",
    # Memory DB
    "DatabaseAdapter",
    "DatabaseChatHistory",
    "create_database_history_manager",
    # Streaming
    "StreamEventType",
    "StreamEvent",
    "StreamingContext",
    "StreamingNode",
    "StreamEventFilter",
    "TokenCollector",
    "collect_tokens",
    "collect_node_outputs",
]
