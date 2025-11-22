"""
Flow System Base Classes
워크플로우 시스템 기본 클래스
"""

import json
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Type, Union

import logging

logger = logging.getLogger(__name__)


# Node types are just strings - users can define custom types
NodeType = str


class CommonNodeTypes:
    """Common node type constants for convenience

    Users can use these predefined types or define their own custom types.
    Since NodeType is just a string, any value is valid.

    Example:
        # Using predefined types
        node_type = CommonNodeTypes.START

        # Using custom types
        node_type = "my-custom-ai-agent"
    """

    # Core workflow control
    START = "start-node"
    END = "end-node"
    CONDITION = "condition-node"
    PARALLEL = "parallel-node"
    LOOP = "loop-node"

    # AI/LLM nodes
    AI_AGENT = "ai-agent-node"
    LLM = "llm"  # Legacy compatibility

    # Data processing
    VARIABLE = "variable-node"
    PYTHON = "python-node"
    TOOL = "tool-node"

    # External integrations
    DATABASE = "database-node"
    MCP = "mcp-node"

    # Generic/extensible
    UNIVERSAL = "universal"
    CUSTOM = "custom"  # For user-defined nodes


class NodeStatus(str, Enum):
    """노드 실행 상태"""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    INTERRUPTED = "interrupted"


@dataclass
class NodeContext:
    """노드 실행 컨텍스트"""

    node_id: str
    node_type: NodeType
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    status: NodeStatus = NodeStatus.PENDING
    variables: dict[str, Any] = field(default_factory=dict)
    input_data: dict[str, Any] = field(default_factory=dict)
    output_data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def execution_time(self) -> float | None:
        """실행 시간 계산"""
        if self.end_time:
            return self.end_time - self.start_time
        return None

    def set_output(self, key: str, value: Any):
        """출력 데이터 설정"""
        self.output_data[key] = value

    def get_input(self, key: str, default: Any = None) -> Any:
        """입력 데이터 가져오기"""
        return self.input_data.get(key, default)

    def set_variable(self, key: str, value: Any):
        """변수 설정"""
        self.variables[key] = value

    def get_variable(self, key: str, default: Any = None) -> Any:
        """변수 가져오기"""
        return self.variables.get(key, default)


@dataclass
class WorkflowContext:
    """워크플로우 실행 컨텍스트"""

    workflow_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    application_id: str | None = None
    user_id: str | None = None
    chat_id: str | None = None
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    global_variables: dict[str, Any] = field(default_factory=dict)
    node_contexts: dict[str, NodeContext] = field(default_factory=dict)
    current_node_id: str | None = None
    execution_path: list[str] = field(default_factory=list)
    status: NodeStatus = NodeStatus.PENDING
    error: str | None = None
    waiting_for: dict[str, Any] | None = None  # Pause information: {type, node_id, reason, message}

    # Agent 2.0 Phase 3-2: SSE & DB Integration
    db: Any = None  # AsyncSession from SQLAlchemy
    workspace_id: str | None = None  # Workspace ID for multi-tenancy
    sse_emitter: Any = None  # SSE event emitter (callable or queue)

    # Additional attributes for node compatibility
    session_id: str | None = None
    node_outputs: dict[str, Any] = field(default_factory=dict)
    error_count: int = 0
    started_at: Any = None  # datetime
    completed_at: Any = None  # datetime

    def add_node_context(self, node_context: NodeContext):
        """노드 컨텍스트 추가"""
        self.node_contexts[node_context.node_id] = node_context
        self.execution_path.append(node_context.node_id)

    def get_node_context(self, node_id: str) -> NodeContext | None:
        """노드 컨텍스트 가져오기"""
        return self.node_contexts.get(node_id)

    def set_global_variable(self, key: str, value: Any):
        """전역 변수 설정"""
        self.global_variables[key] = value

    def get_global_variable(self, key: str, default: Any = None) -> Any:
        """전역 변수 가져오기"""
        return self.global_variables.get(key, default)

    async def emit_sse(self, event: Any):
        """SSE 이벤트 발송 (Agent 2.0)

        Args:
            event: SSE 이벤트 객체 (from src.flow.sse_events)
        """
        if self.sse_emitter is not None:
            try:
                # sse_emitter can be:
                # 1. async function: await sse_emitter(event)
                # 2. queue: await sse_emitter.put(event)
                # 3. sync function: sse_emitter(event)
                if hasattr(self.sse_emitter, "put"):
                    # Queue-like object
                    await self.sse_emitter.put(event)
                elif callable(self.sse_emitter):
                    # Callable
                    import inspect

                    if inspect.iscoroutinefunction(self.sse_emitter):
                        await self.sse_emitter(event)
                    else:
                        self.sse_emitter(event)

                # Yield control to event loop for real-time event processing
                import asyncio

                await asyncio.sleep(0)
            except Exception as e:
                logger.warning(f"Failed to emit SSE event: {e}")

    async def emit_markdown_chunk(self, markdown: str):
        """마크다운 청크 발송
        Args:
            markdown: 마크다운 텍스트 청크
        """
        if not markdown:
            return

        # TEXT_DELTA 이벤트 생성 및 발송 (AI SDK 표준)
        # StreamPartType을 optional import로 처리 (nadoo-flow는 독립 SDK)
        try:
            from src.flow.sse_events import StreamPartType
            text_delta_type = StreamPartType.TEXT_DELTA.value
        except ImportError:
            # Fallback: nadoo-flow가 독립적으로 사용될 때
            text_delta_type = "text-delta"

        event = {
            "type": text_delta_type,
            "text": markdown,
            "node_id": getattr(self, "current_node_id", None),
            "workflow_id": self.workflow_id,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        logger.info(f"[TEXT_DELTA] Emitting text delta: {markdown[:100]}...")  # Debug log
        await self.emit_sse(event)


class NodeResult:
    """노드 실행 결과"""

    def __init__(
        self,
        success: bool,
        output: dict[str, Any] | None = None,
        next_node_id: str | None = None,
        conditional_next: dict[str, str] | None = None,
        error: str | None = None,
        should_interrupt: bool = False,
        metadata: dict[str, Any] | None = None,
    ):
        self.success = success
        self.output = output or {}
        self.next_node_id = next_node_id
        self.conditional_next = conditional_next  # {"accept": "node-id-1", "improve": "node-id-2"}
        self.error = error
        self.should_interrupt = should_interrupt
        self.metadata = metadata or {}


class IStepNode(ABC):
    """스텝 노드 인터페이스"""

    def __init__(self, node_id: str, node_type: NodeType, name: str, config: dict[str, Any]):
        self.node_id = node_id
        self.node_type = node_type
        self.name = name
        self.config = config
        self.next_nodes: list[str] = []
        self.condition_branches: dict[str, str] = {}

    @abstractmethod
    async def execute(self, node_context: NodeContext, workflow_context: WorkflowContext) -> NodeResult:
        """노드 실행"""
        pass

    @abstractmethod
    async def validate(self) -> bool:
        """노드 설정 검증"""
        pass

    def add_next_node(self, node_id: str):
        """다음 노드 추가"""
        if node_id not in self.next_nodes:
            self.next_nodes.append(node_id)

    def add_condition_branch(self, condition: str, node_id: str):
        """조건 분기 추가"""
        self.condition_branches[condition] = node_id

    async def pre_execute(self, node_context: NodeContext, workflow_context: WorkflowContext):
        """실행 전 처리"""
        node_context.status = NodeStatus.RUNNING
        node_context.start_time = time.time()
        logger.info(f"Executing node: {self.node_id} ({self.node_type})")

    async def post_execute(self, node_context: NodeContext, workflow_context: WorkflowContext, result: NodeResult):
        """실행 후 처리"""
        node_context.end_time = time.time()

        if result.success:
            node_context.status = NodeStatus.SUCCESS
            node_context.output_data = result.output
        else:
            node_context.status = NodeStatus.FAILED
            node_context.error = result.error

        logger.info(
            f"Node {self.node_id} completed: {node_context.status} "
            f"(execution_time: {node_context.execution_time:.2f}s)"
        )

    def get_next_node_id(self, result: NodeResult, route_key: str | None = None) -> str | None:
        """다음 노드 ID 결정

        Args:
            result: 노드 실행 결과
            route_key: 조건부 라우팅 키 (예: "accept", "improve")

        Returns:
            다음 노드 ID 또는 None
        """
        # 조건부 라우팅이 있고 route_key가 지정된 경우
        if result.conditional_next and route_key:
            return result.conditional_next.get(route_key)

        # 조건부 라우팅이 있지만 route_key가 없는 경우, output에서 키 찾기
        if result.conditional_next and "route" in result.output:
            route_key = result.output.get("route")
            if route_key in result.conditional_next:
                return result.conditional_next[route_key]

        # 명시적으로 지정된 다음 노드가 있으면 반환
        if result.next_node_id:
            return result.next_node_id

        # 기본 다음 노드 반환
        if self.next_nodes:
            return self.next_nodes[0]

        return None


class BaseNode(IStepNode):
    """기본 노드 구현"""

    async def validate(self) -> bool:
        """기본 검증"""
        return True

    async def execute(self, node_context: NodeContext, workflow_context: WorkflowContext) -> NodeResult:
        """기본 실행 - 하위 클래스에서 구현"""
        return NodeResult(success=True, output={"message": f"Node {self.node_id} executed"})


class Answer:
    """응답 객체"""

    def __init__(self, content: str, type: str = "text", metadata: dict[str, Any] | None = None):
        self.content = content
        self.type = type  # text, markdown, html, json, etc.
        self.metadata = metadata or {}
        self.timestamp = datetime.now(UTC)

    def to_dict(self) -> dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "content": self.content,
            "type": self.type,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


class NodeChunk:
    """노드 청크 - 스트리밍 응답용"""

    def __init__(self, node_id: str, chunk_type: str, content: Any, is_final: bool = False):
        self.node_id = node_id
        self.chunk_type = chunk_type  # text, data, error, etc.
        self.content = content
        self.is_final = is_final
        self.timestamp = time.time()

    def to_json(self) -> str:
        """JSON 문자열로 변환"""
        return json.dumps(
            {
                "node_id": self.node_id,
                "chunk_type": self.chunk_type,
                "content": self.content,
                "is_final": self.is_final,
                "timestamp": self.timestamp,
            }
        )


class WorkflowExecutor:
    """워크플로우 실행 엔진"""

    def __init__(self):
        self.nodes: dict[str, IStepNode] = {}
        self.start_node_id: str | None = None

    def add_node(self, node: IStepNode):
        """노드 추가"""
        self.nodes[node.node_id] = node

        # 시작 노드 설정
        if node.node_type == CommonNodeTypes.START:
            self.start_node_id = node.node_id

    def get_node(self, node_id: str) -> IStepNode | None:
        """노드 가져오기"""
        return self.nodes.get(node_id)

    async def execute(
        self, workflow_context: WorkflowContext, initial_input: dict[str, Any] | None = None
    ) -> WorkflowContext:
        """워크플로우 실행"""
        try:
            workflow_context.status = NodeStatus.RUNNING

            # 시작 노드 찾기
            if not self.start_node_id:
                raise ValueError("No start node defined")

            current_node_id = self.start_node_id

            # 초기 입력 설정
            if initial_input:
                workflow_context.global_variables.update(initial_input)

            # 노드 순차 실행
            while current_node_id:
                node = self.get_node(current_node_id)
                if not node:
                    raise ValueError(f"Node {current_node_id} not found")

                # 노드 컨텍스트 생성
                node_context = NodeContext(
                    node_id=node.node_id,
                    node_type=node.node_type,
                    input_data=self._prepare_node_input(node, workflow_context),
                )

                # 노드 실행
                await node.pre_execute(node_context, workflow_context)
                result = await node.execute(node_context, workflow_context)
                await node.post_execute(node_context, workflow_context, result)

                # 컨텍스트 저장
                workflow_context.add_node_context(node_context)
                workflow_context.current_node_id = current_node_id

                # 인터럽트 체크
                if result.should_interrupt:
                    workflow_context.status = NodeStatus.INTERRUPTED
                    break

                # 실패 체크
                if not result.success:
                    workflow_context.status = NodeStatus.FAILED
                    workflow_context.error = result.error
                    break

                # 다음 노드 결정
                current_node_id = node.get_next_node_id(result)

                # 종료 노드 체크
                if current_node_id and self.get_node(current_node_id):
                    if self.get_node(current_node_id).node_type == CommonNodeTypes.END:
                        current_node_id = None

            # 완료 상태 설정
            if workflow_context.status == NodeStatus.RUNNING:
                workflow_context.status = NodeStatus.SUCCESS

            workflow_context.end_time = time.time()

        except Exception as e:
            logger.error(f"Workflow execution error: {e!s}")
            workflow_context.status = NodeStatus.FAILED
            workflow_context.error = str(e)
            workflow_context.end_time = time.time()

        return workflow_context

    async def execute_streaming(self, workflow_context: WorkflowContext, initial_input: dict[str, Any] | None = None):
        """스트리밍 워크플로우 실행"""
        # 스트리밍 실행 로직
        # yield NodeChunk 객체들
        pass

    def _prepare_node_input(self, node: IStepNode, workflow_context: WorkflowContext) -> dict[str, Any]:
        """노드 입력 데이터 준비"""
        input_data = {}

        # 전역 변수 복사
        input_data.update(workflow_context.global_variables)

        # 이전 노드 출력 추가
        if workflow_context.execution_path:
            last_node_id = workflow_context.execution_path[-1]
            last_context = workflow_context.get_node_context(last_node_id)
            if last_context:
                input_data["previous_output"] = last_context.output_data

        return input_data

    async def validate(self) -> bool:
        """워크플로우 검증

        1. 시작 노드 존재 여부
        2. 개별 노드 검증
        3. 순환 참조 체크 (Cycle detection)
        4. 도달 가능성 체크 (Reachability)
        5. 종료 노드 도달 가능성
        """
        if not self.start_node_id:
            logger.error("No start node defined")
            return False

        # 모든 노드 검증
        for node_id, node in self.nodes.items():
            if not await node.validate():
                logger.error(f"Node {node_id} validation failed")
                return False

        # 연결성 검증
        if not self._validate_graph_structure():
            return False

        return True

    def _validate_graph_structure(self) -> bool:
        """그래프 구조 검증 (순환 참조, 연결성 체크)"""

        # 그래프 구조 생성 (adjacency list)
        graph: dict[str, list[str]] = {}
        for node_id, node in self.nodes.items():
            graph[node_id] = node.next_nodes if hasattr(node, 'next_nodes') else []

        # 1. 순환 참조 체크 (Cycle Detection using DFS)
        if self._has_cycle(graph):
            logger.error("Circular dependency detected in workflow")
            return False

        # 2. 도달 가능성 체크 (Reachability Analysis)
        reachable = self._get_reachable_nodes(graph, self.start_node_id)
        unreachable = set(self.nodes.keys()) - reachable

        if unreachable:
            logger.warning(f"Unreachable nodes detected: {unreachable}")
            # 경고만 출력하고 검증은 통과 (고립된 노드는 치명적이지 않음)

        # 3. 종료 노드 도달 가능성
        end_nodes = [
            node_id for node_id, node in self.nodes.items()
            if node.node_type == CommonNodeTypes.END
        ]

        if end_nodes:
            # END 노드가 정의된 경우, 최소 하나는 도달 가능해야 함
            reachable_end_nodes = [n for n in end_nodes if n in reachable]
            if not reachable_end_nodes:
                logger.error("No reachable END node found in workflow")
                return False

        logger.info("Workflow graph structure validation passed")
        return True

    def _has_cycle(self, graph: dict[str, list[str]]) -> bool:
        """DFS를 사용한 순환 참조 감지

        Args:
            graph: Adjacency list representation of the workflow graph

        Returns:
            True if cycle exists, False otherwise
        """
        visited: set[str] = set()
        rec_stack: set[str] = set()  # Recursion stack for cycle detection

        def dfs(node: str) -> bool:
            """DFS helper function"""
            visited.add(node)
            rec_stack.add(node)

            # 인접 노드 방문
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True  # Cycle found
                elif neighbor in rec_stack:
                    # Back edge found - cycle detected
                    logger.error(f"Cycle detected: {node} -> {neighbor}")
                    return True

            rec_stack.remove(node)
            return False

        # 모든 노드에서 DFS 수행 (disconnected components 처리)
        for node_id in graph:
            if node_id not in visited:
                if dfs(node_id):
                    return True

        return False

    def _get_reachable_nodes(self, graph: dict[str, list[str]], start_node: str) -> set[str]:
        """시작 노드로부터 도달 가능한 모든 노드 반환 (BFS)

        Args:
            graph: Adjacency list representation
            start_node: Starting node ID

        Returns:
            Set of reachable node IDs
        """
        reachable: set[str] = set()
        queue: list[str] = [start_node]

        while queue:
            current = queue.pop(0)

            if current not in reachable:
                reachable.add(current)

                # 인접 노드를 큐에 추가
                for neighbor in graph.get(current, []):
                    if neighbor not in reachable:
                        queue.append(neighbor)

        return reachable
