"""
Callback System for Nadoo Flow
콜백 시스템 - 관찰성, 모니터링, 로깅
"""

import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Optional

from .base import NodeContext, NodeResult, WorkflowContext

logger = logging.getLogger(__name__)


@dataclass
class CallbackEvent:
    """콜백 이벤트 데이터

    모든 콜백 메서드에 전달되는 표준화된 이벤트 객체
    """

    event_type: str
    """이벤트 타입 (node_start, node_end, workflow_start, etc.)"""

    workflow_id: str
    """워크플로우 ID"""

    node_id: Optional[str] = None
    """노드 ID (노드 이벤트인 경우)"""

    node_type: Optional[str] = None
    """노드 타입"""

    timestamp: float = field(default_factory=time.time)
    """이벤트 발생 시간"""

    data: dict[str, Any] = field(default_factory=dict)
    """이벤트 데이터"""

    parent_run_id: Optional[str] = None
    """부모 실행 ID (계층 구조)"""

    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    """현재 실행 ID"""

    tags: list[str] = field(default_factory=list)
    """태그 목록"""

    metadata: dict[str, Any] = field(default_factory=dict)
    """메타데이터"""


class BaseCallbackHandler(ABC):
    """콜백 핸들러 베이스 클래스

    모든 콜백 핸들러는 이 클래스를 상속받아 구현합니다.

    Example:
        class MyCallback(BaseCallbackHandler):
            def on_node_start(self, event: CallbackEvent):
                print(f"Node {event.node_id} started")

            def on_node_end(self, event: CallbackEvent):
                print(f"Node {event.node_id} completed")
    """

    def on_workflow_start(self, event: CallbackEvent):
        """워크플로우 시작 시"""
        pass

    def on_workflow_end(self, event: CallbackEvent):
        """워크플로우 종료 시"""
        pass

    def on_workflow_error(self, event: CallbackEvent):
        """워크플로우 에러 시"""
        pass

    def on_node_start(self, event: CallbackEvent):
        """노드 시작 시"""
        pass

    def on_node_end(self, event: CallbackEvent):
        """노드 종료 시"""
        pass

    def on_node_error(self, event: CallbackEvent):
        """노드 에러 시"""
        pass

    def on_llm_start(self, event: CallbackEvent):
        """LLM 호출 시작 시"""
        pass

    def on_llm_end(self, event: CallbackEvent):
        """LLM 호출 종료 시"""
        pass

    def on_llm_token(self, event: CallbackEvent):
        """LLM 토큰 스트리밍 시"""
        pass

    def on_tool_start(self, event: CallbackEvent):
        """도구 실행 시작 시"""
        pass

    def on_tool_end(self, event: CallbackEvent):
        """도구 실행 종료 시"""
        pass

    def on_tool_error(self, event: CallbackEvent):
        """도구 실행 에러 시"""
        pass

    def on_custom_event(self, event: CallbackEvent):
        """사용자 정의 이벤트"""
        pass


class CallbackManager:
    """콜백 매니저

    여러 콜백 핸들러를 관리하고 이벤트를 전파합니다.

    Example:
        manager = CallbackManager()
        manager.add_handler(ConsoleHandler())
        manager.add_handler(LoggingHandler())

        # 워크플로우에 설정
        workflow_context.callback_manager = manager
    """

    def __init__(
        self,
        handlers: list[BaseCallbackHandler] | None = None,
        inheritable_handlers: list[BaseCallbackHandler] | None = None,
        parent_run_id: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None
    ):
        """
        Args:
            handlers: 이 매니저의 핸들러들
            inheritable_handlers: 하위로 전파될 핸들러들
            parent_run_id: 부모 실행 ID
            tags: 태그 목록
            metadata: 메타데이터
        """
        self.handlers = handlers or []
        self.inheritable_handlers = inheritable_handlers or []
        self.parent_run_id = parent_run_id
        self.tags = tags or []
        self.metadata = metadata or {}
        self.run_id = str(uuid.uuid4())

    def add_handler(self, handler: BaseCallbackHandler, inheritable: bool = False):
        """핸들러 추가"""
        if inheritable:
            self.inheritable_handlers.append(handler)
        else:
            self.handlers.append(handler)

    def remove_handler(self, handler: BaseCallbackHandler):
        """핸들러 제거"""
        if handler in self.handlers:
            self.handlers.remove(handler)
        if handler in self.inheritable_handlers:
            self.inheritable_handlers.remove(handler)

    def _emit(self, event: CallbackEvent):
        """모든 핸들러에게 이벤트 전파"""
        all_handlers = self.handlers + self.inheritable_handlers

        for handler in all_handlers:
            try:
                # 이벤트 타입에 따라 적절한 메서드 호출
                method_name = f"on_{event.event_type}"
                method = getattr(handler, method_name, None)

                if method and callable(method):
                    method(event)

            except Exception as e:
                logger.error(
                    f"Callback handler {handler.__class__.__name__} "
                    f"failed on {event.event_type}: {e}"
                )

    def on_workflow_start(
        self,
        workflow_context: WorkflowContext,
        inputs: dict[str, Any] | None = None
    ):
        """워크플로우 시작 이벤트"""
        event = CallbackEvent(
            event_type="workflow_start",
            workflow_id=workflow_context.workflow_id,
            parent_run_id=self.parent_run_id,
            run_id=self.run_id,
            tags=self.tags,
            metadata=self.metadata,
            data={
                "inputs": inputs or {},
                "application_id": workflow_context.application_id,
                "user_id": workflow_context.user_id
            }
        )
        self._emit(event)

    def on_workflow_end(self, workflow_context: WorkflowContext):
        """워크플로우 종료 이벤트"""
        execution_time = None
        if workflow_context.end_time and workflow_context.start_time:
            execution_time = workflow_context.end_time - workflow_context.start_time

        event = CallbackEvent(
            event_type="workflow_end",
            workflow_id=workflow_context.workflow_id,
            parent_run_id=self.parent_run_id,
            run_id=self.run_id,
            tags=self.tags,
            metadata=self.metadata,
            data={
                "status": workflow_context.status.value,
                "execution_time": execution_time,
                "execution_path": workflow_context.execution_path,
                "error": workflow_context.error
            }
        )
        self._emit(event)

    def on_workflow_error(self, workflow_context: WorkflowContext, error: Exception):
        """워크플로우 에러 이벤트"""
        event = CallbackEvent(
            event_type="workflow_error",
            workflow_id=workflow_context.workflow_id,
            parent_run_id=self.parent_run_id,
            run_id=self.run_id,
            tags=self.tags,
            metadata=self.metadata,
            data={
                "error": str(error),
                "error_type": type(error).__name__,
                "current_node_id": workflow_context.current_node_id
            }
        )
        self._emit(event)

    def on_node_start(
        self,
        node_context: NodeContext,
        workflow_context: WorkflowContext
    ):
        """노드 시작 이벤트"""
        event = CallbackEvent(
            event_type="node_start",
            workflow_id=workflow_context.workflow_id,
            node_id=node_context.node_id,
            node_type=node_context.node_type,
            parent_run_id=self.parent_run_id,
            run_id=self.run_id,
            tags=self.tags,
            metadata=self.metadata,
            data={
                "input_data": node_context.input_data,
                "variables": node_context.variables
            }
        )
        self._emit(event)

    def on_node_end(
        self,
        node_context: NodeContext,
        workflow_context: WorkflowContext,
        result: NodeResult
    ):
        """노드 종료 이벤트"""
        event = CallbackEvent(
            event_type="node_end",
            workflow_id=workflow_context.workflow_id,
            node_id=node_context.node_id,
            node_type=node_context.node_type,
            parent_run_id=self.parent_run_id,
            run_id=self.run_id,
            tags=self.tags,
            metadata=self.metadata,
            data={
                "status": node_context.status.value,
                "execution_time": node_context.execution_time,
                "output_data": node_context.output_data,
                "success": result.success,
                "error": result.error,
                "result_metadata": result.metadata
            }
        )
        self._emit(event)

    def on_node_error(
        self,
        node_context: NodeContext,
        workflow_context: WorkflowContext,
        error: Exception
    ):
        """노드 에러 이벤트"""
        event = CallbackEvent(
            event_type="node_error",
            workflow_id=workflow_context.workflow_id,
            node_id=node_context.node_id,
            node_type=node_context.node_type,
            parent_run_id=self.parent_run_id,
            run_id=self.run_id,
            tags=self.tags,
            metadata=self.metadata,
            data={
                "error": str(error),
                "error_type": type(error).__name__,
                "execution_time": node_context.execution_time
            }
        )
        self._emit(event)

    def on_llm_start(
        self,
        node_id: str,
        workflow_id: str,
        prompts: list[str] | None = None,
        model_name: str | None = None
    ):
        """LLM 시작 이벤트"""
        event = CallbackEvent(
            event_type="llm_start",
            workflow_id=workflow_id,
            node_id=node_id,
            node_type="llm",
            parent_run_id=self.parent_run_id,
            run_id=self.run_id,
            tags=self.tags,
            metadata=self.metadata,
            data={
                "prompts": prompts,
                "model_name": model_name
            }
        )
        self._emit(event)

    def on_llm_end(
        self,
        node_id: str,
        workflow_id: str,
        response: str | None = None,
        token_usage: dict[str, int] | None = None
    ):
        """LLM 종료 이벤트"""
        event = CallbackEvent(
            event_type="llm_end",
            workflow_id=workflow_id,
            node_id=node_id,
            node_type="llm",
            parent_run_id=self.parent_run_id,
            run_id=self.run_id,
            tags=self.tags,
            metadata=self.metadata,
            data={
                "response": response,
                "token_usage": token_usage
            }
        )
        self._emit(event)

    def on_llm_token(
        self,
        node_id: str,
        workflow_id: str,
        token: str
    ):
        """LLM 토큰 스트리밍 이벤트"""
        event = CallbackEvent(
            event_type="llm_token",
            workflow_id=workflow_id,
            node_id=node_id,
            node_type="llm",
            parent_run_id=self.parent_run_id,
            run_id=self.run_id,
            tags=self.tags,
            metadata=self.metadata,
            data={"token": token}
        )
        self._emit(event)

    def on_tool_start(
        self,
        node_id: str,
        workflow_id: str,
        tool_name: str,
        tool_input: dict[str, Any]
    ):
        """도구 시작 이벤트"""
        event = CallbackEvent(
            event_type="tool_start",
            workflow_id=workflow_id,
            node_id=node_id,
            node_type="tool",
            parent_run_id=self.parent_run_id,
            run_id=self.run_id,
            tags=self.tags,
            metadata=self.metadata,
            data={
                "tool_name": tool_name,
                "tool_input": tool_input
            }
        )
        self._emit(event)

    def on_tool_end(
        self,
        node_id: str,
        workflow_id: str,
        tool_name: str,
        tool_output: Any
    ):
        """도구 종료 이벤트"""
        event = CallbackEvent(
            event_type="tool_end",
            workflow_id=workflow_id,
            node_id=node_id,
            node_type="tool",
            parent_run_id=self.parent_run_id,
            run_id=self.run_id,
            tags=self.tags,
            metadata=self.metadata,
            data={
                "tool_name": tool_name,
                "tool_output": tool_output
            }
        )
        self._emit(event)

    def on_tool_error(
        self,
        node_id: str,
        workflow_id: str,
        tool_name: str,
        error: Exception
    ):
        """도구 에러 이벤트"""
        event = CallbackEvent(
            event_type="tool_error",
            workflow_id=workflow_id,
            node_id=node_id,
            node_type="tool",
            parent_run_id=self.parent_run_id,
            run_id=self.run_id,
            tags=self.tags,
            metadata=self.metadata,
            data={
                "tool_name": tool_name,
                "error": str(error),
                "error_type": type(error).__name__
            }
        )
        self._emit(event)


class ConsoleCallbackHandler(BaseCallbackHandler):
    """콘솔 출력 콜백 핸들러

    디버깅용으로 모든 이벤트를 콘솔에 출력합니다.

    Example:
        manager = CallbackManager()
        manager.add_handler(ConsoleCallbackHandler())
    """

    def __init__(self, verbose: bool = True, colors: bool = True):
        """
        Args:
            verbose: 상세 출력 여부
            colors: 색상 사용 여부 (ANSI colors)
        """
        self.verbose = verbose
        self.colors = colors

    def _print(self, message: str, color: str | None = None):
        """색상 지원 출력"""
        if self.colors and color:
            colors_map = {
                "green": "\033[92m",
                "red": "\033[91m",
                "yellow": "\033[93m",
                "blue": "\033[94m",
                "gray": "\033[90m",
                "reset": "\033[0m"
            }
            print(f"{colors_map.get(color, '')}{message}{colors_map['reset']}")
        else:
            print(message)

    def on_workflow_start(self, event: CallbackEvent):
        """워크플로우 시작"""
        self._print(f"\n🚀 Workflow started: {event.workflow_id}", "blue")
        if self.verbose and event.data.get("inputs"):
            self._print(f"   Inputs: {event.data['inputs']}", "gray")

    def on_workflow_end(self, event: CallbackEvent):
        """워크플로우 종료"""
        status = event.data.get("status")
        exec_time = event.data.get("execution_time")

        if status == "success":
            self._print(f"✅ Workflow completed: {event.workflow_id}", "green")
        else:
            self._print(f" Workflow failed: {event.workflow_id}", "red")

        if exec_time:
            self._print(f"   Execution time: {exec_time:.2f}s", "gray")

    def on_node_start(self, event: CallbackEvent):
        """노드 시작"""
        if self.verbose:
            self._print(
                f"  ▶️  Node started: {event.node_id} ({event.node_type})",
                "blue"
            )

    def on_node_end(self, event: CallbackEvent):
        """노드 종료"""
        if self.verbose:
            status = "✓" if event.data.get("success") else "✗"
            exec_time = event.data.get("execution_time", 0)
            self._print(
                f"  {status} Node completed: {event.node_id} ({exec_time:.2f}s)",
                "green" if event.data.get("success") else "red"
            )

    def on_llm_token(self, event: CallbackEvent):
        """LLM 토큰"""
        if self.verbose:
            print(event.data["token"], end="", flush=True)


class LoggingCallbackHandler(BaseCallbackHandler):
    """로깅 콜백 핸들러

    Python logging 모듈을 사용하여 이벤트를 기록합니다.

    Example:
        manager = CallbackManager()
        manager.add_handler(LoggingCallbackHandler())
    """

    def __init__(self, logger_name: str = "nadoo_flow", level: int = logging.INFO):
        """
        Args:
            logger_name: 로거 이름
            level: 로깅 레벨
        """
        self.logger = logging.getLogger(logger_name)
        self.level = level

    def on_workflow_start(self, event: CallbackEvent):
        self.logger.log(
            self.level,
            f"Workflow started: {event.workflow_id}",
            extra={"event": event.data}
        )

    def on_workflow_end(self, event: CallbackEvent):
        self.logger.log(
            self.level,
            f"Workflow ended: {event.workflow_id} - {event.data.get('status')}",
            extra={"event": event.data}
        )

    def on_workflow_error(self, event: CallbackEvent):
        self.logger.error(
            f"Workflow error: {event.workflow_id} - {event.data.get('error')}",
            extra={"event": event.data}
        )

    def on_node_start(self, event: CallbackEvent):
        self.logger.log(
            self.level,
            f"Node started: {event.node_id} ({event.node_type})",
            extra={"event": event.data}
        )

    def on_node_end(self, event: CallbackEvent):
        self.logger.log(
            self.level,
            f"Node ended: {event.node_id} - {event.data.get('status')}",
            extra={"event": event.data}
        )

    def on_node_error(self, event: CallbackEvent):
        self.logger.error(
            f"Node error: {event.node_id} - {event.data.get('error')}",
            extra={"event": event.data}
        )
