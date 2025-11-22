"""
Tools for Nadoo Flow
도구 - 함수를 도구로 자동 변환
"""

import inspect
import json
import logging
from typing import Any, Callable, get_type_hints, Literal
from pydantic import BaseModel, Field, create_model

from .base import BaseNode, NodeContext, NodeResult, WorkflowContext

logger = logging.getLogger(__name__)


def infer_schema_from_function(func: Callable) -> type[BaseModel]:
    """함수 시그니처에서 Pydantic 스키마 자동 생성

    Args:
        func: 스키마를 생성할 함수

    Returns:
        Pydantic 모델 클래스

    Example:
        def search(query: str, limit: int = 10) -> list[str]:
            '''Search for items'''
            pass

        schema = infer_schema_from_function(search)
        # Returns Pydantic model with query (str) and limit (int) fields
    """
    # 함수 시그니처 가져오기
    sig = inspect.signature(func)
    type_hints = get_type_hints(func)

    # Pydantic 필드 생성
    fields = {}

    for param_name, param in sig.parameters.items():
        # 타입 힌트 가져오기
        param_type = type_hints.get(param_name, Any)

        # 기본값 확인
        if param.default is inspect.Parameter.empty:
            # 필수 필드
            fields[param_name] = (param_type, ...)
        else:
            # 선택 필드 (기본값 있음)
            fields[param_name] = (param_type, param.default)

    # Pydantic 모델 동적 생성
    model_name = f"{func.__name__.title()}Schema"
    schema_model = create_model(model_name, **fields)

    return schema_model


def parse_docstring(func: Callable) -> dict[str, str]:
    """Docstring 파싱하여 description 추출

    Google 스타일과 NumPy 스타일 docstring 지원

    Args:
        func: Docstring을 파싱할 함수

    Returns:
        {
            "description": "함수 설명",
            "param_name": "파라미터 설명",
            ...
        }
    """
    docstring = inspect.getdoc(func)

    if not docstring:
        return {"description": ""}

    lines = docstring.split("\n")
    result = {"description": ""}
    param_descriptions = {}

    # 첫 줄을 description으로
    current_section = "description"
    description_lines = []
    in_args_section = False

    for line in lines:
        line = line.strip()

        # Args 섹션 시작
        if line.lower() in ("args:", "arguments:", "parameters:"):
            in_args_section = True
            current_section = "args"
            result["description"] = " ".join(description_lines).strip()
            continue

        # Returns 섹션 (Args 섹션 종료)
        if line.lower() in ("returns:", "return:"):
            in_args_section = False
            continue

        if current_section == "description" and not in_args_section:
            if line:
                description_lines.append(line)

        elif in_args_section:
            # 파라미터 설명 파싱
            # 형식: "param_name: description" 또는 "param_name (type): description"
            if ":" in line:
                parts = line.split(":", 1)
                param_part = parts[0].strip()
                desc_part = parts[1].strip() if len(parts) > 1 else ""

                # 타입 제거 (괄호 안)
                if "(" in param_part:
                    param_part = param_part.split("(")[0].strip()

                param_descriptions[param_part] = desc_part

    # Description이 아직 설정 안 됐으면
    if not result["description"] and description_lines:
        result["description"] = " ".join(description_lines).strip()

    # 파라미터 설명 병합
    result.update(param_descriptions)

    return result


class StructuredTool(BaseNode):
    """구조화된 도구 노드

    함수를 자동으로 도구 노드로 변환합니다.

    Example:
        def search_database(
            query: str,
            limit: int = 10,
            filters: dict[str, Any] | None = None
        ) -> list[dict]:
            '''Search the knowledge base.

            Args:
                query: The search query
                limit: Maximum number of results
                filters: Optional filters to apply
            '''
            return db.search(query, limit, filters)

        tool = StructuredTool.from_function(
            func=search_database,
            parse_docstring=True
        )

        # 자동으로 스키마 생성 및 검증
    """

    def __init__(
        self,
        node_id: str,
        func: Callable,
        args_schema: type[BaseModel] | None = None,
        description: str | None = None,
        return_direct: bool = False,
        name: str | None = None,
        config: dict[str, Any] | None = None
    ):
        """
        Args:
            node_id: 노드 ID
            func: 실행할 함수
            args_schema: 인자 스키마 (None이면 자동 생성)
            description: 도구 설명
            return_direct: 결과를 바로 반환할지 여부
            name: 도구 이름
            config: 노드 설정
        """
        super().__init__(
            node_id=node_id,
            node_type="tool",
            name=name or func.__name__,
            config=config or {}
        )

        self.func = func
        self.args_schema = args_schema
        self.description = description or func.__doc__ or ""
        self.return_direct = return_direct

        # 비동기 함수 여부
        self.is_async = inspect.iscoroutinefunction(func)

    @classmethod
    def from_function(
        cls,
        func: Callable,
        node_id: str | None = None,
        coroutine: Callable | None = None,
        name: str | None = None,
        description: str | None = None,
        args_schema: type[BaseModel] | None = None,
        infer_schema: bool = True,
        parse_docstring: bool = True,
        return_direct: bool = False
    ) -> "StructuredTool":
        """함수에서 도구 생성

        Args:
            func: 실행할 함수
            node_id: 노드 ID (None이면 함수 이름 사용)
            coroutine: 비동기 버전 함수 (선택)
            name: 도구 이름
            description: 도구 설명
            args_schema: 인자 스키마 (None이면 자동 생성)
            infer_schema: 스키마 자동 추론 여부
            parse_docstring: Docstring 파싱 여부
            return_direct: 결과를 바로 반환할지 여부

        Returns:
            StructuredTool 인스턴스
        """
        # Node ID
        if node_id is None:
            node_id = f"tool_{func.__name__}"

        # 스키마 자동 생성
        if args_schema is None and infer_schema:
            try:
                args_schema = infer_schema_from_function(func)
            except Exception as e:
                logger.warning(f"Failed to infer schema for {func.__name__}: {e}")

        # Docstring 파싱
        if parse_docstring:
            docstring_info = parse_docstring(func)

            if description is None:
                description = docstring_info.get("description", "")

            # 스키마에 description 추가
            if args_schema:
                for field_name, field_info in args_schema.model_fields.items():
                    if field_name in docstring_info:
                        field_info.description = docstring_info[field_name]

        # 비동기 함수 선택
        actual_func = coroutine if coroutine else func

        return cls(
            node_id=node_id,
            func=actual_func,
            args_schema=args_schema,
            description=description,
            return_direct=return_direct,
            name=name,
            config={}
        )

    async def execute(
        self,
        node_context: NodeContext,
        workflow_context: WorkflowContext
    ) -> NodeResult:
        """도구 실행"""
        try:
            # 입력 데이터 가져오기
            input_data = {**node_context.input_data}

            # 스키마 검증
            if self.args_schema:
                try:
                    validated = self.args_schema(**input_data)
                    input_data = validated.model_dump()
                except Exception as e:
                    return NodeResult(
                        success=False,
                        error=f"Input validation failed: {e}"
                    )

            # 함수 실행
            if self.is_async:
                result = await self.func(**input_data)
            else:
                result = self.func(**input_data)

            # 결과 래핑
            if not isinstance(result, dict):
                result = {"result": result}

            return NodeResult(
                success=True,
                output=result,
                metadata={"return_direct": self.return_direct}
            )

        except Exception as e:
            logger.error(f"Tool {self.node_id} execution failed: {e}")
            return NodeResult(
                success=False,
                error=str(e)
            )

    def get_input_schema(self) -> dict[str, Any]:
        """입력 스키마 가져오기 (JSON Schema 형식)"""
        if self.args_schema:
            return self.args_schema.model_json_schema()

        return {}

    def to_openai_tool(self) -> dict[str, Any]:
        """OpenAI function calling 형식으로 변환"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.get_input_schema()
            }
        }

    def to_anthropic_tool(self) -> dict[str, Any]:
        """Anthropic tool use 형식으로 변환"""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.get_input_schema()
        }


class ToolRegistry:
    """도구 레지스트리

    여러 도구를 관리합니다.

    Example:
        registry = ToolRegistry()

        registry.register(search_tool)
        registry.register(calculate_tool)

        # OpenAI 형식으로 export
        tools = registry.to_openai_tools()
    """

    def __init__(self):
        self.tools: dict[str, StructuredTool] = {}

    def register(self, tool: StructuredTool):
        """도구 등록"""
        self.tools[tool.name] = tool

    def register_function(
        self,
        func: Callable,
        **kwargs
    ) -> StructuredTool:
        """함수를 도구로 등록

        Args:
            func: 등록할 함수
            **kwargs: StructuredTool.from_function에 전달할 인자

        Returns:
            생성된 StructuredTool
        """
        tool = StructuredTool.from_function(func, **kwargs)
        self.register(tool)
        return tool

    def get(self, name: str) -> StructuredTool | None:
        """도구 가져오기"""
        return self.tools.get(name)

    def list_tools(self) -> list[str]:
        """도구 목록"""
        return list(self.tools.keys())

    def to_openai_tools(self) -> list[dict[str, Any]]:
        """OpenAI function calling 형식으로 export"""
        return [tool.to_openai_tool() for tool in self.tools.values()]

    def to_anthropic_tools(self) -> list[dict[str, Any]]:
        """Anthropic tool use 형식으로 export"""
        return [tool.to_anthropic_tool() for tool in self.tools.values()]
