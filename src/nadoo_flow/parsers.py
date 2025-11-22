"""
Output Parsers for Nadoo Flow
출력 파서 - 구조화된 데이터 추출 및 검증
"""

import json
import logging
import re
from typing import Any, Type, TypeVar, get_args, get_origin
from pydantic import BaseModel, ValidationError

from .base import BaseNode, NodeContext, NodeResult, WorkflowContext

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)


class OutputParser:
    """출력 파서 베이스 클래스"""

    def parse(self, text: str) -> Any:
        """텍스트를 파싱"""
        raise NotImplementedError

    def get_format_instructions(self) -> str:
        """LLM에게 제공할 포맷 지시사항"""
        return ""


class StructuredOutputParser(OutputParser):
    """Pydantic 모델 기반 구조화된 출력 파서

    LLM 출력을 Pydantic 모델로 파싱하고 검증합니다.

    Example:
        class AgentAction(BaseModel):
            action: Literal["search", "answer"]
            reasoning: str
            parameters: dict[str, Any]

        parser = StructuredOutputParser(pydantic_model=AgentAction)

        # LLM 프롬프트에 포함
        prompt = f\"\"\"
        {parser.get_format_instructions()}

        User query: {{query}}
        \"\"\"

        # 출력 파싱
        result = parser.parse(llm_output)
        print(result.action)  # 타입 안전
    """

    def __init__(self, pydantic_model: Type[T]):
        """
        Args:
            pydantic_model: 파싱할 Pydantic 모델 클래스
        """
        self.pydantic_model = pydantic_model

    def parse(self, text: str) -> T:
        """텍스트를 Pydantic 모델로 파싱

        Args:
            text: 파싱할 텍스트 (JSON 문자열 또는 마크다운 코드 블록)

        Returns:
            파싱된 Pydantic 모델 인스턴스

        Raises:
            ValueError: 파싱 실패 시
        """
        try:
            # 마크다운 코드 블록 제거
            json_text = self._extract_json_from_markdown(text)

            # JSON 파싱
            try:
                data = json.loads(json_text)
            except json.JSONDecodeError as e:
                # 부분 JSON 복구 시도
                data = self._try_fix_partial_json(json_text)
                if data is None:
                    raise ValueError(f"Invalid JSON: {e}") from e

            # Pydantic 검증
            return self.pydantic_model(**data)

        except ValidationError as e:
            # Pydantic 검증 실패 시 상세 에러 메시지
            errors = e.errors()
            error_details = "\n".join([
                f"  - {err['loc']}: {err['msg']}"
                for err in errors
            ])
            raise ValueError(
                f"Validation failed for {self.pydantic_model.__name__}:\n{error_details}\n\n"
                f"Original text:\n{text[:500]}"
            ) from e

        except Exception as e:
            raise ValueError(
                f"Failed to parse output as {self.pydantic_model.__name__}: {e}\n\n"
                f"Original text:\n{text[:500]}"
            ) from e

    def get_format_instructions(self) -> str:
        """Pydantic 모델에서 포맷 지시사항 생성"""
        # JSON 스키마 생성
        schema = self.pydantic_model.model_json_schema()

        # 예제가 있으면 사용
        example = None
        if hasattr(self.pydantic_model, 'Config'):
            example = getattr(self.pydantic_model.Config, 'json_schema_extra', {}).get('example')
        if not example and hasattr(self.pydantic_model, 'model_config'):
            example = self.pydantic_model.model_config.get('json_schema_extra', {}).get('example')

        instructions = [
            "Please respond with a JSON object in the following format:",
            "",
            "```json",
            json.dumps(schema, indent=2, ensure_ascii=False),
            "```"
        ]

        if example:
            instructions.extend([
                "",
                "Example:",
                "```json",
                json.dumps(example, indent=2, ensure_ascii=False),
                "```"
            ])

        return "\n".join(instructions)

    def _extract_json_from_markdown(self, text: str) -> str:
        """마크다운 코드 블록에서 JSON 추출"""
        # ```json ... ``` 패턴
        json_pattern = r'```json\s*(.*?)\s*```'
        match = re.search(json_pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()

        # ``` ... ``` 패턴 (언어 지정 없음)
        generic_pattern = r'```\s*(.*?)\s*```'
        match = re.search(generic_pattern, text, re.DOTALL)
        if match:
            content = match.group(1).strip()
            # JSON 같으면 반환
            if content.startswith('{') or content.startswith('['):
                return content

        # 코드 블록 없으면 전체 텍스트에서 JSON 찾기
        # { ... } 또는 [ ... ] 패턴
        json_obj_pattern = r'\{.*\}'
        match = re.search(json_obj_pattern, text, re.DOTALL)
        if match:
            return match.group(0)

        json_arr_pattern = r'\[.*\]'
        match = re.search(json_arr_pattern, text, re.DOTALL)
        if match:
            return match.group(0)

        # 그래도 없으면 원본 반환
        return text.strip()

    def _try_fix_partial_json(self, json_text: str) -> dict[str, Any] | None:
        """부분 JSON 복구 시도

        스트리밍 중 잘린 JSON을 복구합니다.
        """
        try:
            # 닫히지 않은 괄호 추가
            if json_text.count('{') > json_text.count('}'):
                json_text += '}' * (json_text.count('{') - json_text.count('}'))

            if json_text.count('[') > json_text.count(']'):
                json_text += ']' * (json_text.count('[') - json_text.count(']'))

            # 마지막 쉼표 제거
            json_text = re.sub(r',\s*([}\]])', r'\1', json_text)

            # 닫히지 않은 문자열 처리
            quote_count = json_text.count('"') - json_text.count('\\"')
            if quote_count % 2 == 1:
                json_text += '"'

            return json.loads(json_text)
        except:
            return None


class ParserNode(BaseNode):
    """출력 파서 노드

    이전 노드의 텍스트 출력을 파싱하여 구조화된 데이터로 변환합니다.

    Example:
        class UserInfo(BaseModel):
            name: str
            age: int

        parser_node = ParserNode(
            node_id="parse_user",
            parser=StructuredOutputParser(UserInfo),
            input_key="llm_output"
        )
    """

    def __init__(
        self,
        node_id: str,
        parser: OutputParser,
        input_key: str = "text",
        output_key: str = "parsed",
        name: str = "Parser",
        config: dict[str, Any] | None = None
    ):
        """
        Args:
            node_id: 노드 ID
            parser: 사용할 파서
            input_key: 입력에서 텍스트를 가져올 키
            output_key: 출력에 저장할 키
            name: 노드 이름
            config: 노드 설정
        """
        super().__init__(
            node_id=node_id,
            node_type="parser",
            name=name,
            config=config or {}
        )
        self.parser = parser
        self.input_key = input_key
        self.output_key = output_key

    async def execute(self, node_context: NodeContext, workflow_context: WorkflowContext) -> NodeResult:
        """파싱 실행"""
        try:
            # 입력 텍스트 가져오기
            text = node_context.get_input(self.input_key)

            if text is None:
                # previous_output에서 찾기
                prev_output = node_context.get_input("previous_output", {})
                text = prev_output.get(self.input_key)

            if text is None:
                return NodeResult(
                    success=False,
                    error=f"Input key '{self.input_key}' not found in node input"
                )

            # 파싱
            parsed = self.parser.parse(text)

            # Pydantic 모델이면 dict로 변환
            if isinstance(parsed, BaseModel):
                parsed_dict = parsed.model_dump()
            else:
                parsed_dict = parsed

            return NodeResult(
                success=True,
                output={
                    self.output_key: parsed_dict,
                    "original_text": text
                }
            )

        except Exception as e:
            logger.error(f"Parsing failed in node {self.node_id}: {e}")
            return NodeResult(
                success=False,
                error=str(e),
                metadata={"parsing_error": str(e)}
            )


class RetryableParserNode(ParserNode):
    """재시도 기능이 있는 파서 노드

    파싱 실패 시 LLM에게 수정을 요청합니다.

    Example:
        parser_node = RetryableParserNode(
            node_id="parse_with_retry",
            parser=StructuredOutputParser(MyModel),
            llm_node=my_llm_node,
            max_retries=3
        )
    """

    def __init__(
        self,
        node_id: str,
        parser: OutputParser,
        llm_node: BaseNode,
        max_retries: int = 3,
        input_key: str = "text",
        output_key: str = "parsed",
        name: str = "Retryable Parser",
        config: dict[str, Any] | None = None
    ):
        """
        Args:
            node_id: 노드 ID
            parser: 사용할 파서
            llm_node: 재시도에 사용할 LLM 노드
            max_retries: 최대 재시도 횟수
            input_key: 입력 키
            output_key: 출력 키
            name: 노드 이름
            config: 노드 설정
        """
        super().__init__(node_id, parser, input_key, output_key, name, config)
        self.llm_node = llm_node
        self.max_retries = max_retries

    async def execute(self, node_context: NodeContext, workflow_context: WorkflowContext) -> NodeResult:
        """재시도 로직이 포함된 파싱 실행"""
        # 첫 번째 시도
        result = await super().execute(node_context, workflow_context)

        if result.success:
            return result

        # 재시도 루프
        current_text = node_context.get_input(self.input_key)
        last_error = result.error

        for attempt in range(self.max_retries):
            logger.info(
                f"Parser {self.node_id} retrying ({attempt + 1}/{self.max_retries})"
            )

            # LLM에게 수정 요청
            retry_prompt = self._create_retry_prompt(current_text, last_error)

            # LLM 노드 실행
            llm_context = NodeContext(
                node_id=self.llm_node.node_id,
                node_type=self.llm_node.node_type,
                input_data={"prompt": retry_prompt}
            )

            llm_result = await self.llm_node.execute(llm_context, workflow_context)

            if not llm_result.success:
                logger.warning(f"LLM retry failed: {llm_result.error}")
                continue

            # 새로운 텍스트로 다시 파싱 시도
            current_text = llm_result.output.get("text", current_text)

            try:
                parsed = self.parser.parse(current_text)

                if isinstance(parsed, BaseModel):
                    parsed_dict = parsed.model_dump()
                else:
                    parsed_dict = parsed

                logger.info(
                    f"Parser {self.node_id} succeeded after {attempt + 1} retries"
                )

                return NodeResult(
                    success=True,
                    output={
                        self.output_key: parsed_dict,
                        "original_text": current_text
                    },
                    metadata={
                        "retry_count": attempt + 1,
                        "final_attempt": attempt + 1
                    }
                )

            except Exception as e:
                last_error = str(e)
                logger.warning(f"Retry {attempt + 1} failed: {e}")

        # 모든 재시도 실패
        return NodeResult(
            success=False,
            error=f"Parsing failed after {self.max_retries} retries. Last error: {last_error}",
            metadata={"total_retries": self.max_retries}
        )

    def _create_retry_prompt(self, original_text: str, error: str) -> str:
        """재시도를 위한 프롬프트 생성"""
        format_instructions = self.parser.get_format_instructions()

        return f"""The previous output was invalid and failed parsing:

Error: {error}

Please provide a corrected response in the exact format requested.

{format_instructions}

Original output:
{original_text}

Corrected output:"""


class JsonOutputParser(OutputParser):
    """간단한 JSON 파서

    Example:
        parser = JsonOutputParser()
        result = parser.parse('{"key": "value"}')
    """

    def parse(self, text: str) -> dict[str, Any] | list[Any]:
        """JSON 파싱"""
        # 마크다운 코드 블록 제거
        json_text = self._extract_json(text)

        try:
            return json.loads(json_text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}\n\nText: {text[:500]}") from e

    def get_format_instructions(self) -> str:
        """포맷 지시사항"""
        return "Please respond with valid JSON."

    def _extract_json(self, text: str) -> str:
        """JSON 추출 (StructuredOutputParser와 동일 로직)"""
        parser = StructuredOutputParser(pydantic_model=BaseModel)
        return parser._extract_json_from_markdown(text)


class StringOutputParser(OutputParser):
    """문자열 파서 - 그대로 반환

    Example:
        parser = StringOutputParser()
        result = parser.parse("some text")  # "some text"
    """

    def parse(self, text: str) -> str:
        """텍스트 그대로 반환"""
        return text.strip()

    def get_format_instructions(self) -> str:
        """포맷 지시사항"""
        return "Please respond with plain text."
