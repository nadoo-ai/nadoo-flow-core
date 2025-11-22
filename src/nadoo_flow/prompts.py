"""
Prompt Templates for Nadoo Flow
프롬프트 템플릿 - 재사용 가능한 프롬프트 관리
"""

import json
import logging
from dataclasses import dataclass
from typing import Any, Literal

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """채팅 메시지

    LLM 대화에 사용되는 메시지 구조입니다.
    """

    role: Literal["system", "user", "assistant", "function", "tool"]
    """메시지 역할"""

    content: str | list[dict[str, Any]]
    """메시지 내용 (텍스트 또는 멀티모달 컨텐츠)"""

    name: str | None = None
    """함수/도구 이름 (function/tool 역할일 때)"""

    metadata: dict[str, Any] | None = None
    """추가 메타데이터"""

    def to_dict(self) -> dict[str, Any]:
        """딕셔너리로 변환"""
        result = {
            "role": self.role,
            "content": self.content
        }

        if self.name:
            result["name"] = self.name

        if self.metadata:
            result["metadata"] = self.metadata

        return result

    @classmethod
    def system(cls, content: str) -> "Message":
        """시스템 메시지 생성"""
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str) -> "Message":
        """사용자 메시지 생성"""
        return cls(role="user", content=content)

    @classmethod
    def assistant(cls, content: str) -> "Message":
        """어시스턴트 메시지 생성"""
        return cls(role="assistant", content=content)


class PromptTemplate:
    """프롬프트 템플릿

    변수를 포함한 재사용 가능한 프롬프트 템플릿입니다.

    Example:
        template = PromptTemplate(
            "Translate the following text to {language}:\\n\\n{text}"
        )

        prompt = template.format(language="Korean", text="Hello, world!")
        # "Translate the following text to Korean:\\n\\nHello, world!"
    """

    def __init__(self, template: str, input_variables: list[str] | None = None):
        """
        Args:
            template: 템플릿 문자열 ({variable} 형식)
            input_variables: 입력 변수 목록 (None이면 자동 추출)
        """
        self.template = template

        if input_variables is None:
            # 템플릿에서 변수 자동 추출
            import re
            self.input_variables = re.findall(r'\{(\w+)\}', template)
        else:
            self.input_variables = input_variables

    def format(self, **kwargs) -> str:
        """템플릿 포맷팅

        Args:
            **kwargs: 변수 값들

        Returns:
            포맷팅된 프롬프트
        """
        # 누락된 변수 체크
        missing = set(self.input_variables) - set(kwargs.keys())
        if missing:
            raise ValueError(f"Missing template variables: {missing}")

        return self.template.format(**kwargs)

    def format_partial(self, **kwargs) -> "PromptTemplate":
        """일부 변수만 포맷팅

        Args:
            **kwargs: 일부 변수 값들

        Returns:
            새로운 PromptTemplate (나머지 변수 포함)
        """
        # 제공된 변수로 포맷팅
        template = self.template
        for key, value in kwargs.items():
            template = template.replace(f"{{{key}}}", str(value))

        # 남은 변수 목록
        remaining_vars = [v for v in self.input_variables if v not in kwargs]

        return PromptTemplate(template, input_variables=remaining_vars)


class ChatPromptTemplate:
    """채팅 프롬프트 템플릿

    여러 메시지로 구성된 채팅 프롬프트 템플릿입니다.

    Example:
        template = ChatPromptTemplate.from_messages([
            ("system", "You are a {role}"),
            ("user", "{question}")
        ])

        messages = template.format(
            role="helpful assistant",
            question="What is AI?"
        )
    """

    def __init__(self, messages: list[tuple[str, str] | Message]):
        """
        Args:
            messages: 메시지 리스트 (role, content) 튜플 또는 Message 객체
        """
        self.messages: list[Message] = []

        for msg in messages:
            if isinstance(msg, Message):
                self.messages.append(msg)
            else:
                role, content = msg
                self.messages.append(Message(role=role, content=content))

        # 모든 변수 수집
        self.input_variables = set()
        for msg in self.messages:
            if isinstance(msg.content, str):
                import re
                vars_in_content = re.findall(r'\{(\w+)\}', msg.content)
                self.input_variables.update(vars_in_content)

        self.input_variables = list(self.input_variables)

    @classmethod
    def from_messages(cls, messages: list[tuple[str, str]]) -> "ChatPromptTemplate":
        """메시지 리스트에서 생성

        Args:
            messages: (role, content) 튜플 리스트

        Returns:
            ChatPromptTemplate
        """
        return cls(messages)

    def format(self, **kwargs) -> list[Message]:
        """템플릿 포맷팅

        Args:
            **kwargs: 변수 값들

        Returns:
            포맷팅된 메시지 리스트
        """
        formatted_messages = []

        for msg in self.messages:
            if isinstance(msg.content, str):
                content = msg.content.format(**kwargs)
            else:
                content = msg.content

            formatted_messages.append(
                Message(
                    role=msg.role,
                    content=content,
                    name=msg.name,
                    metadata=msg.metadata
                )
            )

        return formatted_messages

    def format_to_string(self, **kwargs) -> str:
        """템플릿을 단일 문자열로 포맷팅

        Args:
            **kwargs: 변수 값들

        Returns:
            포맷팅된 프롬프트 문자열
        """
        messages = self.format(**kwargs)
        result = []

        for msg in messages:
            if msg.role == "system":
                result.append(f"System: {msg.content}")
            elif msg.role == "user":
                result.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                result.append(f"Assistant: {msg.content}")
            else:
                result.append(f"{msg.role.capitalize()}: {msg.content}")

        return "\n\n".join(result)


class MessagesPlaceholder:
    """메시지 플레이스홀더

    런타임에 동적으로 메시지를 주입합니다.

    Example:
        template = ChatPromptTemplate([
            ("system", "You are a helpful assistant"),
            MessagesPlaceholder("chat_history"),
            ("user", "{question}")
        ])

        messages = template.format(
            chat_history=[
                Message.user("Hi"),
                Message.assistant("Hello!")
            ],
            question="How are you?"
        )
    """

    def __init__(
        self,
        variable_name: str,
        optional: bool = False,
        n_messages: int | None = None
    ):
        """
        Args:
            variable_name: 변수 이름
            optional: 선택 사항 여부 (값이 없어도 에러 없음)
            n_messages: 최대 메시지 수 (슬라이딩 윈도우)
        """
        self.variable_name = variable_name
        self.optional = optional
        self.n_messages = n_messages


class FewShotPromptTemplate:
    """Few-shot 프롬프트 템플릿

    예제를 포함한 few-shot learning 프롬프트입니다.

    Example:
        template = FewShotPromptTemplate(
            examples=[
                {"input": "2+2", "output": "4"},
                {"input": "3*5", "output": "15"}
            ],
            example_template="Q: {input}\\nA: {output}",
            suffix="Q: {input}\\nA:"
        )

        prompt = template.format(input="10/2")
    """

    def __init__(
        self,
        examples: list[dict[str, str]],
        example_template: str,
        prefix: str = "",
        suffix: str = "",
        example_separator: str = "\n\n"
    ):
        """
        Args:
            examples: 예제 리스트
            example_template: 예제 포맷 템플릿
            prefix: 접두사
            suffix: 접미사 (실제 질문 포함)
            example_separator: 예제 구분자
        """
        self.examples = examples
        self.example_template = PromptTemplate(example_template)
        self.prefix = prefix
        self.suffix = suffix
        self.example_separator = example_separator

    def format(self, **kwargs) -> str:
        """템플릿 포맷팅"""
        parts = []

        # 접두사
        if self.prefix:
            parts.append(self.prefix)

        # 예제들
        example_strings = []
        for example in self.examples:
            example_str = self.example_template.format(**example)
            example_strings.append(example_str)

        if example_strings:
            parts.append(self.example_separator.join(example_strings))

        # 접미사 (실제 질문)
        if self.suffix:
            suffix_template = PromptTemplate(self.suffix)
            parts.append(suffix_template.format(**kwargs))

        return "\n\n".join(parts)


class PromptLibrary:
    """프롬프트 라이브러리

    재사용 가능한 프롬프트 템플릿을 관리합니다.

    Example:
        library = PromptLibrary()

        library.add("translate", PromptTemplate(
            "Translate to {language}:\\n\\n{text}"
        ))

        template = library.get("translate")
        prompt = template.format(language="Korean", text="Hello")
    """

    def __init__(self):
        self.templates: dict[str, PromptTemplate | ChatPromptTemplate] = {}

    def add(
        self,
        name: str,
        template: PromptTemplate | ChatPromptTemplate
    ):
        """템플릿 추가"""
        self.templates[name] = template

    def get(self, name: str) -> PromptTemplate | ChatPromptTemplate:
        """템플릿 가져오기"""
        if name not in self.templates:
            raise KeyError(f"Template '{name}' not found")

        return self.templates[name]

    def list_templates(self) -> list[str]:
        """템플릿 목록"""
        return list(self.templates.keys())

    def remove(self, name: str):
        """템플릿 제거"""
        if name in self.templates:
            del self.templates[name]


# 자주 사용되는 프롬프트 템플릿들
DEFAULT_PROMPTS = {
    "summarize": PromptTemplate(
        "Please summarize the following text:\n\n{text}\n\nSummary:"
    ),

    "translate": PromptTemplate(
        "Translate the following text to {language}:\n\n{text}\n\nTranslation:"
    ),

    "explain": PromptTemplate(
        "Explain the following concept in simple terms:\n\n{concept}\n\nExplanation:"
    ),

    "classify": PromptTemplate(
        "Classify the following text into one of these categories: {categories}\n\n"
        "Text: {text}\n\nCategory:"
    ),

    "extract_entities": PromptTemplate(
        "Extract named entities from the following text:\n\n{text}\n\n"
        "Entities (as JSON):"
    ),

    "qa": ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer the user's questions accurately."),
        ("user", "{question}")
    ]),

    "cot": ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Think step by step before answering."),
        ("user", "{question}\n\nLet's think step by step:")
    ]),
}


def get_default_library() -> PromptLibrary:
    """기본 프롬프트 라이브러리 생성"""
    library = PromptLibrary()

    for name, template in DEFAULT_PROMPTS.items():
        library.add(name, template)

    return library
