"""
Example 1: Basic Chatbot with Memory
예제 1: 메모리를 가진 기본 챗봇
"""

import asyncio
from nadoo_flow import (
    BaseNode,
    NodeContext,
    NodeResult,
    WorkflowContext,
    InMemoryChatHistory,
    SessionHistoryManager,
    ChatHistoryNode,
    Message,
)


class SimpleChatNode(BaseNode, ChatHistoryNode):
    """간단한 챗봇 노드"""

    def __init__(self, node_id: str, history_manager: SessionHistoryManager):
        BaseNode.__init__(self, node_id=node_id, node_type="chat")
        ChatHistoryNode.__init__(self, history_manager=history_manager)

    async def execute(
        self,
        node_context: NodeContext,
        workflow_context: WorkflowContext
    ) -> NodeResult:
        """대화 처리"""

        # 1. 히스토리 로드
        history = await self.get_history(workflow_context)
        messages = await history.get_messages()

        # 2. 사용자 메시지 저장
        user_message = node_context.input_data.get("message", "")
        await history.add_message(Message.user(user_message))

        # 3. 이전 대화 기반 응답 생성 (실제로는 LLM 호출)
        response = self._generate_response(user_message, messages)

        # 4. 응답 저장
        await history.add_message(Message.assistant(response))

        return NodeResult(
            success=True,
            output={"response": response}
        )

    def _generate_response(self, user_message: str, history: list[Message]) -> str:
        """응답 생성 (Mock)"""

        # 간단한 규칙 기반 응답
        lower_msg = user_message.lower()

        if "안녕" in lower_msg or "hello" in lower_msg:
            return "안녕하세요! 무엇을 도와드릴까요?"

        elif "이름" in lower_msg or "name" in lower_msg:
            return "저는 Nadoo 챗봇입니다."

        elif "이전" in lower_msg or "전에" in lower_msg:
            if len(history) > 2:
                prev_user_msg = history[-3].content  # 2턴 전 사용자 메시지
                return f"이전에 '{prev_user_msg}'에 대해 이야기하셨네요."
            else:
                return "아직 이전 대화가 없습니다."

        else:
            return f"'{user_message}'에 대해 알려드리겠습니다. (여기에 LLM 응답이 들어갑니다)"


async def main():
    """챗봇 실행"""

    # 히스토리 매니저 생성 (최근 10개 메시지 유지)
    history_manager = SessionHistoryManager(
        history_factory=lambda sid: InMemoryChatHistory(),
        window_size=10
    )

    # 챗봇 노드 생성
    chatbot = SimpleChatNode("chatbot", history_manager)

    # 워크플로우 컨텍스트 (세션 ID = 사용자 ID)
    workflow_context = WorkflowContext(
        workflow_id="chat_workflow",
        session_id="user_123"
    )

    # 대화 시뮬레이션
    conversations = [
        "안녕하세요",
        "너의 이름은 뭐야?",
        "날씨가 어때?",
        "이전에 뭐라고 했지?"
    ]

    print("=== Chatbot with Memory ===\n")

    for user_input in conversations:
        print(f"User: {user_input}")

        node_context = NodeContext(
            node_id="chatbot",
            node_type="chat",
            input_data={"message": user_input}
        )

        result = await chatbot.execute(node_context, workflow_context)

        if result.success:
            print(f"Bot: {result.output['response']}\n")
        else:
            print(f"Error: {result.error}\n")

    # 히스토리 확인
    print("=== Chat History ===")
    history = await history_manager.get_history("user_123")
    messages = await history.get_messages()

    print(f"Total messages: {len(messages)}")
    for i, msg in enumerate(messages, 1):
        content_preview = msg.content[:50] + "..." if len(msg.content) > 50 else msg.content
        print(f"{i}. {msg.role}: {content_preview}")


if __name__ == "__main__":
    asyncio.run(main())
