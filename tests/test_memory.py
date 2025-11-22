"""
Tests for Memory Module
메모리 모듈 테스트
"""

import pytest
import asyncio
from nadoo_flow.memory import (
    InMemoryChatHistory,
    SlidingWindowChatHistory,
    SessionHistoryManager,
    ChatHistoryNode,
    create_inmemory_history_manager,
)
from nadoo_flow.prompts import Message
from nadoo_flow import BaseNode, NodeContext, NodeResult, WorkflowContext


@pytest.mark.asyncio
async def test_inmemory_chat_history():
    """인메모리 채팅 히스토리 테스트"""

    history = InMemoryChatHistory()

    # 초기 상태
    messages = await history.get_messages()
    assert len(messages) == 0

    # 메시지 추가
    await history.add_message(Message.user("Hello"))
    messages = await history.get_messages()
    assert len(messages) == 1
    assert messages[0].role == "user"
    assert messages[0].content == "Hello"

    # 여러 메시지 추가
    await history.add_messages([
        Message.assistant("Hi!"),
        Message.user("How are you?")
    ])
    messages = await history.get_messages()
    assert len(messages) == 3

    # 초기화
    await history.clear()
    messages = await history.get_messages()
    assert len(messages) == 0


@pytest.mark.asyncio
async def test_sliding_window_chat_history():
    """슬라이딩 윈도우 히스토리 테스트"""

    base_history = InMemoryChatHistory()
    window_history = SlidingWindowChatHistory(
        base_history=base_history,
        window_size=3
    )

    # 5개 메시지 추가
    messages_to_add = [
        Message.user(f"Message {i}")
        for i in range(5)
    ]
    await window_history.add_messages(messages_to_add)

    # 최근 3개만 반환되어야 함
    messages = await window_history.get_messages()
    assert len(messages) == 3
    assert messages[0].content == "Message 2"
    assert messages[2].content == "Message 4"

    # 기본 히스토리에는 모두 저장됨
    all_messages = await base_history.get_messages()
    assert len(all_messages) == 5


@pytest.mark.asyncio
async def test_session_history_manager():
    """세션 히스토리 매니저 테스트"""

    manager = create_inmemory_history_manager(window_size=10)

    # 세션 1
    history1 = await manager.get_history("session_1")
    await history1.add_message(Message.user("Session 1 message"))

    # 세션 2
    history2 = await manager.get_history("session_2")
    await history2.add_message(Message.user("Session 2 message"))

    # 세션이 분리되어 있어야 함
    messages1 = await history1.get_messages()
    messages2 = await history2.get_messages()

    assert len(messages1) == 1
    assert len(messages2) == 1
    assert messages1[0].content != messages2[0].content

    # 같은 세션 ID로 다시 가져오기
    history1_again = await manager.get_history("session_1")
    messages1_again = await history1_again.get_messages()
    assert len(messages1_again) == 1
    assert messages1_again[0].content == "Session 1 message"

    # 세션 초기화
    await manager.clear_history("session_1")
    history1_cleared = await manager.get_history("session_1")
    messages_cleared = await history1_cleared.get_messages()
    assert len(messages_cleared) == 0


@pytest.mark.asyncio
async def test_chat_history_node():
    """채팅 히스토리 노드 믹스인 테스트"""

    class TestChatNode(BaseNode, ChatHistoryNode):
        def __init__(self, node_id: str, history_manager: SessionHistoryManager):
            BaseNode.__init__(
                self,
                node_id=node_id,
                node_type="test",
                name="test_chat_node",
                config={}
            )
            ChatHistoryNode.__init__(self, history_manager=history_manager)

        async def execute(
            self,
            node_context: NodeContext,
            workflow_context: WorkflowContext
        ) -> NodeResult:
            # 히스토리 가져오기
            history = await self.get_history(workflow_context)

            # 메시지 추가
            user_msg = node_context.input_data.get("message", "")
            await history.add_message(Message.user(user_msg))

            # 응답 생성
            response = f"Echo: {user_msg}"
            await history.add_message(Message.assistant(response))

            return NodeResult(
                success=True,
                output={"response": response}
            )

    manager = create_inmemory_history_manager()
    node = TestChatNode("test_node", manager)

    workflow_context = WorkflowContext(
        workflow_id="test_workflow",
        session_id="test_session"
    )

    # 첫 번째 실행
    node_context1 = NodeContext(
        node_id="test_node",
        node_type="test",
        input_data={"message": "Hello"}
    )
    result1 = await node.execute(node_context1, workflow_context)
    assert result1.success

    # 히스토리 확인
    history = await manager.get_history("test_session")
    messages = await history.get_messages()
    assert len(messages) == 2  # user + assistant
    assert messages[0].role == "user"
    assert messages[1].role == "assistant"


@pytest.mark.asyncio
async def test_save_interaction():
    """대화 저장 테스트"""

    class TestNode(ChatHistoryNode):
        def __init__(self, history_manager: SessionHistoryManager):
            ChatHistoryNode.__init__(self, history_manager=history_manager)

    manager = create_inmemory_history_manager()
    node = TestNode(manager)

    workflow_context = WorkflowContext(
        workflow_id="test",
        session_id="test_session"
    )

    # 대화 저장
    await node.save_interaction(
        workflow_context,
        user_message="What is AI?",
        assistant_message="AI is artificial intelligence..."
    )

    # 확인
    history = await manager.get_history("test_session")
    messages = await history.get_messages()

    assert len(messages) == 2
    assert messages[0].content == "What is AI?"
    assert messages[1].content == "AI is artificial intelligence..."


@pytest.mark.asyncio
async def test_default_session_id():
    """기본 세션 ID 테스트"""

    class TestNode(BaseNode, ChatHistoryNode):
        def __init__(self, history_manager: SessionHistoryManager):
            BaseNode.__init__(
                self,
                node_id="test",
                node_type="test",
                name="test_node",
                config={}
            )
            ChatHistoryNode.__init__(self, history_manager=history_manager)

        async def execute(
            self,
            node_context: NodeContext,
            workflow_context: WorkflowContext
        ) -> NodeResult:
            history = await self.get_history(workflow_context)
            await history.add_message(Message.user("test"))

            return NodeResult(success=True, output={})

    manager = create_inmemory_history_manager()
    node = TestNode(manager)

    # session_id 없는 workflow_context
    workflow_context = WorkflowContext(workflow_id="workflow_123")

    node_context = NodeContext(
        node_id="test",
        node_type="test",
        input_data={}
    )

    # 워크플로우 ID가 세션 ID로 사용되어야 함
    await node.execute(node_context, workflow_context)

    history = await manager.get_history("workflow_123")
    messages = await history.get_messages()
    assert len(messages) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
