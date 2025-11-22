"""
Human-in-the-Loop Example for Nadoo Flow
사람 개입이 필요한 워크플로우 예시
"""

import asyncio
from nadoo_flow import (
    BaseNode,
    NodeContext,
    NodeResult,
    WorkflowContext,
    WorkflowExecutor,
    NodeStatus,
    StreamingContext,
    StreamEventType,
    SessionHistoryManager,
    Message,
    InMemoryChatHistory,
)


class ApprovalRequiredNode(BaseNode):
    """사람의 승인이 필요한 노드

    실행 결과를 제출하고 사용자 승인을 기다립니다.
    """

    def __init__(self, node_id: str, approval_message: str):
        super().__init__(node_id=node_id, node_type="approval")
        self.approval_message = approval_message

    async def execute(
        self,
        node_context: NodeContext,
        workflow_context: WorkflowContext
    ) -> NodeResult:
        """실행 후 승인 대기"""

        # 스트리밍으로 승인 요청 전송
        stream_ctx = getattr(workflow_context, "streaming_context", None)
        if stream_ctx:
            await stream_ctx.emit({
                "event_type": StreamEventType.CUSTOM,
                "name": self.node_id,
                "data": {
                    "type": "approval_required",
                    "message": self.approval_message,
                    "context": node_context.input_data
                }
            })

        # 워크플로우 중단 요청
        return NodeResult(
            success=True,
            output={
                "message": self.approval_message,
                "requires_approval": True
            },
            should_interrupt=True,  # 여기서 중단!
            metadata={
                "approval_status": "pending",
                "waiting_for": "user_approval"
            }
        )


class ConditionalNode(BaseNode):
    """승인 결과에 따라 다음 노드를 결정"""

    def __init__(self, node_id: str):
        super().__init__(node_id=node_id, node_type="conditional")

    async def execute(
        self,
        node_context: NodeContext,
        workflow_context: WorkflowContext
    ) -> NodeResult:
        """승인 결과 확인"""

        approval_data = node_context.input_data.get("approval", {})
        approved = approval_data.get("approved", False)
        feedback = approval_data.get("feedback", "")

        if approved:
            return NodeResult(
                success=True,
                output={
                    "action": "proceed",
                    "feedback": feedback
                }
            )
        else:
            return NodeResult(
                success=True,
                output={
                    "action": "revise",
                    "feedback": feedback,
                    "should_retry": True
                }
            )


class HITLWorkflow:
    """Human-in-the-Loop 워크플로우 관리자"""

    def __init__(self):
        self.executor = WorkflowExecutor()
        self.history_manager = SessionHistoryManager(
            history_factory=lambda sid: InMemoryChatHistory()
        )
        self.paused_workflows: dict[str, tuple[WorkflowContext, NodeContext]] = {}

    async def start_workflow(
        self,
        workflow_id: str,
        initial_data: dict
    ):
        """워크플로우 시작"""

        # 워크플로우 컨텍스트 생성
        workflow_context = WorkflowContext(
            workflow_id=workflow_id,
            session_id=workflow_id  # 세션 ID로 사용
        )

        # 스트리밍 컨텍스트 추가
        streaming_context = StreamingContext()
        workflow_context.streaming_context = streaming_context

        # 노드 정의
        draft_node = DraftGenerationNode("draft")
        approval_node = ApprovalRequiredNode(
            "approval",
            "Please review the draft and approve or provide feedback."
        )
        conditional_node = ConditionalNode("conditional")
        finalize_node = FinalizeNode("finalize")

        # 실행
        node_context = NodeContext(
            node_id="draft",
            node_type="draft",
            input_data=initial_data
        )

        try:
            async with streaming_context:
                # 스트리밍 이벤트 처리
                async def process_events():
                    async for event in streaming_context.stream():
                        print(f"[Event] {event.event_type}: {event.data}")

                        # 승인 요청 이벤트 감지
                        if (event.event_type == StreamEventType.CUSTOM and
                            event.data.get("type") == "approval_required"):

                            # 워크플로우 중단 - 컨텍스트 저장
                            self.paused_workflows[workflow_id] = (
                                workflow_context,
                                node_context
                            )
                            print(f"[HITL] Workflow {workflow_id} paused for approval")

                # Draft 생성
                result = await draft_node.execute(node_context, workflow_context)

                if result.should_interrupt:
                    # 승인 대기
                    self.paused_workflows[workflow_id] = (
                        workflow_context,
                        node_context
                    )
                    workflow_context.status = NodeStatus.INTERRUPTED
                    workflow_context.waiting_for = "user_approval"

                    print(f"[HITL] Workflow paused. Waiting for approval...")
                    return {
                        "status": "paused",
                        "message": result.output.get("message"),
                        "workflow_id": workflow_id
                    }

        except Exception as e:
            print(f"[Error] {e}")
            return {"status": "error", "error": str(e)}

    async def resume_workflow(
        self,
        workflow_id: str,
        approval_data: dict
    ):
        """워크플로우 재개 (승인 후)"""

        if workflow_id not in self.paused_workflows:
            return {"status": "error", "error": "Workflow not found"}

        workflow_context, node_context = self.paused_workflows[workflow_id]

        # 승인 데이터 추가
        node_context.input_data["approval"] = approval_data

        # 세션 히스토리에 승인 기록
        history = await self.history_manager.get_history(workflow_id)
        await history.add_message(Message.system(
            f"User approval: {approval_data.get('approved', False)}, "
            f"Feedback: {approval_data.get('feedback', 'None')}"
        ))

        # 상태 복원
        workflow_context.status = NodeStatus.IN_PROGRESS
        workflow_context.waiting_for = None

        print(f"[HITL] Resuming workflow {workflow_id}...")

        # Conditional 노드 실행
        conditional_node = ConditionalNode("conditional")
        result = await conditional_node.execute(node_context, workflow_context)

        if result.output.get("action") == "proceed":
            # 최종 완료
            finalize_node = FinalizeNode("finalize")
            final_result = await finalize_node.execute(node_context, workflow_context)

            workflow_context.status = NodeStatus.SUCCESS
            del self.paused_workflows[workflow_id]

            return {
                "status": "completed",
                "result": final_result.output,
                "workflow_id": workflow_id
            }
        else:
            # 재작성 필요
            return {
                "status": "needs_revision",
                "feedback": result.output.get("feedback"),
                "workflow_id": workflow_id
            }


# 예시 노드 구현
class DraftGenerationNode(BaseNode):
    """Draft 생성 노드"""

    async def execute(self, node_context, workflow_context) -> NodeResult:
        content = node_context.input_data.get("content", "")

        # Draft 생성 (LLM 호출 등)
        draft = f"Draft version: {content}"

        return NodeResult(
            success=True,
            output={"draft": draft},
            should_interrupt=True  # 승인 대기
        )


class FinalizeNode(BaseNode):
    """최종 완료 노드"""

    async def execute(self, node_context, workflow_context) -> NodeResult:
        draft = node_context.input_data.get("draft", "")

        return NodeResult(
            success=True,
            output={"final_content": draft, "status": "approved"}
        )


# 사용 예시
async def main():
    """HITL 워크플로우 실행 예시"""

    workflow = HITLWorkflow()

    # 1. 워크플로우 시작
    print("=== Starting HITL Workflow ===")
    result1 = await workflow.start_workflow(
        workflow_id="workflow_123",
        initial_data={"content": "Important document content"}
    )
    print(f"Result 1: {result1}")

    # 2. 사용자 승인 시뮬레이션 (실제로는 API 호출 등으로 받음)
    print("\n=== User Approval (simulated) ===")
    await asyncio.sleep(2)  # 사용자 검토 시간

    # 3. 워크플로우 재개 (승인)
    print("\n=== Resuming with Approval ===")
    result2 = await workflow.resume_workflow(
        workflow_id="workflow_123",
        approval_data={
            "approved": True,
            "feedback": "Looks good!"
        }
    )
    print(f"Result 2: {result2}")

    # 4. 거부 케이스
    print("\n\n=== Testing Rejection Case ===")
    result3 = await workflow.start_workflow(
        workflow_id="workflow_456",
        initial_data={"content": "Another document"}
    )
    print(f"Result 3: {result3}")

    print("\n=== User Rejection (simulated) ===")
    await asyncio.sleep(1)

    result4 = await workflow.resume_workflow(
        workflow_id="workflow_456",
        approval_data={
            "approved": False,
            "feedback": "Please add more details in section 2"
        }
    )
    print(f"Result 4: {result4}")


if __name__ == "__main__":
    asyncio.run(main())
