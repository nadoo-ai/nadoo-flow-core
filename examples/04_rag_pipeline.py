"""
Example 4: RAG (Retrieval-Augmented Generation) Pipeline
예제 4: RAG 파이프라인 (검색 증강 생성)
"""

import asyncio
from typing import Any
from nadoo_flow import (
    BaseNode,
    NodeContext,
    NodeResult,
    WorkflowContext,
    NodeChain,
    ParallelNode,
    ParallelStrategy,
    CachedNode,
    ResponseCache,
    InMemoryCache,
)


class EmbeddingNode(BaseNode):
    """임베딩 생성 노드"""

    def __init__(self, node_id: str):
        super().__init__(node_id=node_id, node_type="embedding")

    async def execute(
        self,
        node_context: NodeContext,
        workflow_context: WorkflowContext
    ) -> NodeResult:
        """텍스트를 임베딩 벡터로 변환"""

        text = node_context.input_data.get("text", "")

        # 임베딩 생성 시뮬레이션
        await asyncio.sleep(0.2)

        # Mock 임베딩 (실제로는 OpenAI embeddings API 호출)
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]  # 단순화된 벡터

        return NodeResult(
            success=True,
            output={
                "text": text,
                "embedding": embedding,
                "dimensions": len(embedding)
            }
        )


class VectorSearchNode(BaseNode):
    """벡터 검색 노드"""

    def __init__(self, node_id: str, top_k: int = 5):
        super().__init__(node_id=node_id, node_type="vector_search")
        self.top_k = top_k

        # Mock 지식 베이스
        self.knowledge_base = [
            {
                "id": "doc_1",
                "text": "Artificial Intelligence is the simulation of human intelligence...",
                "metadata": {"source": "AI Textbook", "page": 1}
            },
            {
                "id": "doc_2",
                "text": "Machine Learning is a subset of AI that focuses on learning from data...",
                "metadata": {"source": "ML Guide", "page": 5}
            },
            {
                "id": "doc_3",
                "text": "Deep Learning uses neural networks with multiple layers...",
                "metadata": {"source": "DL Book", "page": 20}
            },
            {
                "id": "doc_4",
                "text": "Natural Language Processing enables computers to understand human language...",
                "metadata": {"source": "NLP Paper", "page": 1}
            },
            {
                "id": "doc_5",
                "text": "Computer Vision allows machines to interpret visual information...",
                "metadata": {"source": "CV Research", "page": 10}
            },
        ]

    async def execute(
        self,
        node_context: NodeContext,
        workflow_context: WorkflowContext
    ) -> NodeResult:
        """벡터 유사도 검색"""

        query_embedding = node_context.input_data.get("embedding", [])

        # 벡터 검색 시뮬레이션
        await asyncio.sleep(0.1)

        # 실제로는 벡터 DB (Pinecone, Weaviate 등)에서 검색
        # 여기서는 랜덤하게 상위 k개 반환
        retrieved_docs = self.knowledge_base[:self.top_k]

        return NodeResult(
            success=True,
            output={
                "documents": retrieved_docs,
                "count": len(retrieved_docs)
            }
        )


class RerankingNode(BaseNode):
    """재순위화 노드"""

    def __init__(self, node_id: str):
        super().__init__(node_id=node_id, node_type="reranking")

    async def execute(
        self,
        node_context: NodeContext,
        workflow_context: WorkflowContext
    ) -> NodeResult:
        """검색된 문서를 재순위화"""

        documents = node_context.input_data.get("documents", [])
        query = node_context.input_data.get("query", "")

        # 재순위화 시뮬레이션 (실제로는 Cross-encoder 사용)
        await asyncio.sleep(0.15)

        # 간단한 키워드 매칭으로 재순위화
        reranked = sorted(
            documents,
            key=lambda doc: self._relevance_score(doc["text"], query),
            reverse=True
        )

        return NodeResult(
            success=True,
            output={
                "documents": reranked,
                "count": len(reranked)
            }
        )

    def _relevance_score(self, text: str, query: str) -> float:
        """간단한 관련성 점수"""

        # 쿼리 키워드가 텍스트에 포함되면 높은 점수
        query_words = query.lower().split()
        text_lower = text.lower()

        score = sum(1 for word in query_words if word in text_lower)
        return score


class PromptAugmentationNode(BaseNode):
    """프롬프트 증강 노드"""

    def __init__(self, node_id: str):
        super().__init__(node_id=node_id, node_type="prompt_augmentation")

    async def execute(
        self,
        node_context: NodeContext,
        workflow_context: WorkflowContext
    ) -> NodeResult:
        """검색된 컨텍스트를 프롬프트에 추가"""

        query = node_context.input_data.get("query", "")
        documents = node_context.input_data.get("documents", [])

        # 컨텍스트 구성
        context_parts = []
        for i, doc in enumerate(documents, 1):
            context_parts.append(
                f"[Document {i}] {doc['text']}\n"
                f"Source: {doc['metadata'].get('source', 'Unknown')}"
            )

        context = "\n\n".join(context_parts)

        # 최종 프롬프트
        augmented_prompt = f"""Given the following context, answer the question.

Context:
{context}

Question: {query}

Answer:"""

        return NodeResult(
            success=True,
            output={
                "prompt": augmented_prompt,
                "context_length": len(context),
                "num_documents": len(documents)
            }
        )


class LLMGenerationNode(BaseNode, CachedNode):
    """LLM 생성 노드 (캐싱 포함)"""

    def __init__(self, node_id: str, model: str = "gpt-4"):
        BaseNode.__init__(self, node_id=node_id, node_type="llm")

        # 캐싱 설정
        cache = InMemoryCache(max_size=100, ttl=3600)
        response_cache = ResponseCache(cache)

        CachedNode.__init__(
            self,
            cache=response_cache,
            cache_key_fn=lambda ctx: ctx.input_data.get("prompt", "")
        )

        self.model = model

    async def _execute_with_cache(
        self,
        node_context: NodeContext,
        workflow_context: WorkflowContext
    ) -> NodeResult:
        """캐시를 거치는 LLM 실행"""

        prompt = node_context.input_data.get("prompt", "")

        # LLM 호출 시뮬레이션
        await asyncio.sleep(0.5)

        # Mock 응답
        response = (
            "Based on the provided context, here is the answer: "
            "Artificial Intelligence is the field of computer science that aims to "
            "create systems capable of performing tasks that typically require human intelligence."
        )

        return NodeResult(
            success=True,
            output={
                "response": response,
                "model": self.model
            }
        )


async def demo_basic_rag():
    """기본 RAG 파이프라인"""

    print("=== Basic RAG Pipeline ===\n")

    # RAG 체인 구성
    embedding_node = EmbeddingNode("embedding")
    search_node = VectorSearchNode("vector_search", top_k=3)
    rerank_node = RerankingNode("reranking")
    augment_node = PromptAugmentationNode("prompt_augmentation")
    llm_node = LLMGenerationNode("llm_generation", model="gpt-4")

    # 체인 연결은 수동으로 (실제 프로덕션에서는 WorkflowExecutor 사용)
    workflow_context = WorkflowContext(workflow_id="rag_workflow")

    query = "What is Artificial Intelligence?"
    print(f"Query: {query}\n")

    # 1. 임베딩 생성
    print("1. Generating embedding...")
    embedding_result = await embedding_node.execute(
        NodeContext(
            node_id="embedding",
            node_type="embedding",
            input_data={"text": query}
        ),
        workflow_context
    )

    # 2. 벡터 검색
    print("2. Searching vector database...")
    search_result = await search_node.execute(
        NodeContext(
            node_id="vector_search",
            node_type="vector_search",
            input_data={"embedding": embedding_result.output["embedding"]}
        ),
        workflow_context
    )

    print(f"   Found {search_result.output['count']} documents")

    # 3. 재순위화
    print("3. Reranking results...")
    rerank_result = await rerank_node.execute(
        NodeContext(
            node_id="reranking",
            node_type="reranking",
            input_data={
                "documents": search_result.output["documents"],
                "query": query
            }
        ),
        workflow_context
    )

    # 4. 프롬프트 증강
    print("4. Augmenting prompt with context...")
    augment_result = await augment_node.execute(
        NodeContext(
            node_id="prompt_augmentation",
            node_type="prompt_augmentation",
            input_data={
                "query": query,
                "documents": rerank_result.output["documents"]
            }
        ),
        workflow_context
    )

    print(f"   Context length: {augment_result.output['context_length']} chars")

    # 5. LLM 생성
    print("5. Generating answer...")
    llm_result = await llm_node.execute(
        NodeContext(
            node_id="llm_generation",
            node_type="llm",
            input_data={"prompt": augment_result.output["prompt"]}
        ),
        workflow_context
    )

    print(f"\nAnswer: {llm_result.output['response']}\n")


async def demo_multi_source_rag():
    """다중 소스 병렬 검색 RAG"""

    print("\n=== Multi-Source Parallel RAG ===\n")

    class WikiSearchNode(BaseNode):
        async def execute(self, node_context, workflow_context):
            await asyncio.sleep(0.2)
            return NodeResult(
                success=True,
                output={
                    "source": "Wikipedia",
                    "documents": [
                        {"text": "AI from Wikipedia...", "url": "wiki.com/ai"}
                    ]
                }
            )

    class ArxivSearchNode(BaseNode):
        async def execute(self, node_context, workflow_context):
            await asyncio.sleep(0.3)
            return NodeResult(
                success=True,
                output={
                    "source": "ArXiv",
                    "documents": [
                        {"text": "AI research paper...", "url": "arxiv.org/123"}
                    ]
                }
            )

    class CompanyDocsSearchNode(BaseNode):
        async def execute(self, node_context, workflow_context):
            await asyncio.sleep(0.1)
            return NodeResult(
                success=True,
                output={
                    "source": "Company Docs",
                    "documents": [
                        {"text": "Internal AI guide...", "url": "docs.company.com/ai"}
                    ]
                }
            )

    # 병렬 검색
    parallel_search = ParallelNode(
        node_id="multi_source_search",
        nodes=[
            WikiSearchNode("wiki"),
            ArxivSearchNode("arxiv"),
            CompanyDocsSearchNode("company_docs"),
        ],
        strategy=ParallelStrategy.ALL_SETTLED,  # 일부 실패해도 OK
        aggregate_outputs=True
    )

    workflow_context = WorkflowContext(workflow_id="multi_rag")
    node_context = NodeContext(
        node_id="multi_source_search",
        node_type="parallel",
        input_data={"query": "What is AI?"}
    )

    print("Searching multiple sources in parallel...")
    result = await parallel_search.execute(node_context, workflow_context)

    print(f"Retrieved from {len(result.output)} sources:")
    for source_id, data in result.output.items():
        source_name = data.get("source", source_id)
        doc_count = len(data.get("documents", []))
        print(f"  - {source_name}: {doc_count} documents")


async def main():
    """모든 데모 실행"""

    await demo_basic_rag()
    await demo_multi_source_rag()


if __name__ == "__main__":
    asyncio.run(main())
