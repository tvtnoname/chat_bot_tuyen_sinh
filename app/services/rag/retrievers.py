from typing import List
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.vectorstores import VectorStore
from flashrank import Ranker, RerankRequest

class HybridRetriever(BaseRetriever):
    """
    Hybrid Retriever kết hợp BM25 (Keyword) và Vector Search (Semantic),
    sau đó Rerank kết quả bằng FlashRank.
    """
    vector_retriever: BaseRetriever
    bm25_retriever: BaseRetriever
    ensemble_retriever: EnsembleRetriever
    reranker: Ranker

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_documents(cls, documents: List[Document], vector_store: VectorStore, k: int = 4):
        # 1. Init BM25
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = k

        # 2. Init Vector Retriever
        vector_retriever = vector_store.as_retriever(search_kwargs={"k": k})

        # 3. Init Ensemble (Weight 0.5 - 0.5)
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.5, 0.5]
        )

        # 4. Init Reranker (Lightweight model)
        reranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="./.cache/flashrank")

        return cls(
            vector_retriever=vector_retriever,
            bm25_retriever=bm25_retriever,
            ensemble_retriever=ensemble_retriever,
            reranker=reranker
        )

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        # Step 1: Hybrid Search (BM25 + Vector) -> Get ~ 2*k candidates
        initial_docs = self.ensemble_retriever.invoke(query)
        
        if not initial_docs:
            return []

        # Step 2: Rerank using Cross-Encoder
        passages = [
            {"id": str(i), "text": doc.page_content, "meta": doc.metadata} 
            for i, doc in enumerate(initial_docs)
        ]
        
        rerank_request = RerankRequest(query=query, passages=passages)
        results = self.reranker.rerank(rerank_request)

        # Step 3: Format output
        # Sort by score and return top k (default 3 or 4)
        # Assuming we want top 3 final docs
        final_docs = []
        for res in results[:3]:
            # Reconstruct Document object
            original_doc = initial_docs[int(res["id"])]
            final_docs.append(Document(
                page_content=res["text"],
                metadata=original_doc.metadata
            ))
            
        return final_docs
