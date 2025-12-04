import logging
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from app.core.config import settings

class RAGService:
    def __init__(self):
        self.qa_chain = None
        self.vector_store = None

    async def initialize(self):
        """Khởi tạo pipeline RAG."""
        try:
            logging.info("Đang khởi tạo hệ thống RAG...")
            
            # 1. Tải Dữ liệu
            loader = TextLoader(settings.KNOWLEDGE_BASE_PATH, encoding="utf-8")
            documents = loader.load()

            # 2. Chia nhỏ văn bản
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(documents)

            # 3. Khởi tạo Embeddings
            embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)

            # 4. Tạo Vector Store
            self.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                collection_name="knowledge_base"
            )

            # 5. Khởi tạo LLM
            llm = ChatGoogleGenerativeAI(model=settings.MODEL_NAME, google_api_key=settings.GOOGLE_API_KEY, temperature=0.3)

            # 6. Tạo chuỗi QA
            retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=False
            )
            logging.info("Hệ thống RAG đã sẵn sàng hoạt động.")
        except Exception as e:
            logging.error(f"Lỗi khởi động RAG: {e}")
            raise e

    def get_answer(self, question: str) -> str:
        """Trả lời câu hỏi."""
        if not self.qa_chain:
            raise Exception("Hệ thống chưa sẵn sàng.")
        
        response = self.qa_chain.invoke({"query": question})
        return response["result"]

rag_service = RAGService()
