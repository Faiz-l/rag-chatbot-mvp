import os
from typing import List
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from src.config import settings

class VectorStoreManager:
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            google_api_key=settings.GOOGLE_API_KEY
        )
        self.index_path = os.path.join(settings.VECTOR_STORE_DIR, "faiss_index")
        self.vector_store = None
        self._load_or_create()

    def _load_or_create(self):
        """Pehle se saved index load karta hai ya new initialize karta hai."""
        if os.path.exists(self.index_path):
            self.vector_store = FAISS.load_local(
                self.index_path, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )

    def add_documents(self, documents: List[Document]):
        """Chunks ko FAISS vector database mein store aur persist karta hai."""
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
        else:
            self.vector_store.add_documents(documents)
        
        # Local system par index save karein
        self.vector_store.save_local(self.index_path)

    def get_retriever(self, k: int = 3):
        """Top-K relevant chunks search karne ke liye retriever return karta hai."""
        if self.vector_store is None:
            return None
        return self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )