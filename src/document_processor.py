import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class DocumentProcessor:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        # Text chunking strategy based on BRD Requirements
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def process_pdf(self, file_path: str) -> List[Document]:
        """PDF Load karta hai, text extract karta hai aur chunks banata hai metadata ke sath."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # PyPDFLoader automatically preserves page numbers in metadata
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        # Split document into chunks
        chunks = self.text_splitter.split_documents(documents)
        
        # Add additional Metadata (FR-03)
        file_name = os.path.basename(file_path)
        for chunk in chunks:
            chunk.metadata["source_file"] = file_name
            # Page indexing human-readable banane ke liye +1 karein (1-based index)
            if "page" in chunk.metadata:
                chunk.metadata["page_number"] = chunk.metadata["page"] + 1

        return chunks