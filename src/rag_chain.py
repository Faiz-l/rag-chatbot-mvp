from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from src.config import settings
from src.vector_store import VectorStoreManager

class RAGChainManager:
    def __init__(self, vector_manager: VectorStoreManager):
        self.vector_manager = vector_manager
        self.llm = ChatGoogleGenerativeAI(
            model=settings.LLM_MODEL,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=0.2  # Low temperature reduces hallucinations
        )
        
        # FR-08 Strict Prompting to prevent inventing answers
        self.prompt_template = """System: Aap aik helpful assistant hain jo company ke official documents ki base par jawab dete hain.
Niche diye gaye Context ko dhyan se parhein aur user ke question ka answer dein.

STRICT INSTRUCTIONS:
1. Sirf context mein di gayi information par depend karein.
2. Agar context mein answer MAUJOOD NAHI HAI, toh EXACTLY ye line kahein:
   "Mujhe provided documents mein is question ka relevant answer nahi mila."
3. Apni taraf se koi information invent ya guess mat karein.
4. Urdu ya English jis language mein user poochay, ussi mein context ke mutabiq jawab dein.

Context:
{context}

User Question: {question}

Answer:"""

        self.prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=["context", "question"]
        )

    def answer_query(self, question: str) -> dict:
        retriever = self.vector_manager.get_retriever(k=3)
        retrieved_docs = retriever.get_relevant_documents(question)

        if not retrieved_docs:
            return {
                "answer": "Mujhe provided documents mein is question ka relevant answer nahi mila.",
                "sources": []
            }

        # Context string build karein
        context_str = "\n\n".join([f"--- Chunk {i+1} ---\n" + doc.page_content for i, doc in enumerate(retrieved_docs)])

        # Query execute karein
        formatted_prompt = self.prompt.format(context=context_str, question=question)
        response = self.llm.invoke(formatted_prompt)

        # Source references format karein (FR-07)
        sources = []
        for doc in retrieved_docs:
            file_name = doc.metadata.get("source_file", "Document")
            page = doc.metadata.get("page_number", "N/A")
            source_str = f"{file_name} — Page {page}"
            if source_str not in sources:
                sources.append(source_str)

        return {
            "answer": response.content.strip(),
            "sources": sources
        }