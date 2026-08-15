import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from src.config import settings
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStoreManager
from src.rag_chain import RAGChainManager

app = FastAPI(title="BRD RAG Chatbot MVP", version="1.0")

# Setup templates
templates = Jinja2Templates(directory="templates")

# Module Initializations
doc_processor = DocumentProcessor()
vector_manager = VectorStoreManager()
rag_chain = RAGChainManager(vector_manager)

class QueryRequest(BaseModel):
    question: str

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """UI Interface serve karne ke liye"""
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """FR-01 & FR-02: Document upload and ingest endpoint"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Filhal sirf PDF files supported hain.")

    file_path = os.path.join(settings.UPLOAD_DIR, file.filename)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Process and ingest
        chunks = doc_processor.process_pdf(file_path)
        vector_manager.add_documents(chunks)
        
        return {
            "status": "success", 
            "message": f"'{file.filename}' successfully process ho gaya hai aur KB mein add ho gaya hai.",
            "total_chunks": len(chunks)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat_endpoint(request: QueryRequest):
    """FR-04 to FR-08: User Question & RAG Response endpoint"""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question empty nahi hona chahiye.")

    try:
        result = rag_chain.answer_query(request.question)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))