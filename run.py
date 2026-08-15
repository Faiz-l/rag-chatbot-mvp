import uvicorn

if __name__ == "__main__":
    print("Starting RAG Chatbot Server on http://127.0.0.1:8000")
    uvicorn.run("src.main:app", host="127.0.0.1", port=8000, reload=True)