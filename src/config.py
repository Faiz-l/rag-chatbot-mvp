import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    VECTOR_STORE_DIR: str = os.getenv("VECTOR_STORE_DIR", "./data/vectorstore")
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "./data/uploads")
    
    # Model configs
    EMBEDDING_MODEL: str = "models/embedding-001"
    LLM_MODEL: str = "gemini-1.5-flash"

settings = Settings()

# Ensure directories exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.VECTOR_STORE_DIR, exist_ok=True)