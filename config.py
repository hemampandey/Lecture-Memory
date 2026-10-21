import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

class Config:
    # --- Project Paths ---
    BASE_DIR = Path(__file__).resolve().parent
    DATA_DIR = BASE_DIR / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"

    # Ensure directories exist
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # --- Model Settings ---
    WHISPER_MODEL = os.getenv("WHISPER_MODEL", "tiny")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

    # --- Qdrant Settings ---
    # Set QDRANT_URL + QDRANT_API_KEY for a hosted cluster (required when deployed).
    # Leave QDRANT_URL empty to fall back to a local Docker instance at QDRANT_HOST:QDRANT_PORT.
    QDRANT_URL = (os.getenv("QDRANT_URL") or "").strip() or None
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
    QDRANT_TIMEOUT = int(os.getenv("QDRANT_TIMEOUT", "30"))
    COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "lecture_memory")

    # --- API Settings ---
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

config = Config()
