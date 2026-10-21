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

    # --- Transcription ---
    # local = openai-whisper on this machine (needs `uv sync --extra local`)
    # api   = OpenAI-compatible /audio/transcriptions endpoint (use this when deployed)
    TRANSCRIBE_BACKEND = os.getenv("TRANSCRIBE_BACKEND", "local").strip().lower()
    TRANSCRIBE_MODEL = os.getenv("TRANSCRIBE_MODEL", "whisper-1")
    # Set TRANSCRIBE_API_KEY (+ TRANSCRIBE_BASE_URL) when your LLM provider has no audio API;
    # otherwise the OPENAI_* credentials are reused.
    if os.getenv("TRANSCRIBE_API_KEY"):
        TRANSCRIBE_API_KEY = os.getenv("TRANSCRIBE_API_KEY")
        TRANSCRIBE_BASE_URL = os.getenv("TRANSCRIBE_BASE_URL") or None
    else:
        TRANSCRIBE_API_KEY = OPENAI_API_KEY
        TRANSCRIBE_BASE_URL = os.getenv("TRANSCRIBE_BASE_URL") or OPENAI_BASE_URL
    # OpenAI rejects uploads over 25MB; larger audio is compressed and split below this
    TRANSCRIBE_MAX_MB = float(os.getenv("TRANSCRIBE_MAX_MB", "24"))

    # --- YouTube Ingestion ---
    # Try existing captions before downloading audio (cloud IPs are often blocked from downloads)
    YOUTUBE_CAPTIONS = os.getenv("YOUTUBE_CAPTIONS", "true").strip().lower() in ("1", "true", "yes")
    CAPTION_LANGUAGES = [l.strip() for l in os.getenv("CAPTION_LANGUAGES", "en").split(",") if l.strip()]
    # Optional http(s) proxy for YouTube requests, e.g. http://user:pass@host:port
    YOUTUBE_PROXY_URL = (os.getenv("YOUTUBE_PROXY_URL") or "").strip() or None

    # Default for the UI's "tag topics with the LLM" option (one LLM call per chunk)
    USE_LLM_TOPICS = os.getenv("USE_LLM_TOPICS", "false").strip().lower() in ("1", "true", "yes")

config = Config()
