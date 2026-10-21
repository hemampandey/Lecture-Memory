"""
Use this as a diagnostic tool
RUN: python main.py
"""


import sys
from config import config
from logger import logger
from ingestion.embed import get_qdrant_client

def check_system():
    """Verify that the environment is ready for Lecture Memory."""
    logger.info("🚦 Starting System Diagnostic...")

    # 1. Check Qdrant (cloud if QDRANT_URL is set, else local Docker)
    target = "Qdrant Cloud (QDRANT_URL)" if config.QDRANT_URL else f"local Qdrant {config.QDRANT_HOST}:{config.QDRANT_PORT}"
    try:
        client = get_qdrant_client()
        if client.collection_exists(config.COLLECTION_NAME):
            count = client.count(config.COLLECTION_NAME).count
            logger.info(f"✅ {target}: CONNECTED — '{config.COLLECTION_NAME}' has {count} chunks")
        else:
            logger.info(f"✅ {target}: CONNECTED — collection '{config.COLLECTION_NAME}' will be created on first ingest")
    except Exception as e:
        hint = "Check the cluster is running in the Qdrant Cloud console and the API key is valid." if config.QDRANT_URL else "Ensure Docker is running."
        logger.error(f"❌ {target}: FAILED. {hint} Error: {e}")

    # 2. Check OpenAI / API
    if config.OPENAI_API_KEY and len(config.OPENAI_API_KEY) > 10:
        logger.info("✅ API Key: CONFIGURED")
    else:
        logger.warning("⚠️ API Key: MISSING or INVALID in .env")

    # 3. Check Folders
    if config.RAW_DATA_DIR.exists():
        logger.info(f"✅ Data Directory: READY ({config.RAW_DATA_DIR})")
    else:
        config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("✅ Data Directory: CREATED")

    logger.info("--- Diagnostic Complete ---")
    print("\nTo start the UI, run:  uv run streamlit run ui/app.py")
    print("To ingest data, run:   uv run python ingest.py --url <URL> --session <N>\n")

if __name__ == "__main__":
    check_system()
