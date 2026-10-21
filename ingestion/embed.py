import uuid
from functools import lru_cache
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, PayloadSchemaType,
    Filter, FieldCondition, MatchValue, FilterSelector,
)
from sentence_transformers import SentenceTransformer
from models import LectureChunk
from config import config
from logger import logger

COLLECTION = config.COLLECTION_NAME
EMBEDDING_MODEL = config.EMBEDDING_MODEL

# Fields we filter on — Qdrant Cloud needs payload indexes for filtered search
PAYLOAD_INDEXES = {
    "session_number": PayloadSchemaType.INTEGER,
    "session_id": PayloadSchemaType.KEYWORD,
    "topic": PayloadSchemaType.KEYWORD,
}

@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    """Returns a shared client for either cloud (QDRANT_URL) or local Qdrant."""
    if config.QDRANT_URL:
        logger.info("Using hosted Qdrant (QDRANT_URL)")
        return QdrantClient(
            url=config.QDRANT_URL,
            api_key=config.QDRANT_API_KEY,
            timeout=config.QDRANT_TIMEOUT,
        )
    logger.warning(f"QDRANT_URL not set — using local Qdrant at {config.QDRANT_HOST}:{config.QDRANT_PORT}")
    return QdrantClient(
        host=config.QDRANT_HOST,
        port=config.QDRANT_PORT,
        timeout=config.QDRANT_TIMEOUT,
    )

@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    # Loaded once per process — the model is ~90MB in memory
    return SentenceTransformer(EMBEDDING_MODEL)

def ensure_collection(client: QdrantClient):
    # Create collection + payload indexes if missing (safe to call repeatedly)
    if not client.collection_exists(COLLECTION):
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(
                size=get_embedder().get_embedding_dimension(),
                distance=Distance.COSINE,
            ),
        )

    existing = client.get_collection(COLLECTION).payload_schema or {}
    for field, schema in PAYLOAD_INDEXES.items():
        if field not in existing:
            client.create_payload_index(COLLECTION, field_name=field, field_schema=schema)

def point_id(chunk_id: str) -> str:
    # Deterministic UUID so re-ingesting the same chunk overwrites instead of duplicating
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"lecture-memory/{chunk_id}"))

def delete_session(client: QdrantClient, session_id: str):
    # Remove every chunk of a session so a re-ingest fully replaces it
    client.delete(
        collection_name=COLLECTION,
        points_selector=FilterSelector(filter=Filter(must=[
            FieldCondition(key="session_id", match=MatchValue(value=session_id))
        ])),
    )

def youtube_timestamp_url(video_url: str | None, seconds: float) -> str | None:
    if not video_url:
        return None
    if "youtu.be/" in video_url:
        video_id = video_url.split("youtu.be/")[-1].split("?")[0].split("&")[0]
    elif "youtube.com" in video_url and "v=" in video_url:
        video_id = video_url.split("v=")[-1].split("&")[0]
    else:
        return None
    return f"https://www.youtube.com/watch?v={video_id}&t={int(seconds)}"

def embed_and_upload(chunks: list[LectureChunk], batch_size: int = 32, upload_batch_size: int = 64):
    if not chunks:
        return

    client = get_qdrant_client()
    embedder = get_embedder()
    ensure_collection(client)

    texts = [chunk.text for chunk in chunks]
    embeddings = embedder.encode(texts, batch_size=batch_size, show_progress_bar=True)

    points = []
    for chunk, vector in zip(chunks, embeddings):
        payload = {
            # Metadata
            "session_number": chunk.session_number,
            "session_label":  chunk.session_label,
            "session_date":   chunk.session_date,
            "session_id":       chunk.session_id,
            "start_time":       chunk.start_time,
            "end_time":         chunk.end_time,
            "timestamp_label":  chunk.timestamp_label,

            # Content
            "text":             chunk.text,
            "speaker":          chunk.speaker.value,
            "topic":            chunk.topic,
            "keywords":         chunk.keywords,

            # Source
            "video_url":        chunk.video_url,
            "video_title":      chunk.video_title,
            "timestamp_url":    youtube_timestamp_url(chunk.video_url, chunk.start_time),
        }

        points.append(PointStruct(
            id=point_id(chunk.chunk_id),
            vector=vector.tolist(),
            payload=payload,
        ))

    for session_id in {chunk.session_id for chunk in chunks}:
        delete_session(client, session_id)

    # Upload in batches to stay under hosted request-size limits
    for i in range(0, len(points), upload_batch_size):
        client.upsert(collection_name=COLLECTION, points=points[i:i + upload_batch_size], wait=True)
    logger.info(f"Uploaded {len(points)} chunks to '{COLLECTION}'")
