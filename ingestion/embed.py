from qdrant_client import QdrantClient
from qdrant_client.models import (Distance, VectorParams, PointStruct)
from sentence_transformers import SentenceTransformer
from models import LectureChunk
from config import config

COLLECTION = config.COLLECTION_NAME
EMBEDDING_MODEL = config.EMBEDDING_MODEL
VECTOR_SIZE = 384

def get_qdrant_client() -> QdrantClient:
    """Returns a client for either local or cloud Qdrant."""
    if config.QDRANT_URL:
        return QdrantClient(
            url=config.QDRANT_URL,
            api_key=config.QDRANT_API_KEY,
        )
    return QdrantClient(
        host=config.QDRANT_HOST,
        port=int(config.QDRANT_PORT),
    )

def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL)

def ensure_collection(client: QdrantClient):
    # Create collection if it doesn't exist.
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION not in existing:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        

def embed_and_upload(chunks: list[LectureChunk], batch_size: int = 32):
    client = get_qdrant_client()
    embedder = get_embedder()
    ensure_collection(client)

    texts = [chunk.text for chunk in chunks]
    embeddings = embedder.encode(texts, batch_size=batch_size, show_progress_bar=True)

    points = []
    for chunk, vector in zip(chunks, embeddings):
        #timestamp URL
        timestamp_url = None
        if chunk.video_url and "youtube.com" in (chunk.video_url or ""):
            video_id = chunk.video_url.split("v=")[-1].split("&")[0]
            timestamp_url = f"https://www.youtube.com/watch?v={video_id}&t={int(chunk.start_time)}"

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
            "timestamp_url":    timestamp_url,
        }

        points.append(PointStruct(
            id=abs(hash(chunk.chunk_id)) % (2**63),  # Qdrant needs int ID
            vector=vector.tolist(),
            payload=payload,
        ))

    client.upsert(collection_name=COLLECTION, points=points)
