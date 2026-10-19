from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, Range, MatchValue, MatchAny
from sentence_transformers import SentenceTransformer
from models import SourceCitation
from config import config
from dataclasses import dataclass
from typing import Optional

COLLECTION = config.COLLECTION_NAME

@dataclass
class RetrievalConfig:
    top_k: int = 5
    session_from: Optional[int] = None     
    session_to: Optional[int] = None       
    session_id: Optional[str] = None    
    topic: Optional[str] = None         
    score_threshold: float = 0.3    

class TemporalRetriever:
    def __init__(self):
        from ingestion.embed import get_qdrant_client, get_embedder
        self.client = get_qdrant_client()
        self.embedder = get_embedder()

    def _build_filter(self, config: RetrievalConfig) -> Optional[Filter]:
        conditions = []

        if config.session_from is not None:
            conditions.append(FieldCondition(
                key="session_number",
                range=Range(gte=config.session_from)
            ))

        if config.session_to is not None:
            conditions.append(FieldCondition(
                key="session_number",
                range=Range(lte=config.session_to)
            ))

        if config.session_id:
            conditions.append(FieldCondition(
                key="session_id",
                match=MatchValue(value=config.session_id)
            ))

        if config.topic:
            conditions.append(FieldCondition(
                key="topic",
                match=MatchValue(value=config.topic)
            ))

        if not conditions:
            return None

        return Filter(must=conditions)

    def retrieve(self, query: str, config: RetrievalConfig = None) -> list[dict]:
        if config is None:
            config = RetrievalConfig()

        query_vector = self.embedder.encode(query).tolist()
        qdrant_filter = self._build_filter(config)

        response = self.client.query_points(
            collection_name=COLLECTION,
            query=query_vector,
            query_filter=qdrant_filter,
            limit=config.top_k,
            score_threshold=config.score_threshold,
            with_payload=True,
        )

        return [
            {"payload": r.payload, "score": r.score}
            for r in response.points
        ]

    def retrieve_for_contradiction(self, query: str, top_k: int = 6) -> list[dict]:
        #results from early sessions
        early = self.retrieve(query, RetrievalConfig(
            top_k=top_k // 2,
            session_to=4,
        ))
        #results from later sessions
        late = self.retrieve(query, RetrievalConfig(
            top_k=top_k // 2,
            session_from=5,
        ))
        return early + late

    def get_concept_timeline(self, concept: str, top_k: int = 8) -> list[dict]:
        #Returns all mentions of a concept sorted by session number.
        results = self.retrieve(concept, RetrievalConfig(top_k=top_k))
        # Sort by session for chronological display
        return sorted(results, key=lambda r: r["payload"]["session_number"])

    def payload_to_citation(self, payload: dict, excerpt_length: int = 200) -> SourceCitation:
        """Convert a Qdrant payload dict → SourceCitation for the UI."""
        text = payload.get("text", "")
        return SourceCitation(
            chunk_id=f"{payload['session_id']}_{int(payload['start_time'])}",
            session_id=payload["session_id"],
            session_number=payload["session_number"],
            timestamp_label=payload["timestamp_label"],
            video_url=payload.get("timestamp_url"),  
            timestamp_seconds=payload["start_time"],
            excerpt=text[:excerpt_length] + ("..." if len(text) > excerpt_length else ""),
        )
