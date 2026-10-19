from retrieval.graph import graph
from retrieval.retriever import TemporalRetriever
from models import QueryResult
from config import config
from logger import logger

retriever = TemporalRetriever()

def answer(
    query: str,
    session_from: int = None,
    session_to: int = None,
    check_for_contradictions: bool = True,
) -> QueryResult:
    logger.info(f"Answering query: {query}")
    
    initial_state = {
        "query": query,
        "session_from": session_from,
        "session_to": session_to,
        "chunks": [],
        "contradiction": {"found": False},
        "answer": ""
    }
    
    final_state = graph.invoke(initial_state)
    
    sources = [retriever.payload_to_citation(c["payload"]) for c in final_state["chunks"]]
    
    return QueryResult(
        answer=final_state["answer"],
        sources=sources,
        contradiction_found=final_state["contradiction"]["found"],
        contradiction_explanation=final_state["contradiction"].get("explanation"),
    )

def concept_timeline(concept: str) -> list[dict]:
    from retrieval.retriever import TemporalRetriever
    from openai import OpenAI
    
    retriever = TemporalRetriever()
    client = OpenAI(api_key=config.OPENAI_API_KEY, base_url=config.OPENAI_BASE_URL)
    
    chunks = retriever.get_concept_timeline(concept, top_k=8)
    if not chunks:
        return []

    timeline = []
    for c in chunks:
        p = c["payload"]
        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": f"Summarize in one sentence how this excerpt discusses '{concept}':\n\n{p['text'][:400]}"}],
            temperature=0
        )
        timeline.append({
            "session_number": p["session_number"],
            "timestamp_label": p["timestamp_label"],
            "timestamp_url": p.get("timestamp_url"),
            "summary": response.choices[0].message.content.strip(),
        })
    return timeline
