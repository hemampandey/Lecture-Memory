# Detects when the same topic is discussed differently across sessions

from openai import OpenAI
from config import config
from logger import logger
from retrieval.retriever import TemporalRetriever
import json

client = OpenAI(
    api_key=config.OPENAI_API_KEY,
    base_url=config.OPENAI_BASE_URL
)

CONTRADICTION_PROMPT = """You are analyzing lecture excerpts on the same topic from different sessions.

Determine if these excerpts:
1. CONTRADICT each other (genuinely conflicting claims)
2. EVOLVE (later content builds on or refines earlier content)
3. COMPLEMENT each other (different angles on same topic)
4. REPEAT (essentially the same content)

Return JSON only:
{{
  "relationship": "contradict" | "evolve" | "complement" | "repeat",
  "found_contradiction": true | false,
  "explanation": "1-2 sentences explaining what changed and why it matters",
  "earlier_claim": "brief summary of what was said earlier",
  "later_claim": "brief summary of what was said later"
}}

Excerpt A (Session {session_a}, {timestamp_a}):
{text_a}

Excerpt B (Session {session_b}, {timestamp_b}):
{text_b}
"""

def check_contradiction(query: str) -> dict:
    """Retrieves temporally diverse chunks and checks for contradictions."""
    retriever = TemporalRetriever()
    chunks = retriever.retrieve_for_contradiction(query, top_k=6)

    if len(chunks) < 2:
        return {"found": False, "explanation": None}

    sorted_chunks = sorted(chunks, key=lambda c: c["payload"]["session_number"])
    earliest = sorted_chunks[0]["payload"]
    latest = sorted_chunks[-1]["payload"]

    if earliest["session_number"] == latest["session_number"]:
        return {"found": False, "explanation": "All relevant content is from the same session."}

    prompt = CONTRADICTION_PROMPT.format(
        session_a=earliest["session_number"],
        timestamp_a=earliest["timestamp_label"],
        text_a=earliest["text"][:600],
        session_b=latest["session_number"],
        timestamp_b=latest["timestamp_label"],
        text_b=latest["text"][:600],
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0,
    )

    result = json.loads(response.choices[0].message.content)

    return {
        "found": result.get("found_contradiction", False),
        "relationship": result.get("relationship"),
        "explanation": result.get("explanation"),
        "earlier_claim": result.get("earlier_claim"),
        "later_claim": result.get("later_claim"),
        "earlier_citation": retriever.payload_to_citation(earliest),
        "later_citation": retriever.payload_to_citation(latest),
    }