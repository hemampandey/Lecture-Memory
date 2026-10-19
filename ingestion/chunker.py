from openai import OpenAI
import json
from config import config
from logger import logger
from models import LectureChunk, Speaker
from utils import format_timestamp

client = OpenAI(
    api_key=config.OPENAI_API_KEY,
    base_url=config.OPENAI_BASE_URL
)

def merge_segments(
    segments: list[dict],
    target_duration: float = 75.0,   # seconds per chunk
    min_words: int = 30,
) -> list[dict]:
    """Merge Whisper micro-segments into larger semantic chunks."""
    chunks = []
    current_texts = []
    current_start = None
    current_end = None

    for seg in segments:
        if current_start is None:
            current_start = seg["start_time"]

        current_texts.append(seg["text"])
        current_end = seg["end_time"]

        duration = current_end - current_start
        word_count = sum(len(t.split()) for t in current_texts)

        if duration >= target_duration and word_count >= min_words:
            chunks.append({
                "text": " ".join(current_texts).strip(),
                "start_time": current_start,
                "end_time": current_end,
            })
            current_texts = []
            current_start = None

    #last partial chunk
    if current_texts:
        chunks.append({
            "text": " ".join(current_texts).strip(),
            "start_time": current_start,
            "end_time": current_end,
        })
    return chunks


def assign_topic_with_llm(text: str) -> tuple[str, list[str]]:
    #Tags a chunk with a topic and keywords using the LLM.
    try:
        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{
                "role": "user",
                "content": f"Analyze this lecture excerpt and return JSON (topic, keywords): {text[:800]}"
            }],
            response_format={"type": "json_object"},
            temperature=0,
        )
        data = json.loads(response.choices[0].message.content)
        return data.get("topic", "General"), data.get("keywords", [])
    except Exception as e:
        logger.error(f"LLM tagging failed: {e}")
        return "General", []


def build_chunks(
    segments: list[dict],
    session_id: str,
    session_number: int,
    session_label: str = "",
    video_url: str = None,
    video_title: str = None,
    speaker: Speaker = Speaker.PROFESSOR,
    use_llm_topics: bool = True,
) -> list[LectureChunk]:
    merged = merge_segments(segments)
    chunks = []

    for i, seg in enumerate(merged):
        topic, keywords = "General", []
        if use_llm_topics:
            topic, keywords = assign_topic_with_llm(seg["text"])
            logger.info(f"Chunk {i+1} tagged: '{topic}'")

        chunks.append(LectureChunk(
            chunk_id=f"{session_id}_{int(seg['start_time'])}",
            session_id=session_id,
            session_number=session_number,
            session_label=session_label or f"Session {session_number}",
            text=seg["text"],
            speaker=speaker,
            start_time=seg["start_time"],
            end_time=seg["end_time"],
            timestamp_label=format_timestamp(seg["start_time"]),
            topic=topic,
            keywords=keywords,
            video_url=video_url,
            video_title=video_title,
        ))

    return chunks
