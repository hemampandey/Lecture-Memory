"""
ingest.py — One command to process a full lecture end-to-end.

Use:
  python ingest.py --url "https://youtube.com/watch?v=..." --session 1 --label "Intro to Deep Learning"
  python ingest.py --file data/raw/lecture.mp3 --session 2 --label "Backpropagation"

YouTube lectures use existing captions when available (YOUTUBE_CAPTIONS=true), falling back to
audio download + transcription. Transcription runs locally or via API (TRANSCRIBE_BACKEND).
"""

import argparse
import sys
from pathlib import Path

sys.path.append(".")
from ingestion.transcribe import (
    Progress, notify, extract_video_id, fetch_captions, fetch_video_title,
    download_audio, transcribe, save_transcript, cleanup_audio,
)
from ingestion.chunker import build_chunks
from ingestion.embed import embed_and_upload
from config import config
from logger import logger


class IngestionError(RuntimeError):
    """A lecture couldn't be ingested for a reason the user can act on."""


def _short_error(e: Exception) -> str:
    # youtube-transcript-api / yt-dlp errors are multi-paragraph; keep the cause only
    text = getattr(e, "cause", None) or str(e) or type(e).__name__
    text = text.strip().split("\n\n")[0].replace("ERROR: ", "").strip()
    return text if len(text) <= 300 else text[:300] + "…"


def _index_segments(
    segments: list[dict], session: int, label: str, video_url: str | None,
    use_llm_topics: bool, progress: Progress,
) -> dict:
    session_id = f"session_{session:02d}"
    if not segments:
        raise IngestionError("The transcript is empty — nothing to index.")

    save_transcript(segments, str(config.RAW_DATA_DIR / f"{session_id}_raw.json"))

    # Chunk + topic tag
    tagging = " and tagging topics with the LLM" if use_llm_topics else ""
    notify(progress, f"Chunking {len(segments)} transcript segments{tagging}…")
    chunks = build_chunks(
        segments=segments,
        session_id=session_id,
        session_number=session,
        session_label=label,
        video_url=video_url,
        video_title=label,
        use_llm_topics=use_llm_topics,
    )

    # Embed + upload to Qdrant
    notify(progress, f"Embedding {len(chunks)} chunks and uploading to Qdrant…")
    embed_and_upload(chunks)

    return {"session_id": session_id, "label": label, "segments": len(segments), "chunks": len(chunks)}


def get_youtube_segments(url: str, whisper_model: str = "base", progress: Progress = None) -> tuple[list[dict], str | None, str]:
    """Returns (segments, detected_title, source) where source is 'captions' or 'audio'."""
    if not extract_video_id(url):
        raise IngestionError(f"That doesn't look like a YouTube video URL: {url}")

    errors = []
    if config.YOUTUBE_CAPTIONS:
        notify(progress, "Fetching YouTube captions…")
        try:
            segments = fetch_captions(url)
            if segments:
                notify(progress, f"Found captions ({len(segments)} lines).")
                return segments, None, "captions"
            errors.append("Captions: the caption track is empty.")
        except Exception as e:
            logger.warning(f"Captions unavailable: {e}")
            errors.append(f"Captions: {_short_error(e)}")
        notify(progress, "No usable captions — falling back to downloading audio…")
    else:
        notify(progress, "Downloading audio…")

    try:
        audio_path, title = download_audio(url, output_dir=str(config.RAW_DATA_DIR))
    except Exception as e:
        logger.warning(f"Audio download failed: {e}")
        errors.append(f"Audio download: {_short_error(e)}")
        raise IngestionError(
            "Couldn't get this lecture from YouTube.\n\n"
            + "\n".join(f"- {err}" for err in errors)
            + "\n\nYouTube often blocks requests from cloud servers. Download the video/audio "
              "yourself and use **Upload file**, or set `YOUTUBE_PROXY_URL`."
        ) from e

    try:
        segments = transcribe(audio_path, model_size=whisper_model, progress=progress)
    finally:
        cleanup_audio(audio_path)
    return segments, title, "audio"


def ingest_youtube(url: str, session: int, label: str, whisper_model: str = "base",
                   use_llm_topics: bool = True, progress: Progress = None) -> dict:
    segments, detected_title, source = get_youtube_segments(url, whisper_model, progress)
    label = label or detected_title or fetch_video_title(url) or f"Session {session}"
    result = _index_segments(segments, session, label, url, use_llm_topics, progress)
    return {**result, "source": source}


def ingest_local(file_path: str, session: int, label: str, whisper_model: str = "base",
                 use_llm_topics: bool = True, progress: Progress = None) -> dict:
    segments = transcribe(file_path, model_size=whisper_model, progress=progress)
    label = label or Path(file_path).stem
    result = _index_segments(segments, session, label, None, use_llm_topics, progress)

    # We don't cleanup local files provided by path unless we specifically want to
    # but for Consistency we could. Let's keep the source file for local uploads.
    return {**result, "source": "file"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest a lecture into Lecture Memory")
    parser.add_argument("--url", help="YouTube URL")
    parser.add_argument("--file", help="Local audio/video file path")
    parser.add_argument("--session", type=int, required=True)
    parser.add_argument("--label", default="", help="Session label e.g. 'Episode 4'")
    parser.add_argument("--title", default="", help="Lecture title")
    parser.add_argument("--model", default="base", help="Whisper model size (tiny/base/small/medium)")
    parser.add_argument("--no-llm-topics", action="store_true", help="Skip LLM topic tagging")
    args = parser.parse_args()

    use_llm = not args.no_llm_topics
    try:
        if args.url:
            result = ingest_youtube(args.url, args.session, args.label, args.model, use_llm_topics=use_llm)
        elif args.file:
            result = ingest_local(args.file, args.session, args.label, args.model, use_llm_topics=use_llm)
        else:
            print("Error: provide --url or --file")
            sys.exit(1)
    except IngestionError as e:
        print(f"Error: {e}")
        sys.exit(1)
    print(f"Ingested '{result['label']}' as {result['session_id']}: {result['chunks']} chunks (source: {result['source']})")
