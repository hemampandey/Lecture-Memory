"""
ingest.py — One command to process a full lecture end-to-end.

Use:
  python ingest.py --url "https://youtube.com/watch?v=..." --session 1 --label "Intro to Deep Learning"
  python ingest.py --file data/raw/lecture.mp3 --session 2 --label "Backpropagation"
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.append(".")
from ingestion.transcribe import download_audio, transcribe, save_transcript, cleanup_audio
from ingestion.chunker import build_chunks
from ingestion.embed import embed_and_upload
from config import config
from logger import logger


def ingest_youtube(url: str, session: int, label: str, whisper_model: str = "base", use_llm_topics: bool = True):
    session_id = f"session_{session:02d}"

    # Step 1: Download
    audio_path, detected_title = download_audio(url, output_dir=str(config.RAW_DATA_DIR))
    label = label or detected_title

    # Step 2: Transcribe
    raw_path = str(config.RAW_DATA_DIR / f"{session_id}_raw.json")
    segments = transcribe(audio_path, model_size=whisper_model)
    save_transcript(segments, raw_path)

    # Step 3: Chunk + topic tag
    chunks = build_chunks(
        segments=segments,
        session_id=session_id,
        session_number=session,
        session_label=label, 
        video_url=url,
        video_title=label,
        use_llm_topics=use_llm_topics,
    )

    # Step 4: Embed + upload to Qdrant
    embed_and_upload(chunks)

    # Step 5: Cleanup
    cleanup_audio(audio_path)


def ingest_local(file_path: str, session: int, label: str, whisper_model: str = "base", use_llm_topics: bool = True):
    session_id = f"session_{session:02d}"

    segments = transcribe(file_path, model_size=whisper_model)
    raw_path = str(config.RAW_DATA_DIR / f"{session_id}_raw.json")
    save_transcript(segments, raw_path)

    chunks = build_chunks(
        segments=segments,
        session_id=session_id,
        session_number=session,
        session_label=label,
        video_url=None,
        video_title=label,
        use_llm_topics=use_llm_topics,
    )

    embed_and_upload(chunks)
    
    # We don't cleanup local files provided by path unless we specifically want to
    # but for Consistency we could. Let's keep the source file for local uploads.


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
    if args.url:
        ingest_youtube(args.url, args.session, args.label, args.model, use_llm_topics=use_llm)
    elif args.file:
        ingest_local(args.file, args.session, args.label, args.model, use_llm_topics=use_llm)
    else:
        print("Error: provide --url or --file")
        sys.exit(1)
