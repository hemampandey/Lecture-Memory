"""
Turns a lecture into timestamped segments: [{"text", "start_time", "end_time"}, ...]

Sources:
  - fetch_captions():  existing YouTube captions, no audio download needed
  - download_audio():  yt-dlp audio download (often blocked from cloud IPs)
  - transcribe():      local Whisper or an OpenAI-compatible transcription API (TRANSCRIBE_BACKEND)

Heavy/optional deps (whisper, torch, yt_dlp) are imported lazily so a deployment
using TRANSCRIBE_BACKEND=api never needs them installed.
"""

import csv
import json
import math
import os
import shutil
import subprocess
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable, Optional
from config import config
from logger import logger

Progress = Optional[Callable[[str], None]]

# Formats the OpenAI transcription endpoint accepts as-is
API_AUDIO_FORMATS = {".flac", ".m4a", ".mp3", ".mp4", ".mpeg", ".mpga", ".oga", ".ogg", ".wav", ".webm"}


def notify(progress: Progress, message: str):
    logger.info(message)
    if progress:
        progress(message)


# ── YouTube ──────────────────────────────────────────────────────────────────

def extract_video_id(url: str) -> str | None:
    parsed = urllib.parse.urlparse(url.strip())
    host = (parsed.hostname or "").lower()
    if host.endswith("youtu.be"):
        return parsed.path.lstrip("/").split("/")[0] or None
    if host.endswith("youtube.com") or host.endswith("youtube-nocookie.com"):
        query_id = urllib.parse.parse_qs(parsed.query).get("v", [None])[0]
        if query_id:
            return query_id
        parts = [p for p in parsed.path.split("/") if p]
        if len(parts) >= 2 and parts[0] in ("shorts", "embed", "live", "v"):
            return parts[1]
    return None


def fetch_video_title(url: str) -> str | None:
    # oEmbed is a lightweight public endpoint, separate from the media/caption servers
    oembed = "https://www.youtube.com/oembed?format=json&url=" + urllib.parse.quote(url, safe="")
    try:
        with urllib.request.urlopen(oembed, timeout=10) as resp:
            return json.load(resp).get("title")
    except Exception as e:
        logger.warning(f"Could not fetch video title: {e}")
        return None


def fetch_captions(url: str, languages: list[str] = None) -> list[dict]:
    """Fetch existing YouTube captions as segments. Prefers `languages`, else any available track."""
    from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound
    from youtube_transcript_api.proxies import GenericProxyConfig

    video_id = extract_video_id(url)
    if not video_id:
        raise ValueError(f"Not a YouTube video URL: {url}")

    proxy = GenericProxyConfig(http_url=config.YOUTUBE_PROXY_URL, https_url=config.YOUTUBE_PROXY_URL) \
        if config.YOUTUBE_PROXY_URL else None
    api = YouTubeTranscriptApi(proxy_config=proxy)
    languages = languages or config.CAPTION_LANGUAGES

    try:
        fetched = api.fetch(video_id, languages=languages)
    except NoTranscriptFound:
        # No track in the preferred languages — take any, human-made first
        tracks = sorted(api.list(video_id), key=lambda t: t.is_generated)
        if not tracks:
            raise
        logger.info(f"No captions in {languages}; using '{tracks[0].language_code}'")
        fetched = tracks[0].fetch()

    snippets = [s for s in fetched.snippets if s.text.strip()]
    segments = []
    for i, s in enumerate(snippets):
        end = s.start + s.duration
        # Auto-generated captions overlap; clip each one at the next caption's start
        if i + 1 < len(snippets):
            end = min(end, snippets[i + 1].start)
        segments.append({
            "text": " ".join(s.text.split()),
            "start_time": s.start,
            "end_time": max(end, s.start),
        })
    return segments


def download_audio(youtube_url: str, output_dir: str = None) -> tuple[str,str]:
    # Download audio from YT lecture. Returns path to audio file.
    import yt_dlp

    if output_dir is None:
        output_dir = str(config.RAW_DATA_DIR)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": f"{output_dir}/%(id)s.%(ext)s",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "128",   # lower = faster, fine for speech
        }],
        "quiet": True,
    }
    if config.YOUTUBE_PROXY_URL:
        ydl_opts["proxy"] = config.YOUTUBE_PROXY_URL

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:     #Downloads audio and info
        info = ydl.extract_info(youtube_url, download=True)
        video_id = info["id"]
        title = info.get("title", "Unknown")
        audio_path = f"{output_dir}/{video_id}.mp3"

    return audio_path, title


# ── Transcription ────────────────────────────────────────────────────────────

def transcribe(audio_path: str, model_size: str = "base", backend: str = None, progress: Progress = None) -> list[dict]:
    backend = backend or config.TRANSCRIBE_BACKEND
    if backend == "api":
        return transcribe_api(audio_path, progress=progress)
    if backend == "local":
        return transcribe_local(audio_path, model_size=model_size, progress=progress)
    raise ValueError(f"Unknown TRANSCRIBE_BACKEND '{backend}' (expected 'local' or 'api')")


def transcribe_local(audio_path: str, model_size: str = "base", progress: Progress = None) -> list[dict]:
    try:
        import whisper
    except ImportError as e:
        raise RuntimeError(
            "Local Whisper isn't installed. Run `uv sync --extra local`, "
            "or set TRANSCRIBE_BACKEND=api to use a hosted transcription API."
        ) from e

    notify(progress, f"Transcribing with local Whisper ({model_size})…")
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path, verbose=False)

    segments = []
    for seg in result["segments"]:
        segments.append({
            "text": seg["text"].strip(),
            "start_time": seg["start"],
            "end_time": seg["end"],
        })

    return segments


def _field(obj, name):
    # SDK objects and plain dicts (some OpenAI-compatible providers) both show up here
    return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)


def _run_ffmpeg(args: list[str]):
    result = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args],
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.strip()[-500:]}")


def _split_for_api(audio_path: str, workdir: str) -> list[tuple[str, float]]:
    """Returns [(part_path, start_offset_seconds)], each part under TRANSCRIBE_MAX_MB."""
    limit = int(config.TRANSCRIBE_MAX_MB * 1024 * 1024)
    if os.path.getsize(audio_path) <= limit and Path(audio_path).suffix.lower() in API_AUDIO_FORMATS:
        return [(audio_path, 0.0)]

    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            f"ffmpeg is required to compress audio over {config.TRANSCRIBE_MAX_MB:g}MB "
            "or in an unsupported format (packages.txt installs it on Streamlit Cloud)."
        )

    # Mono 16kHz 32kbps is plenty for speech: ~14MB per hour
    compressed = os.path.join(workdir, "audio.mp3")
    _run_ffmpeg(["-i", audio_path, "-vn", "-ac", "1", "-ar", "16000", "-b:a", "32k", compressed])
    size = os.path.getsize(compressed)
    if size <= limit:
        return [(compressed, 0.0)]

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", compressed],
        capture_output=True, text=True, check=True,
    )
    duration = float(probe.stdout.strip())
    n_parts = math.ceil(size / (limit * 0.9))     # 10% headroom for uneven bitrate
    segment_time = math.ceil(duration / n_parts)

    # The segment list records each part's real start time, used to offset timestamps
    part_list = os.path.join(workdir, "parts.csv")
    _run_ffmpeg([
        "-i", compressed, "-f", "segment", "-segment_time", str(segment_time),
        "-c", "copy", "-reset_timestamps", "1",
        "-segment_list", part_list, "-segment_list_type", "csv",
        os.path.join(workdir, "part_%03d.mp3"),
    ])
    with open(part_list, newline="") as f:
        return [(os.path.join(workdir, row[0]), float(row[1])) for row in csv.reader(f) if row]


def transcribe_api(audio_path: str, progress: Progress = None) -> list[dict]:
    """Transcribe via an OpenAI-compatible /audio/transcriptions endpoint with segment timestamps."""
    from openai import OpenAI

    if not config.TRANSCRIBE_API_KEY:
        raise RuntimeError("TRANSCRIBE_BACKEND=api needs TRANSCRIBE_API_KEY or OPENAI_API_KEY.")

    client = OpenAI(api_key=config.TRANSCRIBE_API_KEY, base_url=config.TRANSCRIBE_BASE_URL, timeout=600)
    model = config.TRANSCRIBE_MODEL

    with tempfile.TemporaryDirectory(prefix="lm_audio_") as workdir:
        notify(progress, "Preparing audio for the transcription API…")
        parts = _split_for_api(audio_path, workdir)

        segments = []
        for i, (part_path, offset) in enumerate(parts, 1):
            notify(progress, f"Transcribing with {model} (part {i}/{len(parts)})…")
            with open(part_path, "rb") as f:
                resp = client.audio.transcriptions.create(
                    model=model,
                    file=f,
                    response_format="verbose_json",
                    timestamp_granularities=["segment"],
                )

            part_segments = _field(resp, "segments")
            if not part_segments:
                if (_field(resp, "text") or "").strip():
                    raise RuntimeError(
                        f"'{model}' returned text without segment timestamps. Use a Whisper model "
                        "that supports verbose_json, e.g. whisper-1 (OpenAI) or whisper-large-v3 (Groq)."
                    )
                continue  # silent part

            for seg in part_segments:
                text = (_field(seg, "text") or "").strip()
                if text:
                    segments.append({
                        "text": text,
                        "start_time": float(_field(seg, "start")) + offset,
                        "end_time": float(_field(seg, "end")) + offset,
                    })

    return segments


def save_transcript(segments: list[dict], output_path: str):
    with open(output_path, "w") as f:       # Saves transcript to output_path
        json.dump(segments, f, indent=2)

def cleanup_audio(audio_path: str): #Delete the downloaded audio file
    if os.path.exists(audio_path):
        os.remove(audio_path)
