import whisper
import yt_dlp
import os
import json
from pathlib import Path
from tqdm import tqdm
from config import config

def download_audio(youtube_url: str, output_dir: str = None) -> tuple[str,str]:
    # Download audio from YT lecture. Returns path to audio file.
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

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:     #Downloads audio and info
        info = ydl.extract_info(youtube_url, download=True)
        video_id = info["id"]
        title = info.get("title", "Unknown")
        audio_path = f"{output_dir}/{video_id}.mp3"

    return audio_path, title


def transcribe(audio_path: str, model_size: str = "base") -> list[dict]:
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

def save_transcript(segments: list[dict], output_path: str):
    with open(output_path, "w") as f:       # Saves transcript to output_path
        json.dump(segments, f, indent=2)

def cleanup_audio(audio_path: str): #Delete the downloaded audio file
    if os.path.exists(audio_path):
        os.remove(audio_path)
