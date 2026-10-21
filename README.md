# Lecture Memory

**Lecture Memory** is an advanced RAG (Retrieval-Augmented Generation) system designed for students to manage high-volume video lectures. It doesn't just search; it tracks how concepts evolve over and detects if topic contradicts a previous lecture.

---

<img width="1190" height="394" alt="Screenshot 2026-05-11 at 2 36 17 AM" src="https://github.com/user-attachments/assets/0a5a5a72-e31d-4c04-bd62-b89ac758fe63" />

---

## Key Features

- **🎬 Source Ingestion**: YouTube URLs (existing captions first, audio download as fallback) or uploaded audio/video files.
- **🎙️ Fast Transcription**: Local `OpenAI Whisper`, or any OpenAI-compatible transcription API for hosted deployments.
- **🤖 Agentic Workflow**: Used **LangGraph** for modular retrieval and synthesis.
- **🧠 Temporal RAG**: Retrieval that understands chronological sequence.
- **⚠️ Contradiction Detection**: Cross-references early lectures with later ones to find conflicting claims or evolved definitions.
- **⏳ Concept Timeline**: Visualizes how a single topic was introduced, expanded, and summarized across the entire course.

---

## Evaluation
<img width="1000" height="300" alt="Screenshot 2026-05-11 at 1 55 32 AM" src="https://github.com/user-attachments/assets/36b8f10f-0c9d-493e-8c2a-4d01b6731b7f" />

---

## Screenshots

### Ingestion

<img width="1000" height="400" alt="Screenshot 2026-05-10 at 5 11 05 PM" src="https://github.com/user-attachments/assets/ae2b1e13-f704-49f3-9959-143e4f8f584a" />

### Result

<img width="1400" height="750" alt="Screenshot 2026-05-10 at 5 10 51 PM" src="https://github.com/user-attachments/assets/dc421d8d-e8ca-4761-8d61-30a3b142f99c" />

<img width="1105" height="733" alt="Screenshot 2026-05-10 at 3 31 08 PM" src="https://github.com/user-attachments/assets/6044fa9a-76cc-4762-8995-71f927242d69" />

### Conept Timeline

<img width="297" height="245" alt="Screenshot 2026-05-10 at 3 31 59 PM" src="https://github.com/user-attachments/assets/c889f196-22b0-4e27-9437-e02dd4f72954" />

<img width="1127" height="752" alt="Screenshot 2026-05-10 at 3 32 08 PM" src="https://github.com/user-attachments/assets/8cae4d0d-3e92-46c5-87fa-fcf91d13315b" />

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Vector DB**: Qdrant (Local Docker)
- **Transcription**: YouTube captions, local OpenAI Whisper (`tiny` model for speed) or a hosted Whisper API
- **Embeddings**: `sentence-transformers` (all-MiniLM-L6-v2)
- **LLM**: GPT-4o-mini

---

## 💡 Engineering Challenges I Solved

### 1. The "Temporal Context" Problem
Most RAG systems just find the most similar text. But in a course, a definition in Week 1 might be simplified, while Week 10 is the "correct" complex version. I implemented a **Temporal Retriever** that maps chunks to session numbers, allowing the LLM to prioritize or compare explanations chronologically.





---

## 🚦 Getting Started

1. **Create a Qdrant Cloud cluster** (free tier) at [cloud.qdrant.io](https://cloud.qdrant.io), then copy `.env.example` to `.env` and fill in `QDRANT_URL`, `QDRANT_API_KEY` and `OPENAI_API_KEY`.
   The collection and its payload indexes are created automatically on first ingest.

   *Local-only alternative:* leave `QDRANT_URL` empty and run `docker run -p 6333:6333 qdrant/qdrant`.

2. **Install dependencies** (the `local` extra adds Whisper for on-device transcription):
   ```bash
   uv sync --extra local
   ```
   Skip `--extra local` and set `TRANSCRIBE_BACKEND=api` to transcribe through an API instead.

3. **Check System**:
   ```bash
   uv run python main.py
   ```

4. **Launch the UI**:
   ```bash
   uv run streamlit run ui/app.py
   ```

5. **Ingest your first lecture**:
   Go to the **"Add Lecture"** tab in the UI and paste a YouTube URL or upload an audio/video file!

---

## ☁️ Deployment (Streamlit Community Cloud)

1. Push the repo to GitHub and create an app at [share.streamlit.io](https://share.streamlit.io) with main file `ui/app.py`.
2. In **App settings → Secrets**, add your config as top-level keys (Streamlit exposes them as environment variables):
   ```toml
   OPENAI_API_KEY = "..."
   QDRANT_URL = "https://<cluster-id>.<region>.cloud.qdrant.io:6333"
   QDRANT_API_KEY = "..."

   # The host has ~1GB RAM and no GPU, so transcribe through an API
   TRANSCRIBE_BACKEND = "api"
   # Only needed if OPENAI_* points at a provider without an audio API (e.g. GitHub Models)
   TRANSCRIBE_API_KEY = "..."
   TRANSCRIBE_BASE_URL = "https://api.openai.com/v1"   # or https://api.groq.com/openai/v1
   TRANSCRIBE_MODEL = "whisper-1"                      # or "whisper-large-v3" on Groq
   ```
3. Dependencies: Community Cloud installs from `uv.lock` (it takes priority over `requirements.txt`). Whisper is an optional extra, so it isn't installed, and `torch` comes from the CPU-only index on Linux. If you change dependencies, edit `pyproject.toml` and run `uv lock`.
4. `packages.txt` installs `ffmpeg`, used to compress and split audio over the API's 25MB upload limit.

**How "Add Lecture" works when hosted**
- **YouTube URL** → existing captions are fetched first (`youtube-transcript-api`, no download). If there are none, audio is downloaded with `yt-dlp` and transcribed via the API.
- YouTube often blocks both from cloud IPs. When that happens the UI says so; either set `YOUTUBE_PROXY_URL` to a residential proxy, or download the lecture yourself and use **Upload file** (200MB limit).
- Ingesting from your laptop with `ingest.py` writes to the same Qdrant Cloud collection, so you can also ingest locally and only search from the hosted app.

> Free Qdrant Cloud clusters are suspended after a period of inactivity — resume them from the Qdrant console if `main.py` reports a connection failure.
