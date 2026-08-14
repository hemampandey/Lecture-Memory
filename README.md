# Lecture Memory

**Lecture Memory** is an advanced RAG (Retrieval-Augmented Generation) system designed for students to manage high-volume video lectures. It doesn't just search; it tracks how concepts evolve over and detects if topic contradicts a previous lecture.

---

<img width="1190" height="394" alt="Screenshot 2026-05-11 at 2 36 17 AM" src="https://github.com/user-attachments/assets/0a5a5a72-e31d-4c04-bd62-b89ac758fe63" />

---

## Key Features

- **🎬 Source Ingestion**: Direct YouTube ingestion with auto-cleanup of local assets.
- **🎙️ Fast Transcription**: Transcription of lectures using `OpenAI Whisper`.
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
- **Transcription**: OpenAI Whisper (`tiny` model for speed)
- **Embeddings**: `sentence-transformers` (all-MiniLM-L6-v2)
- **LLM**: GPT-4o-mini

---

## 💡 Engineering Challenges I Solved

### 1. The "Temporal Context" Problem
Most RAG systems just find the most similar text. But in a course, a definition in Week 1 might be simplified, while Week 10 is the "correct" complex version. I implemented a **Temporal Retriever** that maps chunks to session numbers, allowing the LLM to prioritize or compare explanations chronologically.





---

## 🚦 Getting Started

1. **Start Qdrant**:
   ```bash
   docker run -p 6333:6333 qdrant/qdrant
   ```

2. **Check System**:
   ```bash
   python main.py
   ```

3. **Launch the UI**:
   ```bash
   uv run streamlit run ui/app.py
   ```

3. **Ingest your first lecture**:
   Go to the **"Add Lecture"** tab in the UI and paste a YouTube URL!
