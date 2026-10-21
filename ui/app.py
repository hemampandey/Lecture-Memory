import streamlit as st
import sys
from pathlib import Path
sys.path.append(".")

from config import config
from retrieval.answer_generator import answer, concept_timeline

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Lecture Memory | AI Learning Assistant",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 Lecture Memory")
st.caption("Query across all your lectures. Find contradictions. Track how concepts evolved.")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Search Settings")
    check_contradictions = st.toggle("Detect contradictions", value=True)
    use_session_filter = st.toggle("Filter by sessions range", value=False)

    session_from, session_to = None, None
    if use_session_filter:
        col1, col2 = st.columns(2)
        session_from = col1.number_input("From session", min_value=1, max_value=20, value=1)
        session_to = col2.number_input("To session", min_value=1, max_value=20, value=10)

    st.divider()
    st.header("📊 Concept Timeline")
    timeline_concept = st.text_input("Track concept:", placeholder="e.g. attention mechanism")
    show_timeline = st.button("Show Timeline")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔍 Search & Ask", "📤 Add Lecture"])

with tab1:
    # ── Main query area ───────────────────────────────────────────────────────────
    query = st.text_input(
        "Ask a question about your lectures:",
        placeholder="How was backpropagation explained? Did it change over the course?",
    )
    search_btn = st.button("Search", type="primary")

    # ── Answer ────────────────────────────────────────────────────────────────────
    if search_btn and query:
        with st.spinner("Searching lectures..."):
            result = answer(
                query=query,
                session_from=session_from,
                session_to=session_to,
                check_for_contradictions=check_contradictions,
            )

        st.subheader("💬 Answer")
        st.write(result.answer)

        # Contradiction alert
        if result.contradiction_found:
            st.error(
                f"⚠️ **Contradiction detected across sessions**\n\n"
                f"{result.contradiction_explanation}"
            )
        elif check_contradictions:
            st.success("✅ No contradictions detected across sessions.")

        # Citations
        st.subheader("📍 Sources")
        for i, source in enumerate(result.sources, 1):
            with st.expander(
                f"Source {i} — Session {source.session_number} | {source.timestamp_label}",
                expanded=(i == 1),
            ):
                st.write(source.excerpt)

                # The demo-maker: timestamp deep link
                if source.video_url:
                    st.markdown(
                        f"[▶️ Jump to this moment in the lecture]({source.video_url})",
                        unsafe_allow_html=True,
                    )
                else:
                    st.caption(f"Session: {source.session_id} | {source.timestamp_label}")

    # ── Concept Timeline ──────────────────────────────────────────────────────────
    if show_timeline and timeline_concept:
        with st.spinner(f"Building timeline for '{timeline_concept}'..."):
            timeline = concept_timeline(timeline_concept)

        if timeline:
            st.subheader(f"📈 How '{timeline_concept}' evolved across lectures")

            for entry in timeline:
                col1, col2 = st.columns([1, 4])
                col1.metric("Session", entry["session_number"])
                with col2:
                    st.write(entry["summary"])
                    if entry.get("timestamp_url"):
                        st.markdown(f"[▶️ {entry['timestamp_label']}]({entry['timestamp_url']})")
                st.divider()
        else:
            st.warning(f"No content found for '{timeline_concept}'.")

with tab2:
    st.header("📥 Ingest New Lecture")
    transcriber = (
        f"local Whisper (`{config.WHISPER_MODEL}`)" if config.TRANSCRIBE_BACKEND == "local"
        else f"`{config.TRANSCRIBE_MODEL}` via API"
    )
    captions_note = "YouTube captions are used when available; otherwise " if config.YOUTUBE_CAPTIONS else ""
    st.caption(f"{captions_note}audio is transcribed with {transcriber}.")

    source_type = st.radio("Source", ["YouTube URL", "Upload file"], horizontal=True)

    with st.form("ingest_form"):
        yt_url, uploaded = None, None
        if source_type == "YouTube URL":
            yt_url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
        else:
            uploaded = st.file_uploader(
                "Lecture audio or video",
                type=["mp3", "m4a", "wav", "ogg", "oga", "flac", "webm", "mp4", "mov", "mkv", "mpeg", "mpga"],
            )
        col1, col2 = st.columns(2)
        s_num = col1.number_input("Session Number", min_value=1, value=1)
        s_label = col2.text_input("Session Label", placeholder="e.g. Intro to Neural Networks")
        use_llm_topics = st.checkbox(
            "Tag topics with the LLM (one call per chunk — slower)", value=config.USE_LLM_TOPICS
        )

        ingest_btn = st.form_submit_button("🚀 Start Ingestion")

    if ingest_btn:
        if source_type == "YouTube URL" and not yt_url:
            st.error("Please provide a YouTube URL.")
        elif source_type == "Upload file" and uploaded is None:
            st.error("Please choose a file to upload.")
        else:
            from ingest import ingest_youtube, ingest_local, IngestionError

            with st.status("Processing lecture...", expanded=True) as status:
                def report(message: str):
                    st.write(message)
                    status.update(label=message)

                upload_path, result, failure = None, None, None
                try:
                    if yt_url:
                        result = ingest_youtube(
                            yt_url.strip(), s_num, s_label, whisper_model=config.WHISPER_MODEL,
                            use_llm_topics=use_llm_topics, progress=report,
                        )
                    else:
                        # Transcription works on a file path; the upload only lives in memory
                        upload_path = config.RAW_DATA_DIR / f"upload_session_{s_num:02d}{Path(uploaded.name).suffix.lower()}"
                        upload_path.write_bytes(uploaded.getbuffer())
                        result = ingest_local(
                            str(upload_path), s_num, s_label or Path(uploaded.name).stem,
                            whisper_model=config.WHISPER_MODEL, use_llm_topics=use_llm_topics, progress=report,
                        )
                    status.update(label="✅ Ingestion Complete!", state="complete", expanded=False)
                except IngestionError as e:
                    status.update(label="❌ Ingestion Failed", state="error", expanded=False)
                    failure = str(e)
                except Exception as e:
                    status.update(label="❌ Ingestion Failed", state="error", expanded=False)
                    failure = f"Error: {str(e)}"
                finally:
                    if upload_path and upload_path.exists():
                        upload_path.unlink()

            # Shown outside the status box so it stays visible once the box collapses
            if failure:
                st.error(failure)
            else:
                source_note = {"captions": "YouTube captions", "audio": "transcribed audio", "file": "the uploaded file"}[result["source"]]
                st.success(
                    f"Lecture '{result['label']}' (Session {s_num}) is now searchable — "
                    f"{result['chunks']} chunk(s) from {source_note}."
                )

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Built with Whisper · sentence-transformers · Qdrant · GPT-4o · Streamlit")
