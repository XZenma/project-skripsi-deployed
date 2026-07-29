import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from pipeline.rag_chain import run_rag
from pipeline.embedder import load_faiss

st.set_page_config(
    page_title="Chatbot Akademik RAG",
    page_icon="🎓",
    layout="wide"
)

# --- Sidebar ---
with st.sidebar:
    st.title("⚙️ Konfigurasi RAG")
    st.divider()

    st.subheader("Teknik Chunking")
    teknik = st.radio(
        label="Pilih teknik chunking:",
        options=["fixed", "recursive", "sentence", "semantic"],
        format_func=lambda x: {
            "fixed":     "📏 Fixed-size",
            "recursive": "🔁 Recursive",
            "sentence":  "📝 Sentence-aware",
            "semantic":  "🧠 Semantic",
        }[x],
        index=0
    )

    st.divider()

    # cek status FAISS index
    st.subheader("Status Index")
    try:
        index, docstore = load_faiss(teknik)
        st.success(f"✅ Index tersedia\n\n{index.ntotal} vektor | {len(docstore)} chunk")
    except FileNotFoundError:
        st.error(f"❌ Index '{teknik}' belum dibuat.\n\nJalankan:\n```\npython run_each.py\n```")
        docstore = []

    st.divider()

    st.subheader("Model")
    st.caption("🤖 LLM: Sailor2 via Ollama")
    st.caption("📐 Embedding: Qwen3-Embedding")
    st.caption("🗄️ Vector DB: FAISS IndexFlatL2")

    st.divider()

    if st.button("🗑️ Reset Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_teknik = teknik
        st.rerun()

# --- Reset otomatis kalau ganti teknik ---
if "last_teknik" not in st.session_state:
    st.session_state.last_teknik = teknik

if st.session_state.last_teknik != teknik:
    st.session_state.messages = []
    st.session_state.last_teknik = teknik
    st.rerun()

# --- Header ---
st.title("💬 Chatbot Akademik RAG")
st.caption(f"Teknik aktif: **{teknik}** | Model: Sailor2 | Embedding: Qwen3-Embedding")

# --- Inisialisasi history chat ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Halo! Saya chatbot akademik berbasis RAG. Silakan ajukan pertanyaan seputar materi kuliah yang sudah diindeks.",
            "konteks": [],
            "sumber": []
        }
    ]

# --- Tampilkan history chat ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

        if msg["role"] == "assistant" and msg.get("konteks"):
            with st.expander(f"📄 Lihat konteks ({len(msg['konteks'])} chunk)"):
                for i, (ctx, src) in enumerate(zip(msg["konteks"], msg.get("sumber", []))):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"**Chunk {i+1}** — `{src.get('filename', '-')}` ({src.get('matkul', '-')})")
                    with col2:
                        st.caption(f"Distance: {src.get('distance', '-')}")
                    st.text(ctx[:500] + "..." if len(ctx) > 500 else ctx)
                    if i < len(msg["konteks"]) - 1:
                        st.divider()

# --- Input pertanyaan ---
if prompt := st.chat_input("Tulis pertanyaan kamu tentang materi kuliah..."):

    # cek index tersedia
    try:
        load_faiss(teknik)
    except FileNotFoundError:
        st.error(f"Index untuk teknik '{teknik}' belum tersedia. Jalankan `python run_each.py` dulu.")
        st.stop()

    # tampilkan pesan user
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "konteks": [],
        "sumber": []
    })
    with st.chat_message("user"):
        st.write(prompt)

    # generate jawaban
    with st.chat_message("assistant"):
        with st.spinner("Mencari konteks dan menghasilkan jawaban..."):
            try:
                hasil = run_rag(prompt, teknik)

                st.write(hasil["jawaban"])

                if hasil["konteks"]:
                    with st.expander(f"📄 Lihat konteks ({len(hasil['konteks'])} chunk)"):
                        for i, (ctx, src) in enumerate(zip(hasil["konteks"], hasil["sumber"])):
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.markdown(f"**Chunk {i+1}** — `{src.get('filename', '-')}` ({src.get('matkul', '-')})")
                            with col2:
                                st.caption(f"Distance: {src.get('distance', '-')}")
                            st.text(ctx[:500] + "..." if len(ctx) > 500 else ctx)
                            if i < len(hasil["konteks"]) - 1:
                                st.divider()

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": hasil["jawaban"],
                    "konteks": hasil["konteks"],
                    "sumber": hasil["sumber"]
                })

            except Exception as e:
                pesan_error = f"Terjadi error: {str(e)}"
                st.error(pesan_error)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": pesan_error,
                    "konteks": [],
                    "sumber": []
                })