import streamlit as st
import requests

# URL API (Sesuaikan jika di-deploy ke server production)
API_BASE_URL = "http://localhost:9001"

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

    # Cek status FAISS index via API
    st.subheader("Status Index")
    try:
        res = requests.get(f"{API_BASE_URL}/api/status", params={"teknik": teknik})
        if res.status_code == 200:
            data = res.json()
            st.success(f"✅ Index tersedia\n\n{data['vektor_total']} vektor | {data['chunk_total']} chunk")
            index_ready = True
        else:
            st.error(f"❌ Index '{teknik}' belum dibuat.\n\nJalankan:\n```\npython run_each.py\n```")
            index_ready = False
    except requests.exceptions.ConnectionError:
        st.error("❌ Gagal terhubung ke API Server. Pastikan `api.py` sedang berjalan.")
        index_ready = False

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

    if not index_ready:
        st.error(f"Index untuk teknik '{teknik}' belum tersedia atau API Server mati.")
        st.stop()

    # Tampilkan pesan user
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "konteks": [],
        "sumber": []
    })
    with st.chat_message("user"):
        st.write(prompt)

    # Generate jawaban via API
    with st.chat_message("assistant"):
        with st.spinner("Mencari konteks dan menghasilkan jawaban via API..."):
            try:
                # Memanggil API Server
                payload = {"query": prompt, "teknik": teknik}
                response = requests.post(f"{API_BASE_URL}/api/rag", json=payload)
                
                if response.status_code == 200:
                    hasil = response.json()
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

                    # Simpan ke memori chat
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": hasil["jawaban"],
                        "konteks": hasil["konteks"],
                        "sumber": hasil["sumber"]
                    })
                else:
                    pesan_error = f"Error dari API: {response.json().get('detail', 'Unknown error')}"
                    st.error(pesan_error)

            except requests.exceptions.ConnectionError:
                st.error("Gagal terhubung ke API. Pastikan API berjalan di http://localhost:8000")
            except Exception as e:
                pesan_error = f"Terjadi error tak terduga: {str(e)}"
                st.error(pesan_error)