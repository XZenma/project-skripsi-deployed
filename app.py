import streamlit as st
import requests
from urllib.parse import quote
from html import escape


# ============================================================
# URL API
# ============================================================

API_BASE_URL = st.secrets["API_BASE_URL"]

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Chatbot Akademik RAG",
    page_icon="🎓",
    layout="wide"
)


# ============================================================
# HELPER UI
# ============================================================

def get_pdf_url(src: dict) -> str | None:
    """
    Mengubah URL relatif dari API menjadi URL lengkap.
    """
    pdf_url = src.get("pdf_url")

    if not pdf_url:
        filename = src.get("filename")
        matkul = src.get("matkul")

        if not filename:
            return None

        pdf_url = (
            f"/api/pdf?filename={quote(filename)}"
            f"&matkul={quote(matkul or '')}"
        )

    if pdf_url.startswith("http://") or pdf_url.startswith("https://"):
        return pdf_url

    return f"{API_BASE_URL}{pdf_url}"


# ============================================================
# TAMPILKAN DOKUMEN SUMBER
# ============================================================

def render_sources(konteks, sumber):
    """
    Menampilkan dokumen sumber dalam bentuk card kecil.
    Card langsung dapat diklik untuk membuka PDF.
    """

    if not sumber:
        return

    # Ambil dokumen unik
    documents = []
    seen = set()

    for src in sumber:
        filename = src.get("filename")
        matkul_src = src.get("matkul", "")

        if not filename:
            continue

        key = (filename, matkul_src)

        if key not in seen:
            seen.add(key)
            documents.append(src)

    if not documents:
        return

    st.markdown(
        """
        <div style="
            margin-top: 12px;
            margin-bottom: 7px;
            font-size: 13px;
            font-weight: 600;
        ">
            📚 Dokumen
        </div>
        """,
        unsafe_allow_html=True
    )

    cols = st.columns(3)

    for i, src in enumerate(documents):

        filename = src.get("filename", "Dokumen")
        pdf_url = get_pdf_url(src)

        if not pdf_url:
            continue

        safe_filename = escape(filename)
        safe_url = escape(pdf_url, quote=True)

        with cols[i % 3]:

            st.components.v1.html(
                f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <meta charset="UTF-8">

                    <style>

                        * {{
                            box-sizing: border-box;
                        }}

                        html, body {{
                            margin: 0;
                            padding: 0;
                            background: transparent;
                            font-family:
                                -apple-system,
                                BlinkMacSystemFont,
                                "Segoe UI",
                                sans-serif;
                        }}

                        .document-card {{
                            display: flex;
                            align-items: center;

                            width: 100%;
                            height: 62px;

                            padding: 8px 10px;

                            border: 1px solid #3A3A3A;
                            border-radius: 9px;

                            background-color: #262730;

                            text-decoration: none;
                            color: #FAFAFA;

                            cursor: pointer;

                            overflow: hidden;

                            transition:
                                border-color 0.2s ease,
                                box-shadow 0.2s ease;
                        }}

                        .document-card:hover {{
                            border-color: #666666;

                            box-shadow:
                                0 2px 7px
                                rgba(0, 0, 0, 0.18);
                        }}

                        .document-icon {{
                            flex-shrink: 0;

                            width: 34px;
                            height: 34px;

                            display: flex;
                            align-items: center;
                            justify-content: center;

                            font-size: 22px;

                            margin-right: 9px;
                        }}

                        .document-name {{
                            min-width: 0;

                            font-size: 12px;
                            font-weight: 600;
                            line-height: 1.3;

                            overflow: hidden;
                            text-overflow: ellipsis;
                            white-space: nowrap;
                        }}

                    </style>
                </head>

                <body>

                    <a
                        class="document-card"
                        href="{safe_url}"
                        target="_blank"
                        title="{safe_filename}"
                    >

                        <div class="document-icon">
                            📄
                        </div>

                        <div class="document-name">
                            {safe_filename}
                        </div>

                    </a>

                </body>
                </html>
                """,
                height=70,
                scrolling=False
            )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("⚙️ Konfigurasi RAG")
    st.divider()

    # Deployment menggunakan hasil teknik chunking terpilih.
    # Fixed-size. Tidak ada pilihan teknik lain pada UI.
    teknik = "fixed"

    # --------------------------------------------------------
    # Filter Mata Kuliah
    # --------------------------------------------------------

    st.subheader("🎓 Mata Kuliah")

    matkul_options = ["Pilih mata kuliah..."]
    matkul_ready = False

    try:

        res = requests.get(
            f"{API_BASE_URL}/api/matkul",
            params={"teknik": teknik},
            timeout=30
        )

        if res.status_code == 200:

            data_matkul = res.json()

            matkul_options += data_matkul.get(
                "matkul",
                []
            )

            matkul_ready = True

        else:

            st.error(
                "Daftar mata kuliah tidak dapat dimuat."
            )

    except requests.exceptions.ConnectionError:

        st.error(
            "❌ Gagal terhubung ke API Server."
        )

    except Exception as e:

        st.error(
            f"❌ Gagal memuat mata kuliah: {e}"
        )

    selected_matkul = st.selectbox(
        "Pilih mata kuliah:",
        options=matkul_options,
        index=0,
        disabled=not matkul_ready
    )

    matkul = (
        None
        if selected_matkul == "Pilih mata kuliah..."
        else selected_matkul
    )

    if matkul:

        st.success(
            f"Filter aktif: **{matkul}**"
        )

    else:

        st.info(
            "Pilih mata kuliah sebelum bertanya."
        )

    st.divider()

    # --------------------------------------------------------
    # Status Index
    # --------------------------------------------------------

    st.subheader("Status Index")

    try:

        res = requests.get(
            f"{API_BASE_URL}/api/status",
            params={"teknik": teknik},
            timeout=10
        )

        if res.status_code == 200:

            data = res.json()

            st.success(
                f"✅ Index tersedia\n\n"
                f"{data['vektor_total']} vektor | "
                f"{data['chunk_total']} chunk"
            )

            index_ready = True

        else:

            st.error(
                f"❌ Index '{teknik}' belum dibuat.\n\n"
                "Jalankan proses embedding terlebih dahulu."
            )

            index_ready = False

    except requests.exceptions.ConnectionError:

        st.error(
            "❌ Gagal terhubung ke API Server. "
            "Pastikan `api.py` sedang berjalan."
        )

        index_ready = False

    except Exception as e:

        st.error(
            f"❌ Gagal memeriksa status index: {e}"
        )

        index_ready = False

    st.divider()

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    st.subheader("Model")

    st.caption(
        "🤖 LLM: Sailor2 via Ollama"
    )

    st.caption(
        "📐 Embedding: Qwen3-Embedding"
    )

    st.caption(
        "🗄️ Vector DB: FAISS IndexFlatL2"
    )

    st.divider()

    # --------------------------------------------------------
    # Reset Chat
    # --------------------------------------------------------

    if st.button(
        "🗑️ Reset Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.session_state.last_matkul = matkul

        st.rerun()


# ============================================================
# RESET CHAT JIKA MATA KULIAH BERUBAH
# ============================================================

if "last_matkul" not in st.session_state:

    st.session_state.last_matkul = matkul


if st.session_state.last_matkul != matkul:

    st.session_state.messages = []

    st.session_state.last_matkul = matkul

    st.rerun()


# ============================================================
# HEADER
# ============================================================

st.title("💬 Chatbot Akademik RAG")


if matkul:

    st.caption(
        f"Mata kuliah: **{matkul}** | "
        "Model: Sailor2 | "
        "Embedding: Qwen3-Embedding"
    )

else:

    st.caption(
        f"Pilih mata kuliah untuk memulai | "   
        "Model: Sailor2 | "
        "Embedding: Qwen3-Embedding"
    )


# ============================================================
# HISTORY
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Halo! Saya chatbot akademik berbasis RAG. "
                "Silakan pilih mata kuliah pada sidebar, "
                "kemudian ajukan pertanyaan."
            ),
            "konteks": [],
            "sumber": []
        }
    ]


for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        if msg["role"] == "assistant" and msg.get("generation_skipped"):
            st.warning(msg["content"], icon="⚠️")
        else:
            st.write(msg["content"])

        # ----------------------------------------------------
        # Dokumen hanya muncul pada pesan assistant
        # yang memiliki sumber.
        # ----------------------------------------------------

        if (
            msg["role"] == "assistant"
            and msg.get("konteks")
        ):

            render_sources(
                msg["konteks"],
                msg.get("sumber", [])
            )


# ============================================================
# INPUT PERTANYAAN
# ============================================================

if prompt := st.chat_input(
    "Tulis pertanyaan tentang materi kuliah..."
):

    # --------------------------------------------------------
    # Validasi mata kuliah
    # --------------------------------------------------------

    if not matkul:

        st.warning(
            "Silakan pilih mata kuliah terlebih dahulu."
        )

        st.stop()

    # --------------------------------------------------------
    # Validasi index
    # --------------------------------------------------------

    if not index_ready:

        st.error(
            f"Index untuk teknik '{teknik}' "
            "belum tersedia atau API Server mati."
        )

        st.stop()

    # --------------------------------------------------------
    # Pesan user
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
            "konteks": [],
            "sumber": []
        }
    )

    with st.chat_message("user"):

        st.write(prompt)

    # --------------------------------------------------------
    # RAG
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "Mencari chunk pada mata kuliah terpilih..."
        ):

            try:

                payload = {
                    "query": prompt,
                    "teknik": teknik,
                    "matkul": matkul
                }

                response = requests.post(
                    f"{API_BASE_URL}/api/rag",
                    json=payload,
                    timeout=300
                )

                if response.status_code == 200:

                    hasil = response.json()
                    generation_skipped = hasil.get("generation_skipped", False)

                    # ------------------------------------------------
                    # Jawaban
                    # ------------------------------------------------

                    if generation_skipped:
                        st.warning(hasil["jawaban"], icon="⚠️")
                    else:
                        st.write(hasil["jawaban"])

                    # ------------------------------------------------
                    # Dokumen sumber
                    # ------------------------------------------------

                    render_sources(
                        hasil["konteks"],
                        hasil["sumber"]
                    )

                    # ------------------------------------------------
                    # Simpan history
                    # ------------------------------------------------

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": hasil["jawaban"],
                            "konteks": hasil["konteks"],
                            "sumber": hasil["sumber"],
                            "generation_skipped": generation_skipped
                        }
                    )

                else:

                    try:

                        detail = response.json().get(
                            "detail",
                            "Unknown error"
                        )

                    except Exception:

                        detail = response.text

                    st.error(
                        f"Error dari API: {detail}"
                    )

            except requests.exceptions.ConnectionError:

                st.error(
                    "Gagal terhubung ke API. "
                    "Pastikan API berjalan "
                    "di http://localhost:9001"
                )

            except requests.exceptions.Timeout:

                st.error(
                    "Request terlalu lama. "
                    "Periksa API, Ollama, dan model "
                    "yang digunakan."
                )

            except Exception as e:

                st.error(
                    f"Terjadi error tak terduga: {str(e)}"
                )