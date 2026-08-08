import os
import sys
from pathlib import Path
from urllib.parse import quote

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pipeline.rag_chain import run_rag
from pipeline.embedder import load_faiss

# Folder root untuk mencari PDF sumber.
# Default: folder "data" di root project.
PDF_ROOT = Path(os.getenv("PDF_ROOT", "data/raw")).resolve()

app = FastAPI(
    title="RAG Academic API",
    description="API untuk sistem Retrieval-Augmented Generation chatbot akademik.",
    version="1.1.0"
)


# -------------------------
# Skema Data
# -------------------------
class RAGRequest(BaseModel):
    query: str
    teknik: str = "fixed"
    matkul: Optional[str] = None


class SumberDokumen(BaseModel):
    filename: Optional[str]
    matkul: Optional[str]
    index_chunk: Optional[int]
    panjang_karakter: Optional[int]
    distance: Optional[float]
    pdf_url: Optional[str] = None


class RAGResponse(BaseModel):
    query: str
    teknik: str
    matkul: Optional[str]
    jawaban: str
    konteks: List[str]
    sumber: List[SumberDokumen]
    generation_skipped: bool = False


# -------------------------
# Helper
# -------------------------
def find_pdf(filename: str, matkul: Optional[str] = None) -> Path:
    """
    Mencari PDF sumber berdasarkan filename.
    Jika matkul diberikan, pencarian diprioritaskan pada folder/nama
    yang mengandung nama mata kuliah.

    Metadata filename pada docstore mewarisi nama dari file hasil
    ekstraksi teks (.txt), sehingga ekstensinya perlu dinormalisasi
    ke .pdf sebelum pencarian dilakukan.
    """
    if not filename:
        raise FileNotFoundError("Nama file PDF kosong.")

    safe_name = Path(filename).name

    if safe_name.lower().endswith(".txt"):
        safe_name = safe_name[:-4] + ".pdf"

    if not PDF_ROOT.exists():
        raise FileNotFoundError(f"PDF_ROOT tidak ditemukan: {PDF_ROOT}")

    candidates = list(PDF_ROOT.rglob(safe_name))

    if matkul:
        matkul_lower = matkul.lower()
        filtered = [
            p for p in candidates
            if matkul_lower in str(p.parent).lower()
            or matkul_lower in p.stem.lower()
        ]
        if filtered:
            candidates = filtered

    if not candidates:
        raise FileNotFoundError(
            f"PDF '{safe_name}' tidak ditemukan di {PDF_ROOT}"
        )

    return candidates[0]


def make_pdf_url(filename: Optional[str], matkul: Optional[str]) -> Optional[str]:
    if not filename:
        return None

    params = f"filename={quote(filename)}"
    if matkul:
        params += f"&matkul={quote(matkul)}"

    # URL relatif terhadap API server.
    return f"/api/pdf?{params}"


# -------------------------
# Endpoints
# -------------------------
@app.post("/api/rag", response_model=RAGResponse)
async def process_rag_query(request: RAGRequest):
    """
    Endpoint utama RAG.
    Jika matkul diberikan, retrieval dibatasi pada mata kuliah tersebut.
    """
    try:
        hasil = run_rag(
            request.query,
            request.teknik,
            matkul=request.matkul
        )

        for sumber in hasil["sumber"]:
            sumber["pdf_url"] = make_pdf_url(
                sumber.get("filename"),
                sumber.get("matkul")
            )

        hasil["matkul"] = request.matkul
        return hasil

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/matkul")
async def get_matkul(teknik: str):
    """
    Mengambil daftar mata kuliah yang tersedia pada docstore
    untuk teknik chunking tertentu.
    """
    try:
        _, docstore = load_faiss(teknik)

        matkul = sorted({
            item.get("matkul")
            for item in docstore
            if item.get("matkul")
        })

        return {
            "teknik": teknik,
            "matkul": matkul
        }

    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Index untuk teknik '{teknik}' belum dibuat."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/pdf")
async def open_pdf(filename: str, matkul: Optional[str] = None):
    """
    Menampilkan PDF sumber secara inline di browser.
    """
    try:
        pdf_path = find_pdf(filename, matkul)

        return FileResponse(
            path=pdf_path,
            media_type="application/pdf",
            filename=pdf_path.name,
            headers={
                "Content-Disposition": f'inline; filename="{pdf_path.name}"'
            }
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/status")
async def check_index_status(teknik: str = "fixed"):
    """
    Mengecek apakah FAISS index untuk teknik tertentu tersedia.
    """
    try:
        index, docstore = load_faiss(teknik)

        return {
            "status": "ready",
            "vektor_total": index.ntotal,
            "chunk_total": len(docstore)
        }

    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Index untuk teknik '{teknik}' belum dibuat."
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=9001,
        reload=True
    )