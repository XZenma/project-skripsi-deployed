import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

# Tambahkan path jika diperlukan
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pipeline.rag_chain import run_rag
from pipeline.embedder import load_faiss

app = FastAPI(
    title="RAG Academic API",
    description="API untuk sistem Retrieval-Augmented Generation chatbot akademik.",
    version="1.0.0"
)

# --- Skema Data (Pydantic Models) ---
class RAGRequest(BaseModel):
    query: str
    teknik: str

class SumberDokumen(BaseModel):
    filename: Optional[str]
    matkul: Optional[str]
    index_chunk: Optional[int]
    panjang_karakter: Optional[int]
    distance: Optional[float]

class RAGResponse(BaseModel):
    query: str
    teknik: str
    jawaban: str
    konteks: List[str]
    sumber: List[SumberDokumen]

# --- Endpoints ---
@app.post("/api/rag", response_model=RAGResponse)
async def process_rag_query(request: RAGRequest):
    """
    Endpoint utama untuk menjalankan RAG pipeline.
    """
    try:
        hasil = run_rag(request.query, request.teknik)
        return hasil
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status")
async def check_index_status(teknik: str):
    """
    Endpoint untuk mengecek apakah FAISS index untuk teknik tertentu sudah tersedia.
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
    # Jalankan server di port 8000
    uvicorn.run("api:app", host="0.0.0.0", port=9001, reload=True)