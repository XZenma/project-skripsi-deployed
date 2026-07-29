import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import faiss
import json
import numpy as np
from pathlib import Path
from langchain_ollama import OllamaEmbeddings
from config import EMBEDDING_MODEL, FAISS_PATHS, CHUNKS_PATH


def get_embeddings():
    return OllamaEmbeddings(model=EMBEDDING_MODEL)


def load_from_disk(teknik: str, folder_path: str = CHUNKS_PATH) -> tuple[list[str], list[dict]]:
    """
    Baca semua chunk dari data/chunks/<teknik>/<matkul>/*.json
    (output langsung dari chunker.py -- tanpa tahap chunk_deduplicator.py,
    yang tidak lagi digunakan pada pipeline ini).
    Return: (list teks chunk, list metadata+teks untuk docstore)
    """
    folder = Path(folder_path) / teknik
    if not folder.exists():
        raise FileNotFoundError(f"Folder tidak ditemukan: {folder}. Jalankan chunker.py dulu.")

    all_chunks = []
    docstore = []
    print(f"Membaca chunk dari {folder}...")

    for matkul_folder in folder.iterdir():
        if not matkul_folder.is_dir():
            continue
        for json_file in matkul_folder.glob("*.json"):
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            for c in data["chunks"]:
                all_chunks.append(c["text"])
                docstore.append({
                    "filename": data["filename"],
                    "matkul": data["matkul"],
                    "teknik": data["teknik"],
                    "index_chunk": c["index"],
                    "panjang_karakter": c["panjang_karakter"],
                    "text": c["text"]
                })
            print(f"  {matkul_folder.name}/{json_file.name}: {len(data['chunks'])} chunks")

    print(f"\nTotal chunks dibaca: {len(all_chunks)}")
    return all_chunks, docstore


def embed_chunks(chunks: list[str]) -> np.ndarray:
    embeddings = get_embeddings()
    print(f"Embedding {len(chunks)} chunks...")
    vectors = []
    for i, chunk in enumerate(chunks):
        vector = embeddings.embed_query(chunk)
        vectors.append(vector)
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(chunks)}")
    return np.array(vectors, dtype=np.float32)


def save_faiss(vectors: np.ndarray, docstore: list[dict], teknik: str):
    """
    Simpan vektor ke FAISS IndexFlatL2.
    Simpan docstore (teks + metadata) ke docstore.json.
    """
    save_path = FAISS_PATHS[teknik]
    os.makedirs(save_path, exist_ok=True)

    dimensi = vectors.shape[1]
    index = faiss.IndexFlatL2(dimensi)
    index.add(vectors)
    faiss.write_index(index, os.path.join(save_path, "index.faiss"))

    docstore_path = os.path.join(save_path, "docstore.json")
    with open(docstore_path, "w", encoding="utf-8") as f:
        json.dump(docstore, f, ensure_ascii=False, indent=2)

    print(f"FAISS index tersimpan di: {save_path}")
    print(f"Docstore tersimpan di: {docstore_path}")
    print(f"Total vektor tersimpan: {index.ntotal}")


def load_faiss(teknik: str) -> tuple:
    """
    Load FAISS index dan docstore.
    Return: (index, docstore)
    """
    load_path = FAISS_PATHS[teknik]

    index_path = os.path.join(load_path, "index.faiss")
    docstore_path = os.path.join(load_path, "docstore.json")

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index tidak ditemukan di: {load_path}. Jalankan embedder dulu.")

    index = faiss.read_index(index_path)

    with open(docstore_path, "r", encoding="utf-8") as f:
        docstore = json.load(f)

    print(f"FAISS index berhasil diload: {index.ntotal} vektor")
    print(f"Docstore berhasil diload: {len(docstore)} entri")
    return index, docstore


def build_index(teknik: str):
    all_chunks, docstore = load_from_disk(teknik)
    vectors = embed_chunks(all_chunks)
    save_faiss(vectors, docstore, teknik)
    return all_chunks


def ensure_input(teknik: str, folder_path: str = CHUNKS_PATH):
    """
    Pastikan folder input (hasil chunker.py) tersedia.
    Kalau tidak ada, jalankan chunker.py otomatis.
    """
    folder = Path(folder_path) / teknik
    if not folder.exists() or not any(folder.rglob("*.json")):
        print(f"⚠️  Folder '{folder}' tidak ditemukan atau kosong.")
        print("    → Menjalankan chunker.py otomatis...")
        from pipeline.chunker import ensure_input as chunker_ensure, load_from_disk as chunker_load, chunk_document
        chunker_ensure()
        docs = chunker_load()
        if docs:
            for doc in docs:
                chunk_document(
                    doc["text"], teknik,
                    filename=doc["filename"], matkul=doc["matkul"],
                    save=True
                )


if __name__ == "__main__":
    for teknik in ["fixed", "recursive", "sentence", "semantic"]:
        print(f"\n=== Membangun index untuk teknik: {teknik} ===")
        ensure_input(teknik)
        build_index(teknik)

    print(f"\n=== Test load semua index ===")
    for teknik in ["fixed", "recursive", "sentence", "semantic"]:
        index, docstore = load_faiss(teknik)
        print(f"{teknik}: {index.ntotal} vektor, {len(docstore)} entri docstore")