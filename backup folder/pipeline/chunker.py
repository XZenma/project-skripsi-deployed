import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import spacy
from pathlib import Path
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings
from config import CHUNK_SIZE, CHUNK_OVERLAP, SEMANTIC_THRESHOLD, EMBEDDING_MODEL, SPACY_MODEL, CHUNKS_PATH, DEDUPLICATED_PATH


def fixed_chunking(text: str) -> list[str]:
    splitter = CharacterTextSplitter(
        separator="", chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP, length_function=len,
    )
    return splitter.split_text(text)


def recursive_chunking(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        length_function=len, separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_text(text)


def sentence_chunking(text: str) -> list[str]:
    nlp = spacy.load(SPACY_MODEL, exclude=["parser", "ner", "lemmatizer", "tagger"])
    nlp.max_length = len(text) + 1000
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= CHUNK_SIZE:
            current_chunk += " " + sentence if current_chunk else sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks


def semantic_chunking(text: str) -> list[str]:
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=SEMANTIC_THRESHOLD,
    )
    return splitter.split_text(text)


def save_chunks_to_file(matkul: str, filename: str, chunks: list[str], teknik: str, output_path: str = CHUNKS_PATH):
    save_folder = Path(output_path) / teknik / matkul
    save_folder.mkdir(parents=True, exist_ok=True)
    json_filename = Path(filename).stem + ".json"
    save_path = save_folder / json_filename
    data = {
        "filename": filename,
        "matkul": matkul,
        "teknik": teknik,
        "total_chunks": len(chunks),
        "chunks": [
            {"index": i + 1, "panjang_karakter": len(chunk), "text": chunk}
            for i, chunk in enumerate(chunks)
        ]
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return save_path


def chunk_document(text: str, teknik: str, filename: str = None, matkul: str = None, save: bool = False) -> list[str]:
    teknik_map = {
        "fixed": fixed_chunking,
        "recursive": recursive_chunking,
        "sentence": sentence_chunking,
        "semantic": semantic_chunking,
    }
    if teknik not in teknik_map:
        raise ValueError(f"Teknik tidak dikenal: {teknik}. Pilih: {list(teknik_map.keys())}")
    print(f"Menjalankan {teknik} chunking...")
    chunks = teknik_map[teknik](text)
    print(f"Total chunks: {len(chunks)}")
    if save:
        if not filename or not matkul:
            raise ValueError("filename dan matkul wajib diisi jika save=True")
        saved_path = save_chunks_to_file(matkul, filename, chunks, teknik)
        print(f"Chunks tersimpan: {saved_path}")
    return chunks


def load_from_disk(folder_path: str = DEDUPLICATED_PATH) -> list[dict]:
    """
    Baca semua file .txt dari data/deduplicated/<matkul>/*.txt
    """
    folder = Path(folder_path)
    if not folder.exists() or not any(folder.glob("**/*.txt")):
        print(f"Data di {folder_path} tidak ditemukan atau kosong. Menjalankan deduplicator...")
        import pipeline.deduplicator as deduplicator
        cleaned_docs = deduplicator.load_from_disk()
        if cleaned_docs:
            deduplicator.deduplicate_documents(cleaned_docs, save_output=True)

    documents = []

    if folder.exists():
        matkul_folders = [f for f in folder.iterdir() if f.is_dir()]
    else:
        matkul_folders = []
    print(f"Membaca dari {folder_path} — {len(matkul_folders)} matkul ditemukan")

    for matkul_folder in matkul_folders:
        txt_files = list(matkul_folder.glob("*.txt"))
        if not txt_files:
            continue
        print(f"\nMatkul: {matkul_folder.name} ({len(txt_files)} file)")
        for txt_file in txt_files:
            text = txt_file.read_text(encoding="utf-8")
            documents.append({
                "filename": txt_file.name,
                "matkul": matkul_folder.name,
                "text": text
            })

    print(f"\nTotal dokumen dibaca: {len(documents)}")
    return documents

def ensure_input(folder_path: str = DEDUPLICATED_PATH):
    """
    Pastikan folder input tersedia.
    Kalau tidak ada, jalankan deduplicator dulu secara otomatis.
    """
    if not Path(folder_path).exists() or not any(Path(folder_path).rglob("*.txt")):
        print(f"⚠️  Folder '{folder_path}' tidak ditemukan atau kosong.")
        print("    → Menjalankan deduplicator.py otomatis...")
        from pipeline.deduplicator import ensure_input as dedup_ensure, load_from_disk, deduplicate_documents
        dedup_ensure()
        docs = load_from_disk()
        if docs:
            deduplicate_documents(docs)

if __name__ == "__main__":
    ensure_input()
    docs = load_from_disk()
    if docs:
        for teknik in ["fixed", "recursive", "sentence", "semantic"]:
            print(f"\n=== Menjalankan teknik: {teknik} ===")
            total_chunks_teknik = 0
            for doc in docs:
                chunks = chunk_document(
                    doc["text"], teknik,
                    filename=doc["filename"], matkul=doc["matkul"],
                    save=True
                )
                total_chunks_teknik += len(chunks)
            print(f"\n{teknik}: total {total_chunks_teknik} chunks dari {len(docs)} dokumen\n")