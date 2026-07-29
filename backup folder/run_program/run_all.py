import os
import sys

# Memastikan direktori root masuk ke sys.path agar config.py bisa diakses
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import semua modul pipeline
import pipeline.loader as loader
import pipeline.cleaner as cleaner
import pipeline.deduplicator as deduplicator
import pipeline.chunker as chunker
import pipeline.chunk_cleaner as chunk_cleaner
import pipeline.embedder as embedder

def main():
    print("=========================================")
    print("       MEMULAI RAG PIPELINE LENGKAP      ")
    print("=========================================")

    # ---------------------------------------------------------
    # TAHAP 1: Loader (Ekstrak teks dari PDF)
    # ---------------------------------------------------------
    print("\n>>> TAHAP 1: LOADING PDF")
    raw_docs = loader.load_documents()
    if not raw_docs:
        print("Tidak ada dokumen PDF yang diproses. Pipeline dihentikan.")
        return

    # ---------------------------------------------------------
    # TAHAP 2: Cleaner (Membersihkan teks mentah)
    # ---------------------------------------------------------
    print("\n>>> TAHAP 2: CLEANING TEXT")
    loaded_docs = cleaner.load_from_disk()
    if loaded_docs:
        cleaner.clean_documents(loaded_docs, save_output=True)

    # ---------------------------------------------------------
    # TAHAP 3: Document Deduplication (Hapus dokumen identik)
    # ---------------------------------------------------------
    print("\n>>> TAHAP 3: DEDUPLIKASI DOKUMEN")
    cleaned_docs = deduplicator.load_from_disk()
    if cleaned_docs:
        deduplicator.deduplicate_documents(cleaned_docs, save_output=True)

    # ---------------------------------------------------------
    # TAHAP 4: Chunker (Memecah teks dengan 4 teknik)
    # ---------------------------------------------------------
    print("\n>>> TAHAP 4: CHUNKING DOKUMEN")
    docs_to_chunk = chunker.load_from_disk()
    teknik_list = ["fixed", "recursive", "sentence", "semantic"]

    if docs_to_chunk:
        for teknik in teknik_list:
            print(f"\n--- Menjalankan teknik chunking: {teknik} ---")
            for doc in docs_to_chunk:
                chunker.chunk_document(
                    text=doc["text"], 
                    teknik=teknik,
                    filename=doc["filename"], 
                    matkul=doc["matkul"],
                    save=True
                )

    # ---------------------------------------------------------
    # TAHAP 5: Chunk Deduplication (Hapus duplikasi pada level chunk)
    # ---------------------------------------------------------
    print("\n>>> TAHAP 5: DEDUPLIKASI CHUNK")
    for teknik in teknik_list:
        print(f"\n--- Deduplikasi chunk untuk teknik: {teknik} ---")
        data_chunks = chunk_cleaner.load_from_disk(teknik=teknik)
        documents_meta = data_chunks.get(teknik, [])

        if documents_meta:
            chunk_cleaner.deduplicate_chunks(
                documents_meta=documents_meta,
                teknik=teknik,
                save_output=True
            )

    # ---------------------------------------------------------
    # TAHAP 6: Embedder (Membuat FAISS Index)
    # ---------------------------------------------------------
    print("\n>>> TAHAP 6: EMBEDDING & PEMBUATAN FAISS INDEX")
    for teknik in teknik_list:
        print(f"\n--- Membangun index untuk teknik: {teknik} ---")
        try:
            embedder.build_index(teknik)
        except Exception as e:
            print(f"Gagal membangun index untuk {teknik}: {e}")

    print("\n=========================================")
    print("       PIPELINE SELESAI DIEKSEKUSI       ")
    print("=========================================")

if __name__ == "__main__":
    main()