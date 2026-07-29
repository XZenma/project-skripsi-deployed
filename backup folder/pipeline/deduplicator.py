import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hashlib
import json
from pathlib import Path
from config import CLEANED_PATH, DEDUPLICATED_PATH, REMOVED_DUPLICATES_PATH


def get_text_hash(text: str) -> str:
    normalized = " ".join(text.lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def load_from_disk(folder_path: str = CLEANED_PATH) -> list[dict]:
    """
    Baca semua file .txt dari data/cleaned/<matkul>/*.txt
    """
    folder = Path(folder_path)
    if not folder.exists() or not any(folder.glob("**/*.txt")):
        print(f"Data di {folder_path} tidak ditemukan atau kosong. Menjalankan cleaner...")
        import pipeline.cleaner as cleaner
        loaded_docs = cleaner.load_from_disk()
        if loaded_docs:
            cleaner.clean_documents(loaded_docs, save_output=True)

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


def exact_deduplicate(documents: list[dict]) -> tuple[list[dict], list[dict]]:
    seen_hashes = {}
    unique_documents = []
    removed_documents = []

    for doc in documents:
        text_hash = get_text_hash(doc["text"])
        if text_hash not in seen_hashes:
            seen_hashes[text_hash] = doc
            unique_documents.append(doc)
        else:
            original = seen_hashes[text_hash]
            doc["_duplicate_type"] = "exact"
            doc["_duplicate_of"] = original["filename"]
            doc["_duplicate_of_matkul"] = original["matkul"]
            doc["_similarity_score"] = 1.0
            removed_documents.append(doc)
            print(f"  [EXACT DUPLICATE] '{doc['filename']}' ({doc['matkul']}) "
                  f"identik dengan '{original['filename']}' ({original['matkul']})")

    return unique_documents, removed_documents


def save_text_file(matkul: str, filename: str, text: str, output_path: str):
    matkul_folder = Path(output_path) / matkul
    matkul_folder.mkdir(parents=True, exist_ok=True)
    save_path = matkul_folder / filename
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(text)
    return save_path


def save_dedup_results(unique_documents: list[dict], removed_documents: list[dict]):
    print("\n--- Menyimpan dokumen yang lolos dedup ---")
    for doc in unique_documents:
        save_text_file(doc["matkul"], doc["filename"], doc["text"], DEDUPLICATED_PATH)

    print("--- Menyimpan dokumen yang dibuang ---")
    for doc in removed_documents:
        save_text_file(doc["matkul"], doc["filename"], doc["text"], REMOVED_DUPLICATES_PATH)

    report = {
        "total_dokumen_lolos": len(unique_documents),
        "total_dokumen_dibuang": len(removed_documents),
        "metode": "Exact deduplication via SHA-256 hash (normalized)",
        "detail_dibuang": [
            {
                "dokumen_dibuang": {
                    "filename": doc["filename"],
                    "matkul": doc["matkul"],
                    "panjang_karakter": len(doc["text"]),
                    "preview": doc["text"][:200] + "..." if len(doc["text"]) > 200 else doc["text"]
                },
                "dokumen_asli": {
                    "filename": doc.get("_duplicate_of"),
                    "matkul": doc.get("_duplicate_of_matkul"),
                },
                "tipe_duplikat": doc.get("_duplicate_type"),
                "similarity_score": doc.get("_similarity_score"),
            }
            for doc in removed_documents
        ]
    }

    report_path = Path(REMOVED_DUPLICATES_PATH) / "report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\nLaporan dedup tersimpan di: {report_path}")


def deduplicate_documents(documents: list[dict], save_output: bool = True) -> list[dict]:
    print(f"Total dokumen sebelum dedup: {len(documents)}")
    print("\n--- Exact Deduplication ---")
    documents, exact_removed = exact_deduplicate(documents)
    print(f"Dihapus (exact duplicate): {len(exact_removed)}")
    print(f"Total dokumen final: {len(documents)}")
    if save_output:
        save_dedup_results(documents, exact_removed)
    return documents

def ensure_input(folder_path: str = CLEANED_PATH):
    """
    Pastikan folder input tersedia.
    Kalau tidak ada, jalankan cleaner dulu secara otomatis.
    """
    if not Path(folder_path).exists() or not any(Path(folder_path).rglob("*.txt")):
        print(f"⚠️  Folder '{folder_path}' tidak ditemukan atau kosong.")
        print("    → Menjalankan cleaner.py otomatis...")
        from pipeline.cleaner import ensure_input as cleaner_ensure, load_from_disk, clean_documents
        cleaner_ensure()
        docs = load_from_disk()
        if docs:
            clean_documents(docs)

if __name__ == "__main__":
    ensure_input()
    docs = load_from_disk()
    if docs:
        deduplicate_documents(docs)