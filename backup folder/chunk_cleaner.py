import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hashlib
import json
from pathlib import Path
from config import CHUNKS_PATH, CHUNK_DEDUPLICATED_PATH


def get_chunk_hash(chunk: str) -> str:
    return hashlib.sha256(chunk.encode("utf-8")).hexdigest()


def load_from_disk(folder_path: str = CHUNKS_PATH, teknik: str = None) -> dict:
    """
    Baca file .json dari data/chunks/<teknik>/<matkul>/*.json
    Return dict: {teknik: [{filename, matkul, chunks: [str]}]}
    """
    folder = Path(folder_path)

    need_chunking = False
    if teknik:
        target_folder = folder / teknik
        if not target_folder.exists() or not any(target_folder.glob("**/*.json")):
            need_chunking = True
    else:
        if not folder.exists() or not any(folder.glob("**/*.json")):
            need_chunking = True

    if need_chunking:
        print(f"Data di {folder_path} untuk teknik {teknik or 'all'} tidak ditemukan atau kosong. Menjalankan chunker...")
        import pipeline.chunker as chunker
        docs = chunker.load_from_disk()
        if docs:
            tekniks = [teknik] if teknik else ["fixed", "recursive", "sentence", "semantic"]
            for t in tekniks:
                print(f"  Menjalankan teknik chunking: {t}")
                for doc in docs:
                    chunker.chunk_document(
                        doc["text"], t,
                        filename=doc["filename"], matkul=doc["matkul"],
                        save=True
                    )

    result = {}

    if folder.exists():
        teknik_folders = [f for f in folder.iterdir() if f.is_dir()] if not teknik else [folder / teknik]
    else:
        teknik_folders = []

    for teknik_folder in teknik_folders:
        if not teknik_folder.exists():
            print(f"Folder tidak ditemukan: {teknik_folder}")
            continue

        teknik_name = teknik_folder.name
        result[teknik_name] = []

        print(f"\nMembaca teknik: {teknik_name}")
        for matkul_folder in teknik_folder.iterdir():
            if not matkul_folder.is_dir():
                continue
            for json_file in matkul_folder.glob("*.json"):
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                chunks_text = [c["text"] for c in data["chunks"]]
                result[teknik_name].append({
                    "filename": data["filename"],
                    "matkul": data["matkul"],
                    "chunks": chunks_text
                })

    return result


def save_unique_chunks(matkul: str, filename: str, chunks: list[str], teknik: str, output_path: str = CHUNK_DEDUPLICATED_PATH):
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
            {
                "index": i + 1,
                "panjang_karakter": len(chunk),
                "hash_sha256": get_chunk_hash(chunk),
                "text": chunk
            }
            for i, chunk in enumerate(chunks)
        ]
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return save_path


def save_removed_chunks(removed_chunks: list[dict], teknik: str, output_path: str = CHUNK_DEDUPLICATED_PATH):
    removed_folder = Path(output_path) / "removed"
    removed_folder.mkdir(parents=True, exist_ok=True)
    save_path = removed_folder / f"{teknik}_removed.json"
    data = {
        "teknik": teknik,
        "total_chunk_dibuang": len(removed_chunks),
        "chunks_dibuang": removed_chunks
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  Laporan chunk dibuang tersimpan di: {save_path}")
    return save_path


def deduplicate_chunks(
    documents_meta: list[dict],
    teknik: str,
    save_output: bool = True,
    output_path: str = CHUNK_DEDUPLICATED_PATH
) -> tuple[list[str], list[dict]]:
    """
    Deduplikasi chunk byte-exact (SHA-256 murni tanpa normalisasi).
    Input: documents_meta dari load_from_disk()

    PERBAIKAN PENTING:
    Sebelumnya, penentuan chunk mana yang disimpan ke file per-dokumen
    dilakukan dengan mengecek "apakah hash chunk ini termasuk salah satu
    hash yang unik?" (get_chunk_hash(c) in unique_hashes). Karena semua
    salinan duplikat memiliki hash yang SAMA dengan versi aslinya, filter
    ini justru meloloskan SEMUA salinan duplikat ke file tersimpan --
    meskipun sudah tercatat sebagai "dibuang" di laporan removed_info.

    Sekarang, setiap posisi chunk (dokumen ke-i, index ke-j) yang menjadi
    KANONIK (kemunculan pertama dari suatu hash) dilacak secara eksplisit
    lewat `kept_positions`. Saat menyimpan file per dokumen, hanya posisi
    yang tercatat sebagai kanonik yang dipertahankan -- salinan duplikat
    di posisi lain (baik dalam dokumen yang sama maupun dokumen lain)
    benar-benar dibuang dari file yang disimpan, konsisten dengan laporan.
    """
    all_chunks = []
    chunk_origin = {}
    # index_map: idx_global -> (doc_index, chunk_index_dalam_dokumen)
    index_map = {}
    idx = 0
    for doc_i, doc_meta in enumerate(documents_meta):
        for chunk_i, chunk in enumerate(doc_meta["chunks"]):
            all_chunks.append(chunk)
            chunk_origin[idx] = {
                "filename": doc_meta["filename"],
                "matkul": doc_meta["matkul"]
            }
            index_map[idx] = (doc_i, chunk_i)
            idx += 1

    seen_hashes = {}
    unique_chunks = []
    removed_info = []
    kept_positions = set()  # {(doc_i, chunk_i), ...} posisi kanonik yang dipertahankan

    for i, chunk in enumerate(all_chunks):
        chunk_hash = get_chunk_hash(chunk)
        if chunk_hash not in seen_hashes:
            seen_hashes[chunk_hash] = {
                "index_unique": len(unique_chunks),
                "text": chunk,
                "origin": chunk_origin.get(i, {"filename": "unknown", "matkul": "unknown"})
            }
            unique_chunks.append(chunk)
            kept_positions.add(index_map[i])
        else:
            original = seen_hashes[chunk_hash]
            removed_info.append({
                "index_asli": i + 1,
                "panjang_karakter": len(chunk),
                "hash_sha256": chunk_hash,
                "chunk_dibuang": {
                    "filename": chunk_origin.get(i, {}).get("filename", "unknown"),
                    "matkul": chunk_origin.get(i, {}).get("matkul", "unknown"),
                    "preview": chunk[:200] + "..." if len(chunk) > 200 else chunk
                },
                "chunk_asli": {
                    "index_unique": original["index_unique"] + 1,
                    "filename": original["origin"].get("filename", "unknown"),
                    "matkul": original["origin"].get("matkul", "unknown"),
                    "preview": original["text"][:200] + "..." if len(original["text"]) > 200 else original["text"]
                }
            })

    print(f"\nTotal chunks sebelum dedup: {len(all_chunks)}")
    print(f"Chunk duplikat dibuang (byte-exact SHA-256): {len(removed_info)}")
    print(f"Total chunks setelah dedup: {len(unique_chunks)}")

    if save_output:
        save_removed_chunks(removed_info, teknik, output_path)

        print(f"\n  Menyimpan chunk unik per dokumen...")
        for doc_i, doc_meta in enumerate(documents_meta):
            # PERBAIKAN: filter berdasarkan posisi kanonik (doc_i, chunk_i),
            # bukan berdasarkan keberadaan hash di himpunan global.
            unique_doc_chunks = [
                chunk for chunk_i, chunk in enumerate(doc_meta["chunks"])
                if (doc_i, chunk_i) in kept_positions
            ]
            save_unique_chunks(
                matkul=doc_meta["matkul"],
                filename=doc_meta["filename"],
                chunks=unique_doc_chunks,
                teknik=teknik,
                output_path=output_path
            )

    return unique_chunks, removed_info


def ensure_input(folder_path: str = CHUNKS_PATH, teknik: str = None):
    """
    Pastikan folder input tersedia.
    Kalau tidak ada, jalankan chunker dulu secara otomatis.
    """
    folder = Path(folder_path)
    pattern = f"**/*.json"
    if not folder.exists() or not any(folder.rglob(pattern)):
        print(f"⚠️  Folder '{folder_path}' tidak ditemukan atau kosong.")
        print("    → Menjalankan chunker.py otomatis...")
        from pipeline.chunker import ensure_input as chunker_ensure, load_from_disk, chunk_document
        chunker_ensure()
        docs = load_from_disk()
        if docs:
            teknik_list = [teknik] if teknik else ["fixed", "recursive", "sentence", "semantic"]
            for t in teknik_list:
                for doc in docs:
                    chunk_document(doc["text"], t, filename=doc["filename"], matkul=doc["matkul"], save=True)


if __name__ == "__main__":
    for teknik in ["fixed", "recursive", "sentence", "semantic"]:
        print(f"\n=== Chunk deduplication untuk teknik: {teknik} ===")

        ensure_input(teknik=teknik)

        data = load_from_disk(teknik=teknik)
        documents_meta = data.get(teknik, [])

        if not documents_meta:
            print(f"  Tidak ada data untuk teknik {teknik}, skip.")
            continue

        unique_chunks, removed_info = deduplicate_chunks(
            documents_meta=documents_meta,
            teknik=teknik,
            save_output=True
        )

        print(f"\nHasil {teknik}:")
        print(f"  Unik   : {len(unique_chunks)} chunks")
        print(f"  Dibuang: {len(removed_info)} chunks")