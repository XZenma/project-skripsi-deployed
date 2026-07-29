import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pipeline.loader as loader
import pipeline.cleaner as cleaner
import pipeline.deduplicator as deduplicator
import pipeline.chunker as chunker
import pipeline.embedder as embedder

OUTPUT_DIR = Path("run_program/data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEKNIK_LIST = ["fixed", "recursive", "sentence", "semantic"]


def run_tahap(nama: str, fn):
    """
    Jalankan satu tahap pipeline, ukur waktu eksekusinya.
    Return (hasil, durasi_detik)
    """
    print(f"\n  → {nama}...")
    start = time.perf_counter()
    hasil = fn()
    durasi = time.perf_counter() - start
    print(f"  ✓ {nama} selesai ({durasi:.2f}s)")
    return hasil, durasi


def main():
    print("=========================================")
    print("   RAG PIPELINE — RUN PER TEKNIK CHUNKING")
    print("=========================================")
    print(f"Waktu mulai: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    hasil_semua = {
        "timestamp": datetime.now().isoformat(),
        "tahap_bersama": {},
        "per_teknik": {}
    }

    # ---------------------------------------------------------
    # TAHAP BERSAMA (dijalankan sekali, sebelum loop teknik)
    # ---------------------------------------------------------
    print("\n========== TAHAP BERSAMA ==========")

    # Tahap 1: Loader
    print("\n[1/3] LOADING PDF")
    raw_docs, t = run_tahap("Ekstraksi PDF", lambda: loader.load_documents())
    hasil_semua["tahap_bersama"]["loader"] = {
        "total_dokumen": len(raw_docs) if raw_docs else 0,
        "durasi_detik": round(t, 4)
    }
    if not raw_docs:
        print("Tidak ada dokumen. Pipeline dihentikan.")
        return

    # Tahap 2: Cleaner
    print("\n[2/3] CLEANING TEXT")
    loaded_docs = cleaner.load_from_disk()
    cleaned_docs, t = run_tahap("Cleaning teks", lambda: cleaner.clean_documents(loaded_docs, save_output=True))
    hasil_semua["tahap_bersama"]["cleaner"] = {
        "total_dokumen": len(cleaned_docs) if cleaned_docs else 0,
        "durasi_detik": round(t, 4)
    }

    # Tahap 3: Deduplicator (Document Level)
    print("\n[3/3] DEDUPLIKASI DOKUMEN")
    cleaned_disk = deduplicator.load_from_disk()
    dedup_docs, t = run_tahap("Dedup dokumen", lambda: deduplicator.deduplicate_documents(cleaned_disk, save_output=True))
    hasil_semua["tahap_bersama"]["deduplicator"] = {
        "total_dokumen_sebelum": len(cleaned_disk) if cleaned_disk else 0,
        "total_dokumen_sesudah": len(dedup_docs) if dedup_docs else 0,
        "total_dibuang": (len(cleaned_disk) - len(dedup_docs)) if (cleaned_disk and dedup_docs) else 0,
        "durasi_detik": round(t, 4)
    }

    docs_to_chunk = chunker.load_from_disk()
    if not docs_to_chunk:
        print("Tidak ada dokumen untuk di-chunk. Pipeline dihentikan.")
        return

    # ---------------------------------------------------------
    # LOOP PER TEKNIK CHUNKING
    # ---------------------------------------------------------
    print("\n========== LOOP PER TEKNIK ==========")

    for teknik in TEKNIK_LIST:
        print(f"\n{'='*42}")
        print(f"  TEKNIK: {teknik.upper()}")
        print(f"{'='*42}")

        hasil_teknik = {
            "teknik": teknik,
            "timestamp_mulai": datetime.now().isoformat(),
            "tahap": {}
        }

        waktu_total_start = time.perf_counter()

        # Tahap 4: Chunking
        print(f"\n[1/2] CHUNKING — {teknik}")
        total_chunks = 0

        start = time.perf_counter()
        for doc in docs_to_chunk:
            chunks = chunker.chunk_document(
                text=doc["text"],
                teknik=teknik,
                filename=doc["filename"],
                matkul=doc["matkul"],
                save=True
            )
            total_chunks += len(chunks)
        t_chunk = time.perf_counter() - start
        print(f"  ✓ Chunking selesai ({t_chunk:.2f}s) — {total_chunks} chunks")

        hasil_teknik["tahap"]["chunker"] = {
            "total_chunks": total_chunks,
            "total_dokumen": len(docs_to_chunk),
            "durasi_detik": round(t_chunk, 4)
        }

        # Tahap 5: Embedding + FAISS
        print(f"\n[2/2] EMBEDDING & FAISS — {teknik}")
        start = time.perf_counter()
        try:
            embedder.build_index(teknik)
            t_embed = time.perf_counter() - start
            print(f"  ✓ Embedding selesai ({t_embed:.2f}s)")
            hasil_teknik["tahap"]["embedder"] = {
                "total_chunks_diembed": total_chunks,
                "durasi_detik": round(t_embed, 4),
                "status": "sukses"
            }
        except Exception as e:
            t_embed = time.perf_counter() - start
            print(f"  ✗ Embedding gagal: {e}")
            hasil_teknik["tahap"]["embedder"] = {
                "durasi_detik": round(t_embed, 4),
                "status": "gagal",
                "error": str(e)
            }

        # Total waktu teknik ini
        waktu_total = time.perf_counter() - waktu_total_start
        hasil_teknik["durasi_total_detik"] = round(waktu_total, 4)
        hasil_teknik["timestamp_selesai"] = datetime.now().isoformat()

        print(f"\n  Total waktu teknik {teknik}: {waktu_total:.2f}s")

        hasil_semua["per_teknik"][teknik] = hasil_teknik

        # Simpan hasil sementara per teknik
        output_path = OUTPUT_DIR / f"{teknik}_result.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(hasil_teknik, f, ensure_ascii=False, indent=2)
        print(f"  Hasil disimpan: {output_path}")

    # ---------------------------------------------------------
    # SIMPAN HASIL LENGKAP SEMUA TEKNIK
    # ---------------------------------------------------------
    hasil_semua["timestamp_selesai"] = datetime.now().isoformat()

    # Ringkasan perbandingan waktu antar teknik
    hasil_semua["ringkasan"] = {
        teknik: {
            "total_chunks_final": hasil_semua["per_teknik"][teknik]["tahap"].get("chunker", {}).get("total_chunks", 0),
            "durasi_chunking_detik": hasil_semua["per_teknik"][teknik]["tahap"].get("chunker", {}).get("durasi_detik", 0),
            "durasi_embedding_detik": hasil_semua["per_teknik"][teknik]["tahap"].get("embedder", {}).get("durasi_detik", 0),
            "durasi_total_detik": hasil_semua["per_teknik"][teknik].get("durasi_total_detik", 0),
        }
        for teknik in TEKNIK_LIST
        if teknik in hasil_semua["per_teknik"]
    }

    summary_path = OUTPUT_DIR / "run_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(hasil_semua, f, ensure_ascii=False, indent=2)

    print(f"\n=========================================")
    print(f"  PIPELINE SELESAI")
    print(f"  Ringkasan disimpan: {summary_path}")
    print(f"=========================================")

    # Print ringkasan ke terminal
    print(f"\n{'Teknik':<12} {'Chunks':<10} {'Chunking(s)':<14} {'Embedding(s)':<14} {'Total(s)'}")
    print("-" * 65)
    for teknik, r in hasil_semua["ringkasan"].items():
        print(f"{teknik:<12} {r['total_chunks_final']:<10} "
              f"{r['durasi_chunking_detik']:<14} {r['durasi_embedding_detik']:<14} {r['durasi_total_detik']}")


if __name__ == "__main__":
    main()