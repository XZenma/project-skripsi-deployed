"""
Analisis distribusi panjang seluruh chunk di docstore (bukan hanya hasil query),
disimpan dalam bentuk JSON dengan daftar lengkap chunk yang berada di bawah
setiap ambang batas (20, 30, 50, 75, 100 karakter).

Cara pakai:
    python analisis_panjang_chunk.py
"""

import os, sys, json
import re

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import FAISS_PATHS

AMBANG_UJI = [20, 30, 50, 75, 100]

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_BASENAME = "analisis_panjang_chunk"


def load_docstore(teknik: str):
    docstore_path = os.path.join(FAISS_PATHS[teknik], "docstore.json")
    with open(docstore_path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalisasi_ringan(teks: str) -> str:
    return re.sub(r"\s+", " ", teks).strip()


def get_next_output_path() -> str:
    pattern = re.compile(rf"^{re.escape(OUTPUT_BASENAME)}_(\d+)\.json$")
    existing_numbers = []
    for fname in os.listdir(OUTPUT_DIR):
        match = pattern.match(fname)
        if match:
            existing_numbers.append(int(match.group(1)))
    next_number = max(existing_numbers, default=0) + 1
    return os.path.join(OUTPUT_DIR, f"{OUTPUT_BASENAME}_{next_number}.json")


def analisis_teknik(teknik: str) -> dict:
    try:
        docstore = load_docstore(teknik)
    except FileNotFoundError:
        return {"error": f"docstore.json untuk teknik '{teknik}' tidak ditemukan."}

    n = len(docstore)
    lengths = [len(entry.get("text", "")) for entry in docstore]
    lengths_sorted = sorted(lengths)

    # --- statistik dasar ---
    statistik = {
        "total_chunk": n,
        "min": lengths_sorted[0] if n else None,
        "max": lengths_sorted[-1] if n else None,
        "median": lengths_sorted[n // 2] if n else None,
    }

    # --- daftar lengkap chunk per ambang batas ---
    chunk_dibawah_ambang = {}
    for ambang in AMBANG_UJI:
        daftar = []
        for entry in docstore:
            teks = entry.get("text", "")
            if len(teks) < ambang:
                daftar.append({
                    "filename": entry.get("filename"),
                    "matkul": entry.get("matkul"),
                    "index_chunk": entry.get("index_chunk"),
                    "panjang_karakter": len(teks),
                    "text": teks
                })
        jumlah = len(daftar)
        persen = round((jumlah / n) * 100, 2) if n else 0
        chunk_dibawah_ambang[f"kurang_dari_{ambang}"] = {
            "jumlah": jumlah,
            "persen": persen,
            "daftar_chunk": daftar
        }

    # --- deteksi near-duplicate whitespace-only (byte beda, isi sama setelah normalisasi) ---
    hash_normed = {}
    near_duplicates = []
    for entry in docstore:
        teks = entry.get("text", "")
        key = normalisasi_ringan(teks)
        if key in hash_normed:
            near_duplicates.append({
                "filename_1": hash_normed[key].get("filename"),
                "index_chunk_1": hash_normed[key].get("index_chunk"),
                "filename_2": entry.get("filename"),
                "index_chunk_2": entry.get("index_chunk"),
                "preview": teks[:150]
            })
        else:
            hash_normed[key] = entry

    return {
        "teknik": teknik,
        "statistik": statistik,
        "chunk_dibawah_ambang": chunk_dibawah_ambang,
        "near_duplicate_whitespace": {
            "jumlah": len(near_duplicates),
            "daftar": near_duplicates
        }
    }


def main():
    hasil_akhir = {}

    for teknik in ["fixed", "recursive", "sentence", "semantic"]:
        print(f"Menganalisis teknik: {teknik}...")
        hasil_akhir[teknik] = analisis_teknik(teknik)

    output_path = get_next_output_path()
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(hasil_akhir, f, ensure_ascii=False, indent=2)

    print(f"\nSelesai. Hasil tersimpan di: {output_path}")

    # ringkasan singkat di terminal
    print("\n=== Ringkasan ===")
    for teknik, hasil in hasil_akhir.items():
        if "error" in hasil:
            print(f"{teknik}: {hasil['error']}")
            continue
        stat = hasil["statistik"]
        print(f"\n{teknik} (total {stat['total_chunk']} chunk, min={stat['min']}, max={stat['max']}, median={stat['median']})")
        for ambang in AMBANG_UJI:
            info = hasil["chunk_dibawah_ambang"][f"kurang_dari_{ambang}"]
            print(f"  < {ambang:3d} karakter: {info['jumlah']} ({info['persen']}%)")
        print(f"  Near-duplicate whitespace-only: {hasil['near_duplicate_whitespace']['jumlah']}")


if __name__ == "__main__":
    main()