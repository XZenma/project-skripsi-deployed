import json
import sys
import os
import re

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.rag_chain import run_rag

# --- Daftar teknik chunking yang ingin diuji ---
TEKNIK_LIST = ["fixed", "recursive", "sentence", "semantic"]

# --- Daftar pertanyaan uji (silakan sesuaikan) ---
PERTANYAAN_LIST = [
    "Apakah source code di dalam stored procedure yang dibuat dengan opsi WITH ENCRYPTION dapat ditampilkan kembali menggunakan perintah sp_helptext?",
]

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_BASENAME = "hasil_uji_chatbot"


def get_next_output_path() -> str:
    """
    Cari nomor increment berikutnya berdasarkan file yang sudah ada.
    hasil_uji_chatbot_1.json -> hasil_uji_chatbot_2.json -> dst.
    """
    pattern = re.compile(rf"^{re.escape(OUTPUT_BASENAME)}_(\d+)\.json$")
    existing_numbers = []

    for fname in os.listdir(OUTPUT_DIR):
        match = pattern.match(fname)
        if match:
            existing_numbers.append(int(match.group(1)))

    next_number = max(existing_numbers, default=0) + 1
    return os.path.join(OUTPUT_DIR, f"{OUTPUT_BASENAME}_{next_number}.json")


def main():
    hasil_akhir = {}

    for teknik in TEKNIK_LIST:
        print(f"\n=== Menguji teknik: {teknik} ===")
        hasil_teknik = []

        for pertanyaan in PERTANYAAN_LIST:
            print(f"  -> {pertanyaan}")
            try:
                hasil = run_rag(pertanyaan, teknik)

                # Susun chunk hasil retrieval jadi format chunk_1, chunk_2, ...
                chunks_dict = {
                    f"chunk_{i+1}": teks
                    for i, teks in enumerate(hasil["konteks"])
                }

                hasil_teknik.append({
                    "pertanyaan": pertanyaan,
                    "jawaban": hasil["jawaban"],
                    "chunks": chunks_dict
                })
            except Exception as e:
                print(f"     ERROR: {e}")
                hasil_teknik.append({
                    "pertanyaan": pertanyaan,
                    "jawaban": f"[ERROR] {str(e)}",
                    "chunks": {}
                })

        hasil_akhir[teknik] = hasil_teknik

    output_path = get_next_output_path()

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(hasil_akhir, f, ensure_ascii=False, indent=2)

    print(f"\nSelesai. Hasil disimpan di: {output_path}")


if __name__ == "__main__":
    main()