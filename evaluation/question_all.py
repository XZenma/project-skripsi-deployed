import json
import sys
import os
import re

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from pipeline.rag_chain import run_rag


# ============================================================
# KONFIGURASI TESTING
# ============================================================

# Deployment hanya menggunakan Fixed-size
TEKNIK = "fixed"

# Mata kuliah yang ingin diuji
MATKUL = "Basis Data 2"

# Daftar pertanyaan uji
PERTANYAAN_LIST = [
    "Apa itu CMMI",
]


# ============================================================
# KONFIGURASI OUTPUT
# ============================================================

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_BASENAME = "hasil_uji_chatbot"


def get_next_output_path() -> str:
    """
    Mencari nomor increment berikutnya berdasarkan file
    yang sudah ada.

    Contoh:
    hasil_uji_chatbot_1.json
    hasil_uji_chatbot_2.json
    hasil_uji_chatbot_3.json
    """

    pattern = re.compile(
        rf"^{re.escape(OUTPUT_BASENAME)}_(\d+)\.json$"
    )

    existing_numbers = []

    for fname in os.listdir(OUTPUT_DIR):
        match = pattern.match(fname)

        if match:
            existing_numbers.append(
                int(match.group(1))
            )

    next_number = max(
        existing_numbers,
        default=0
    ) + 1

    return os.path.join(
        OUTPUT_DIR,
        f"{OUTPUT_BASENAME}_{next_number}.json"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("TESTING CHATBOT RAG")
    print("=" * 60)

    print(f"Teknik  : {TEKNIK}")
    print(f"Matkul  : {MATKUL}")
    print()

    hasil_akhir = {
        TEKNIK: []
    }

    for pertanyaan in PERTANYAAN_LIST:

        print("-" * 60)
        print(f"Pertanyaan: {pertanyaan}")
        print(f"Mata kuliah: {MATKUL}")

        try:

            # ==================================================
            # RUN RAG
            # ==================================================

            hasil = run_rag(
                pertanyaan,
                TEKNIK,
                matkul=MATKUL
            )

            # ==================================================
            # SIMPAN CHUNK + METADATA
            # ==================================================

            sumber = hasil.get("sumber", [])
            konteks = hasil.get("konteks", [])

            chunks_dict = {}

            for i, teks in enumerate(konteks):

                metadata = {}

                if i < len(sumber):
                    metadata = sumber[i]

                chunks_dict[f"chunk_{i + 1}"] = {
                    "text": teks,
                    "filename": metadata.get("filename"),
                    "matkul": metadata.get("matkul"),
                    "index_chunk": metadata.get("index_chunk"),
                    "distance": metadata.get("distance")
                }

            # ==================================================
            # SIMPAN HASIL
            # ==================================================

            hasil_akhir[TEKNIK].append({
                "pertanyaan": pertanyaan,
                "matkul": MATKUL,
                "jawaban": hasil["jawaban"],
                "chunks": chunks_dict
            })

            # ==================================================
            # PRINT HASIL KE TERMINAL
            # ==================================================

            print("\nJawaban:")
            print(hasil["jawaban"])

            print("\nChunk yang digunakan:")

            for i, chunk in enumerate(
                chunks_dict.values(),
                start=1
            ):

                print(
                    f"\n--- Chunk {i} ---"
                )

                print(
                    f"File     : {chunk['filename']}"
                )

                print(
                    f"Matkul   : {chunk['matkul']}"
                )

                print(
                    f"Index    : {chunk['index_chunk']}"
                )

                print(
                    f"Distance : {chunk['distance']}"
                )

                print(
                    f"Text     : {chunk['text'][:300]}..."
                )

        except Exception as e:

            print(
                f"\nERROR: {str(e)}"
            )

            hasil_akhir[TEKNIK].append({
                "pertanyaan": pertanyaan,
                "matkul": MATKUL,
                "jawaban": f"[ERROR] {str(e)}",
                "chunks": {}
            })


    # ========================================================
    # SIMPAN JSON
    # ========================================================

    output_path = get_next_output_path()

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            hasil_akhir,
            f,
            ensure_ascii=False,
            indent=2
        )

    print("\n" + "=" * 60)
    print("TESTING SELESAI")
    print("=" * 60)

    print(
        f"Hasil disimpan di:\n{output_path}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()