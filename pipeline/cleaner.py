import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import emoji
from pathlib import Path
from config import LOADED_PATH, CLEANED_PATH


def normalize_unicode(text: str) -> str:
    replacements = {
        "\u201c": '"', "\u201d": '"',
        "\u2018": "'", "\u2019": "'",
        "\u2013": "-", "\u2014": "-",
        "\u2026": "...", "\u00a0": " ",
        "\uff0c": ",", "\u3002": ".",
    }
    for old_char, new_char in replacements.items():
        text = text.replace(old_char, new_char)
    return text


def clean_text(text: str) -> str:
    
    # ── GRUP 1: Normalisasi karakter dasar ────────────────────────
    # Tujuan: seragamkan karakter "siluman" dari PDF dan hapus emoji
    text = normalize_unicode(text)          
    text = emoji.replace_emoji(text, replace='')   
    text = re.sub(r'[\u2460-\u2473\u24ea\u2776-\u277f]', '', text)  # Hapus lingkaran angka (①-⑳)
    text = re.sub(r'[\ue000-\uf8ff]', '', text)                     # Hapus artefak Wingdings (, )
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\x80-\x9f]', '', text) # Hapus karakter kontrol (invisible)
    text = re.sub(r'[\u2022\u2219\u25E6\u25CF\u25AA\u00B7]', '', text)    # Hapus simbol bullet points

    # ── GRUP 1.6: Sapu Bersih Karakter Alien (Metode Whitelist) ───
    # HAPUS SEMUA karakter di luar alfabet Latin standar (ASCII), angka, tanda baca standar, enter, dan tab.
    # Regex ini otomatis memusnahkan huruf Mandarin, Arab, Thai, aksara aneh, koma lebar (，), dll.
    text = re.sub(r'[\u4e00-\u9fff\u3040-\u30ff\uff00-\uffef\u0600-\u06FF\u0750-\u077F\u0400-\u04FF\uac00-\ud7af\u0e00-\u0e7f]', '', text)

    # ── GRUP 1.7: Hapus Teks Rusak Tanpa Spasi (Garbage Filter) ───
    # Menghapus kata apa pun (huruf yang menempel) yang panjangnya lebih dari 15 karakter.
    # Ini menangani kasus "6OEFSTUBOEUIFJNQPSUBODF..." akibat hilangnya CMap spasi di PDF.
    text = re.sub(r'\b[A-Z0-9]{25,}\b', '', text)

    # ── GRUP 2: Normalisasi whitespace per baris ──────────────────
    # Tujuan: bersihkan spasi tersembunyi di tiap baris dulu
    lines = text.split('\n')
    lines = [line.strip() for line in lines]
    text = '\n'.join(lines)

    # ── GRUP 3: Rapikan struktur teks ─────────────────────────────
    # Tujuan: hapus kelebihan baris kosong dan spasi berlebihan
    text = re.sub(r'\n{3,}', '\n\n', text)   # maks 2 baris kosong berturut-turut
    text = re.sub(r'[ \t]+', ' ', text)      # spasi/tab ganda → 1 spasi
    text = re.sub(r' {2,}', ' ', text)       # jaga-jaga sisa spasi ganda

    # ── GRUP 4: Finalisasi ────────────────────────────────────────
    # Tujuan: hapus sisa whitespace di awal/akhir seluruh teks
    text = text.strip()

    return text


def save_text_file(matkul: str, filename: str, text: str, output_path: str):
    matkul_folder = Path(output_path) / matkul
    matkul_folder.mkdir(parents=True, exist_ok=True)
    save_path = matkul_folder / filename
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(text)
    return save_path


def load_from_disk(folder_path: str = LOADED_PATH) -> list[dict]:
    """
    Baca semua file .txt dari data/loaded/<matkul>/*.txt
    """
    folder = Path(folder_path)
    if not folder.exists() or not any(folder.glob("**/*.txt")):
        print(f"Data di {folder_path} tidak ditemukan atau kosong. Menjalankan loader...")
        import pipeline.loader as loader
        loader.load_documents()

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
            print(f"  Dibaca: {txt_file.name}")

    print(f"\nTotal dokumen dibaca: {len(documents)}")
    return documents


def clean_documents(documents: list[dict], save_output: bool = True) -> list[dict]:
    cleaned_documents = []
    for doc in documents:
        clean = clean_text(doc["text"])
        cleaned_documents.append({
            "filename": doc["filename"],
            "matkul": doc["matkul"],
            "text": clean
        })
        if save_output:
            saved_path = save_text_file(doc["matkul"], doc["filename"], clean, CLEANED_PATH)
            print(f"  Dibersihkan: {doc['filename']} → {saved_path}")
    return cleaned_documents

def ensure_input(folder_path: str = LOADED_PATH):
    """
    Pastikan folder input tersedia.
    Kalau tidak ada, jalankan loader dulu secara otomatis.
    """
    if not Path(folder_path).exists() or not any(Path(folder_path).rglob("*.txt")):
        print(f"⚠️  Folder '{folder_path}' tidak ditemukan atau kosong.")
        print("    → Menjalankan loader.py otomatis...")
        from pipeline.loader import load_documents
        load_documents()


if __name__ == "__main__":
    ensure_input()
    docs = load_from_disk()
    if docs:
        cleaned = clean_documents(docs)
        print(f"\nTotal dokumen dibersihkan: {len(cleaned)}")
        
        