import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from config import LOAD_MODE, TARGET_MATKUL, RAW_DATA_PATH, LOADED_PATH
from langchain_community.document_loaders import PyMuPDFLoader


def load_pdf(file_path: str) -> str:
    """
    Load dan ekstrak teks dari PDF menggunakan PyMuPDFLoader.
    Seluruh halaman digabung menjadi satu teks utuh (mode single).
    """
    loader = PyMuPDFLoader(file_path, mode="single")
    docs = loader.load()
    text = docs[0].page_content if docs else ""
    return text


def save_text_file(matkul: str, filename: str, text: str, output_path: str):
    """
    Simpan teks ke <output_path>/<matkul>/<nama_file>.txt
    """
    matkul_folder = Path(output_path) / matkul
    matkul_folder.mkdir(parents=True, exist_ok=True)

    txt_filename = Path(filename).stem + ".txt"
    save_path = matkul_folder / txt_filename

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(text)

    return save_path


def load_documents(folder_path: str = RAW_DATA_PATH, save_output: bool = True) -> list[dict]:
    """
    Load PDF sesuai mode:
    - all    = semua matkul
    - single = satu matkul saja (TARGET_MATKUL)

    HANYA melakukan ekstraksi teks mentah dari PDF.
    Tidak melakukan cleaning — itu tugas cleaner.py, dipanggil terpisah.
    """
    folder = Path(folder_path)
    documents = []

    if LOAD_MODE == "single":
        if not TARGET_MATKUL:
            print("TARGET_MATKUL belum diisi di .env")
            return []

        matkul_path = folder / TARGET_MATKUL
        if not matkul_path.exists():
            print(f"Folder matkul tidak ditemukan: {matkul_path}")
            return []   

        matkul_folders = [matkul_path]
        print(f"Mode: single matkul → {TARGET_MATKUL}")

    else:
        matkul_folders = [f for f in folder.iterdir() if f.is_dir()]
        print(f"Mode: semua matkul → {len(matkul_folders)} matkul ditemukan")

    for matkul_folder in matkul_folders:
        pdf_files = list(matkul_folder.glob("*.pdf"))

        if not pdf_files:
            print(f"  Tidak ada PDF di: {matkul_folder.name}")
            continue

        print(f"\nMatkul: {matkul_folder.name} ({len(pdf_files)} file)")

        for pdf_file in pdf_files:
            print(f"  Memproses: {pdf_file.name}")

            raw_text = load_pdf(str(pdf_file))

            documents.append({
                "filename": pdf_file.name,
                "matkul": matkul_folder.name,
                "text": raw_text
            })

            if save_output:
                saved_path = save_text_file(matkul_folder.name, pdf_file.name, raw_text, LOADED_PATH)
                print(f"    Tersimpan: {saved_path}")

    print(f"\nTotal dokumen berhasil diload: {len(documents)}")
    return documents


# test loader
if __name__ == "__main__":
    docs = load_documents()
    for doc in docs:
        print(f"\nFile: {doc['filename']}")
        print(f"Panjang teks: {len(doc['text'])} karakter")