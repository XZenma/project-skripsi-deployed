import pandas as pd
import json

# 1. Baca file CSV
file_csv = "dataset_skripsi.csv"
df = pd.read_csv(file_csv, encoding="utf-8-sig")

# 2. Bersihkan kolom kosong tak bernama (sisa trailing koma di CSV)
df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

# 3. Bersihkan whitespace di kolom teks (misal " Komparatif" -> "Komparatif")
for kolom in ["question", "ground_truth", "major", "document_source", "question_type"]:
    if kolom in df.columns:
        df[kolom] = df[kolom].astype(str).str.strip()

# 4. Susun jadi list datar (flat list), sesuai urutan baris di CSV
hasil_json = []
for _, row in df.iterrows():
    hasil_json.append({
        "question": row["question"],
        "ground_truth": row["ground_truth"],
        "major": row["major"],
        "document_source": row["document_source"],
        "question_type": row["question_type"],
    })

# 5. Simpan hasilnya ke dalam file .json
file_json = "Dataset_skripsi.json"
with open(file_json, "w", encoding="utf-8") as f:
    json.dump(hasil_json, f, indent=4, ensure_ascii=False)

print(f"Konversi selesai! Total {len(hasil_json)} entri disimpan ke {file_json}")