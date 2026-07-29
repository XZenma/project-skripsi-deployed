import pandas as pd
import json

# 1. Baca file CSV
file_csv = "ground_truth_final_2.csv"
df = pd.read_csv(file_csv)

# 2. Siapkan dictionary utama
hasil_json = {}

# 3. Tambahkan sort=False agar urutannya tidak diubah jadi alfabetis!
for sumber_dokumen, group in df.groupby("sumber_dokumen", sort=False):
    qa_list = []
    
    # Looping setiap baris dalam dokumen yang sama
    for index, row in group.iterrows():
        qa_list.append({
            "pertanyaan": row["pertanyaan"],
            "ground_truth": row["ground_truth"],
            "sumber_text": row["sumber"] 
        })
        
    hasil_json[sumber_dokumen] = qa_list

# 4. Simpan hasilnya ke dalam file .json
file_json = "ground_truth_final.json"
with open(file_json, "w", encoding="utf-8") as f:
    json.dump(hasil_json, f, indent=4, ensure_ascii=False)

print("Konversi selesai! Urutan sekarang sudah sesuai dengan baris di CSV.")