import pandas as pd
import json

def convert_csv_to_json_pandas(csv_path, json_path):
    # 1. Baca file CSV
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    
    # 2. Filter hanya 3 kolom yang dibutuhkan (untuk mencegah kolom ekstra ikut masuk)
    kolom_dibutuhkan = ["question", "ground_truth", "matkul"]
    df_filtered = df[kolom_dibutuhkan]
    
    # 3. Simpan ke JSON dengan format list of dictionaries (orient="records")
    df_filtered.to_json(json_path, orient="records", force_ascii=False, indent=4)
    
    print(f"Berhasil mengonversi CSV ke JSON! Disimpan di: {json_path}")

# Cara memanggil fungsinya:
if __name__ == "__main__":
    file_csv = "ground_truth_final_2.csv"   # Ganti dengan path file CSV Anda
    file_json = "ground_truth_final.json" # Ganti dengan path file JSON tujuan
    
    convert_csv_to_json_pandas(file_csv, file_json)