import os
import sys
import json
import time
import pandas as pd
from langchain_openai import ChatOpenAI

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Sangat disarankan mengatur API key via Environment Variable (misal: os.environ.get("GROQ_API_KEY"))
# Namun untuk keperluan testing cepat, ini tetap bisa digunakan:
GROQ_API_KEY = "xxxx"

# LLM dari API eksternal (Model disesuaikan dengan daftar model Groq yang valid)
llm = ChatOpenAI(
    model="openai/gpt-oss-120b", # <-- Ganti dengan model Groq yang valid
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY,
    temperature=0
)

prompt_template = """Berikut adalah isi dokumen materi kuliah:

{teks_dokumen}

Berdasarkan dokumen di atas, ekstrak informasi dan buatkan MAKSIMAL 10 pasang pertanyaan dan ground_truth. Jika dokumen pendek, buatkan seadanya saja (tidak perlu dipaksakan 15).

KETENTUAN SANGAT PENTING:
1. Pertanyaan dan Ground Truth HARUS menggunakan BAHASA INDONESIA yang baku dan natural, meskipun teks dokumen berbahasa Inggris. (Lakukan penerjemahan dengan akurat pada Q&A yang dibuat).
2. Teks untuk key "sumber" boleh tetap menggunakan bahasa asli dokumen (kalimat asli).
3. Pertanyaan harus spesifik dan ground_truthnya tertera jelas di dokumen.
4. Ground Truth harus lengkap dan akurat. Jangan membuat pertanyaan yang terlalu umum atau trivial.
5. Format output HARUS MURNI berupa array JSON, TANPA teks awalan/akhiran apapun, dan TANPA format markdown (jangan gunakan ```json).
6. SANGAT PENTING: Pastikan format JSON tertutup dengan sempurna. Respons Anda HARUS dan WAJIB diakhiri dengan karakter `]` (kurung siku tutup).

Struktur JSON yang WAJIB digunakan:
[
  {{"pertanyaan": "...", "ground_truth": "...", "sumber": "kalimat asli dari dokumen yang menjadi dasar jawaban"}},
  {{"pertanyaan": "...", "ground_truth": "...", "sumber": "kalimat asli dari dokumen yang menjadi dasar jawaban"}}
]"""

base_folder = "data/deduplicate/saved"
hasil_qa = []

# Pastikan folder base ada
if not os.path.exists(base_folder):
    print(f"Folder {base_folder} tidak ditemukan!")
    sys.exit()

# ==========================================
# MENU INTERAKTIF UNTUK MEMILIH TARGET PROSES
# ==========================================
print("=== MENU PEMROSESAN Q&A ===")
print("1. Proses SEMUA folder dan file")
print("2. Proses HANYA SATU FOLDER mata kuliah tertentu")
print("3. Proses HANYA SATU FILE tertentu")
pilihan = input("Masukkan pilihan Anda (1/2/3): ").strip()

target_folder = ""
target_file = ""

if pilihan == "2":
    target_folder = input("Masukkan nama folder mata kuliah (contoh: Basis Data 2): ").strip()
elif pilihan == "3":
    target_file = input("Masukkan nama file beserta ekstensinya (contoh: TM-01.txt): ").strip()
elif pilihan != "1":
    print("Pilihan tidak valid. Membatalkan program.")
    sys.exit()

# Mengumpulkan daftar file yang akan diproses berdasarkan pilihan
files_to_process = []

for matkul in os.listdir(base_folder):
    matkul_path = os.path.join(base_folder, matkul)
    
    if not os.path.isdir(matkul_path):
        continue
        
    # Filter Folder (Jika pilih 2)
    if pilihan == "2" and matkul != target_folder:
        continue
    
    for filename in os.listdir(matkul_path):
        if not filename.endswith(".txt"):
            continue
            
        # Filter File (Jika pilih 3)
        if pilihan == "3" and filename != target_file:
            continue
            
        filepath = os.path.join(matkul_path, filename)
        files_to_process.append((matkul, filename, filepath))

if not files_to_process:
    print("\nTidak ada file yang cocok dengan kriteria pencarian Anda!")
    sys.exit()

print(f"\nDitemukan {len(files_to_process)} file untuk diproses. Memulai...\n")

# ==========================================
# EKSEKUSI PEMROSESAN
# ==========================================
for matkul, filename, filepath in files_to_process:
    print(f"=== Matkul: {matkul} ===")
    
    with open(filepath, "r", encoding="utf-8") as f:
        teks = f.read()
    
    if not teks.strip():
        print(f"  Teks kosong, skip: {filename}")
        continue
    
    print(f"  Memproses: {filename}")
    prompt = prompt_template.format(teks_dokumen=teks)
    
    try:
        response = llm.invoke(prompt)
        raw_json = response.content.strip()
        
        # Membersihkan markdown jika LLM membungkus dengan tag
        if raw_json.startswith("```json"):
            raw_json = raw_json[7:]
        elif raw_json.startswith("```"):
            raw_json = raw_json[3:]
        
        if raw_json.endswith("```"):
            raw_json = raw_json[:-3]
            
        raw_json = raw_json.strip()

        # [Jaring Pengaman JSON Terpotong]
        if raw_json.endswith("}"):
            raw_json += "\n]"
        elif raw_json.endswith('"'):
            raw_json += "\n  }\n]"
        elif not raw_json.endswith("]"):
            raw_json += '"\n  }\n]'
        
        # Coba Parse JSON
        qa_list = json.loads(raw_json)
        
        for qa in qa_list:
            qa["mata_kuliah"] = matkul
            qa["sumber_dokumen"] = filename
            hasil_qa.append(qa)
        
        print(f"  -> {len(qa_list)} Q&A berhasil digenerate")
        
    except json.JSONDecodeError:
        print(f"  -> Gagal parse JSON: {filename}")
        print("  === MULAI RAW OUTPUT AI ===")
        print(response.content)
        print("  === SELESAI RAW OUTPUT AI ===\n")
        
        with open(f"failed_{matkul}_{filename}.txt", "w", encoding="utf-8") as f:
            f.write(response.content)
            
    except Exception as e:
        print(f"  -> Error API / Eksekusi: {filename} - {str(e)}")
    
    # Simpan progress setiap file selesai
    if hasil_qa:
        df_progress = pd.DataFrame(hasil_qa)
        df_progress.to_csv("ground_truth_progress.csv", index=False)
    
    # Jeda agar tidak terkena Rate Limit API
    print("  [Jeda 5 detik...]\n")
    time.sleep(60)

# Simpan hasil final
if hasil_qa:
    df_final = pd.DataFrame(hasil_qa)
    df_final.to_csv("ground_truth_final.csv", index=False)
    print(f"\nSELESAI! Total Q&A keseluruhan: {len(hasil_qa)} disimpan ke ground_truth_final.csv")
else:
    print("\nTidak ada Q&A yang berhasil diekstrak.")