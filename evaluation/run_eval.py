import os, sys, json, time
from openai import AsyncOpenAI
from openai import AsyncOpenAI
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_openai import ChatOpenAI
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_correctness,
)
from ragas.llms import llm_factory
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_ollama import ChatOllama
from ragas.run_config import RunConfig
from pipeline.rag_chain import run_rag
from pipeline.embedder import get_embeddings
from config import MODEL_NAME, GROUND_TRUTH_PATH, EVAL_RESULTS_PATH

OPENAI_API_KEY = "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"

def load_ground_truth(path: str = GROUND_TRUTH_PATH) -> list[dict]:
    """
    Baca dataset ground truth (100 pasang pertanyaan-jawaban)
    dari file JSON. Setiap entri minimal punya: question, ground_truth, matkul.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Ground truth dimuat: {len(data)} entri")
    return data

def get_answer_cache_path(teknik: str) -> str:
    """
    Path file cache jawaban LLM untuk satu teknik chunking.
    Format: evaluation/RAGAS/<teknik>/LLM_answer.json
    """
    folder = os.path.join("evaluation", "RAGAS", teknik)
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, "LLM_answer.json")


def run_pipeline_for_teknik(ground_truth: list[dict], teknik: str) -> list[dict]:
    """
    Jalankan pipeline RAG (retrieve + generate) untuk seluruh ground truth
    pada satu teknik chunking, kumpulkan jawaban dan konteks yang terambil.
 
    Kalau cache jawaban untuk teknik ini sudah ada (LLM_answer.json), hasil
    generation dibaca langsung dari situ tanpa memanggil LLM ulang. Ini
    penting supaya kalau evaluasi RAGAS gagal/error di tengah jalan, proses
    generate 200x jawaban tidak perlu diulang dari nol.
    """
    cache_path = get_answer_cache_path(teknik)
 
    if os.path.exists(cache_path):
        print(f"[{teknik}] Cache ditemukan di {cache_path}, membaca hasil generation sebelumnya...")
        with open(cache_path, "r", encoding="utf-8") as f:
            hasil = json.load(f)
 
        if len(hasil) != len(ground_truth):
            print(
                f"[{teknik}] ⚠️  PERINGATAN: jumlah cache ({len(hasil)}) tidak sama dengan "
                f"jumlah ground truth saat ini ({len(ground_truth)}). Kemungkinan cache ini "
                f"berasal dari run sebelumnya dengan limit_data berbeda. Cache DIABAIKAN, "
                f"generation akan diulang dari awal untuk teknik ini."
            )
        else:
            print(f"[{teknik}] {len(hasil)} jawaban dimuat dari cache.")
            return hasil
 
    print(f"[{teknik}] Cache belum ada, menjalankan generation untuk {len(ground_truth)} pertanyaan...")
    hasil = []
    total = len(ground_truth)
 
    for i, entry in enumerate(ground_truth):
        query = entry["question"]
        print(f"[{teknik}] {i+1}/{total}: {query[:60]}...")
 
        rag_output = run_rag(query, teknik)
 
        hasil.append({
            "question": query,
            "answer": rag_output["jawaban"],
            "contexts": rag_output["konteks"],
            "ground_truth": entry["ground_truth"],
            "major": entry.get("major"),
            "document_source": entry.get("document_source"),
            "question_type": entry.get("question_type"),
        })
 
    # Simpan ke cache setelah semua selesai di-generate
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(hasil, f, ensure_ascii=False, indent=2)
    print(f"[{teknik}] Hasil generation disimpan ke cache: {cache_path}")
 
    return hasil


def get_ragas_llm_and_embeddings():
    """
    Siapkan wrapper LLM dan embeddings untuk dipakai sebagai 'juri'
    penilaian metrik RAGAS (faithfulness, answer_correctness butuh LLM;
    context_precision, context_recall, answer_correctness butuh embeddings).
    """
    # 1. Setup LLM menggunakan standar baru v0.4
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    
    # llm_factory akan mendeteksi model Anda otomatis dan membereskan masalah temperature!
    llm = llm_factory("gpt-5.6-luna", client=client, max_tokens=8192)

    embeddings = LangchainEmbeddingsWrapper(get_embeddings())
    return llm, embeddings

def evaluate_teknik(hasil_pipeline: list[dict], llm, embeddings) -> pd.DataFrame:
    """
    Jalankan evaluasi RAGAS terhadap hasil pipeline satu teknik.
    """
    samples = []
    for r in hasil_pipeline:
        sample = SingleTurnSample(
            user_input=r["question"],          # Menggantikan "question"
            response=r["answer"],              # Menggantikan "answer"
            retrieved_contexts=r["contexts"],  # Menggantikan "contexts"
            reference=r["ground_truth"]        # Menggantikan "ground_truth"
        )
        samples.append(sample)

    dataset = EvaluationDataset(samples=samples)

    run_config = RunConfig(
        max_workers=1,
        timeout=600,
        max_retries=3,
    )
 
    result = evaluate(
        dataset=dataset,
        metrics=[context_precision, context_recall, faithfulness, answer_correctness],
        llm=llm,
        embeddings=embeddings,
        run_config=run_config,
    )

    df = result.to_pandas()

    df["major"] = [r["major"] for r in hasil_pipeline]
    df["document_source"] = [r["document_source"] for r in hasil_pipeline]
    df["question_type"] = [r["question_type"] for r in hasil_pipeline]

    return df

def run_full_evaluation(limit_data=None):
    """
    Jalankan evaluasi RAGAS lengkap untuk keempat teknik chunking,
    simpan skor per-pertanyaan (JSON) dan ringkasan rata-rata per teknik.
 
    Kalau file detail_<teknik>.json sudah ada dari run sebelumnya, evaluasi
    RAGAS untuk teknik itu TIDAK diulang -- skor langsung dibaca dari file
    tersebut untuk disusun ke ringkasan. Ini menghemat waktu dan biaya API
    kalau proses sempat berhenti di tengah jalan (misal error di teknik ke-3,
    teknik ke-1 dan ke-2 tidak perlu dievaluasi ulang).
    """
    ground_truth = load_ground_truth()
 
    if limit_data is not None:
        ground_truth = ground_truth[:limit_data]
        print(f"⚠️ MODE TESTING AKTIF: Hanya menggunakan {len(ground_truth)} data pertama.")
 
    llm, embeddings = get_ragas_llm_and_embeddings()
 
    os.makedirs(EVAL_RESULTS_PATH, exist_ok=True)
    ringkasan = {}
 
    for teknik in ["fixed", "recursive", "sentence", "semantic"]:
        print(f"\n=== Evaluasi teknik: {teknik} ===")
 
        detail_path = os.path.join(EVAL_RESULTS_PATH, f"detail_{teknik}.json")
 
        if os.path.exists(detail_path):
            print(f"[{teknik}] Hasil evaluasi sudah ada di {detail_path}, "
                  f"langsung dipakai untuk ringkasan (evaluasi RAGAS tidak diulang).")
            df_skor = pd.read_json(detail_path, orient="records")
        else:
            hasil_pipeline = run_pipeline_for_teknik(ground_truth, teknik)
            df_skor = evaluate_teknik(hasil_pipeline, llm, embeddings)
 
            df_skor.to_json(detail_path, orient="records", force_ascii=False, indent=4)
            print(f"Skor per-pertanyaan disimpan di: {detail_path}")
 
        ringkasan[teknik] = {
            "context_precision": round(df_skor["context_precision"].mean(), 4),
            "context_recall": round(df_skor["context_recall"].mean(), 4),
            "faithfulness": round(df_skor["faithfulness"].mean(), 4),
            "answer_correctness": round(df_skor["answer_correctness"].mean(), 4),
        }
 
    ringkasan_path = os.path.join(EVAL_RESULTS_PATH, "ringkasan_evaluasi.json")
    with open(ringkasan_path, "w", encoding="utf-8") as f:
        json.dump(ringkasan, f, ensure_ascii=False, indent=2)
 
    print(f"\n=== Ringkasan evaluasi disimpan di: {ringkasan_path} ===")
    print(json.dumps(ringkasan, indent=2, ensure_ascii=False))
 
    return ringkasan

if __name__ == "__main__":
    run_full_evaluation()