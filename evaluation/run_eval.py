import os, sys, json, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_openai import ChatOpenAI
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_correctness,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_ollama import ChatOllama
from ragas.run_config import RunConfig
from pipeline.rag_chain import run_rag
from pipeline.embedder import get_embeddings
from config import MODEL_NAME, GROUND_TRUTH_PATH, EVAL_RESULTS_PATH

def load_ground_truth(path: str = GROUND_TRUTH_PATH) -> list[dict]:
    """
    Baca dataset ground truth (100 pasang pertanyaan-jawaban)
    dari file JSON. Setiap entri minimal punya: question, ground_truth, matkul.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Ground truth dimuat: {len(data)} entri")
    return data


def run_pipeline_for_teknik(ground_truth: list[dict], teknik: str) -> list[dict]:
    """
    Jalankan pipeline RAG (retrieve + generate) untuk seluruh ground truth
    pada satu teknik chunking, kumpulkan jawaban dan konteks yang terambil.
    """
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
            "matkul": entry.get("matkul"),
            "tipe_pertanyaan": entry.get("tipe_pertanyaan"),
        })

    return hasil


def get_ragas_llm_and_embeddings():
    """
    Siapkan wrapper LLM dan embeddings untuk dipakai sebagai 'juri'
    penilaian metrik RAGAS (faithfulness, answer_correctness butuh LLM;
    context_precision, context_recall, answer_correctness butuh embeddings).
    """
    GROQ_API_KEY = "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"
    llm = LangchainLLMWrapper(
        ChatOpenAI(
            model="openai/gpt-oss-20b", # <-- Ganti dengan model Groq yang valid
            base_url="https://api.groq.com/openai/v1",
            api_key=GROQ_API_KEY,
            temperature=0
        )
    )
    embeddings = LangchainEmbeddingsWrapper(get_embeddings())
    return llm, embeddings


def evaluate_teknik(hasil_pipeline: list[dict], llm, embeddings) -> pd.DataFrame:
    """
    Jalankan evaluasi RAGAS terhadap hasil pipeline satu teknik.
    """
    dataset = Dataset.from_list([
        {
            "question": r["question"],
            "answer": r["answer"],
            "contexts": r["contexts"],
            "ground_truth": r["ground_truth"],
        }
        for r in hasil_pipeline
    ])

    konfigurasi_ollama = RunConfig(
        max_workers=1, 
        timeout=180, 
        max_retries=3
    )
    # -------------------------------------

    result = evaluate(
        dataset=dataset,
        metrics=[context_precision, context_recall, faithfulness, answer_correctness],
        llm=llm,
        embeddings=embeddings,
        run_config=konfigurasi_ollama
    )

    df = result.to_pandas()

    # Sisipkan kembali metadata (matkul, tipe_pertanyaan)
    df["matkul"] = [r["matkul"] for r in hasil_pipeline]
    df["tipe_pertanyaan"] = [r["tipe_pertanyaan"] for r in hasil_pipeline]

    return df

def run_full_evaluation(limit_data=None):
    """
    Jalankan evaluasi RAGAS lengkap untuk keempat teknik chunking,
    simpan skor per-pertanyaan dan ringkasan rata-rata per teknik.
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
        start = time.time()

        hasil_pipeline = run_pipeline_for_teknik(ground_truth, teknik)
        df_skor = evaluate_teknik(hasil_pipeline, llm, embeddings)

        # --- BAGIAN YANG DIUBAH (Dari CSV ke JSON) ---
        # Ubah ekstensi file menjadi .json
        detail_path = os.path.join(EVAL_RESULTS_PATH, f"detail_{teknik}.json")
        
        # Simpan DataFrame ke JSON dengan format yang mudah dibaca (indent=4)
        df_skor.to_json(detail_path, orient="records", force_ascii=False, indent=4)
        print(f"Skor per-pertanyaan disimpan di: {detail_path}")
        # ---------------------------------------------

        # Simpan ringkasan rata-rata per teknik
        ringkasan[teknik] = {
            "context_precision": round(df_skor["context_precision"].mean(), 4),
            "context_recall": round(df_skor["context_recall"].mean(), 4),
            "faithfulness": round(df_skor["faithfulness"].mean(), 4),
            "answer_correctness": round(df_skor["answer_correctness"].mean(), 4),
            "durasi_evaluasi_detik": round(time.time() - start, 4),
        }

    ringkasan_path = os.path.join(EVAL_RESULTS_PATH, "ringkasan_evaluasi.json")
    with open(ringkasan_path, "w", encoding="utf-8") as f:
        json.dump(ringkasan, f, ensure_ascii=False, indent=2)

    print(f"\n=== Ringkasan evaluasi disimpan di: {ringkasan_path} ===")
    print(json.dumps(ringkasan, indent=2, ensure_ascii=False))

    return ringkasan

if __name__ == "__main__":
    run_full_evaluation(limit_data=5)