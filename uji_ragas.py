"""
Uji signifikansi statistik perbandingan 4 teknik chunking (Fixed-Size, Recursive,
Sentence-Aware, Semantic) pada metrik RAGAS dari 4 file JSON terpisah.
"""

import pandas as pd
import json
from scipy.stats import wilcoxon, friedmanchisquare, shapiro
import scikit_posthocs as sp
from itertools import combinations

# ============ CONFIG - sesuaikan dengan data kamu ============
# Mapping nama teknik ke nama file JSON masing-masing
JSON_FILES = {
    "fixed_size": "evaluation_results/detail_fixed.json",
    "recursive": "evaluation_results/detail_recursive.json",
    "sentence_aware": "evaluation_results/detail_sentence.json",
    "semantic": "evaluation_results/detail_semantic.json"
}

METRICS = ["context_precision", "context_recall", "faithfulness", "answer_correctness"]
ALPHA = 0.05
# ===============================================================


def load_wide_from_jsons(metric):
    """
    Membaca 4 JSON, mengekstrak 'user_input' dan metrik yang diuji, 
    lalu menggabungkannya ke format wide (baris = pertanyaan, kolom = teknik).
    """
    dfs = []
    
    for tech, filepath in JSON_FILES.items():
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Ekstrak data yang dibutuhkan saja
        records = []
        for item in data:
            records.append({
                "user_input": item["user_input"],
                tech: item[metric]
            })
            
        # Buat DataFrame per teknik dan jadikan user_input sebagai index
        df_tech = pd.DataFrame(records)
        # Menghapus duplikat pertanyaan jika ada agar join tidak error
        df_tech = df_tech.drop_duplicates(subset=["user_input"]) 
        df_tech.set_index("user_input", inplace=True)
        dfs.append(df_tech)

    # Gabungkan semua DataFrame berdampingan berdasarkan index (user_input)
    wide = pd.concat(dfs, axis=1)
    
    # Buang baris yang tidak memiliki nilai di salah satu teknik (tidak lengkap)
    wide = wide.dropna()
    
    # Pastikan urutan kolom sesuai dengan dictionary
    wide = wide[list(JSON_FILES.keys())]
    
    return wide


def check_normality(wide):
    """Shapiro-Wilk per kolom teknik - untuk info, membantu memilih uji parametrik/non-parametrik."""
    results = {}
    for tech in wide.columns:
        # Peringatan jika semua nilai sama (shapiro akan error/tidak valid)
        if len(wide[tech].unique()) == 1:
            results[tech] = 1.0 # Anggap normal/tidak relevan diuji
        else:
            stat, p = shapiro(wide[tech])
            results[tech] = p
    return results


def run_friedman(wide, metric):
    stat, p = friedmanchisquare(*[wide[t] for t in wide.columns])
    print(f"\n[{metric}] Friedman test: chi2={stat:.4f}, p={p:.4f}", end=" ")
    print("-> SIGNIFIKAN, ada perbedaan antar teknik" if p < ALPHA else "-> tidak signifikan, teknik dianggap setara")
    return p


def run_posthoc_nemenyi(wide, metric):
    result = sp.posthoc_nemenyi_friedman(wide.values)
    result.columns = wide.columns
    result.index = wide.columns
    print(f"\n[{metric}] Post-hoc Nemenyi (p-value antar pasangan teknik):")
    print(result.round(4))
    return result


def run_pairwise_wilcoxon(wide, metric):
    print(f"\n[{metric}] Wilcoxon signed-rank per pasangan:")
    rows = []
    for a, b in combinations(wide.columns, 2):
        d = wide[a] - wide[b]
        if (d == 0).all():
            print(f"  {a} vs {b}: SKIPPED (seluruh data identik, W tidak dapat dihitung)")
            continue
            
        stat, p = wilcoxon(wide[a], wide[b], zero_method='zsplit')
        mean_diff = d.mean()
        sig = "signifikan" if p < ALPHA else "tidak signifikan"
        pemenang = a if mean_diff > 0 else b
        print(f"  {a} vs {b}: W={stat:.2f}, p={p:.4f}, selisih rata-rata={mean_diff:+.4f} "
              f"({sig}, cenderung unggul: {pemenang if p < ALPHA else '-'})")
        rows.append({"metric": metric, "a": a, "b": b, "W": stat, "p_value": p,
                      "mean_diff": mean_diff, "significant": p < ALPHA})
    return rows


def main():
    all_pairwise_results = []

    for metric in METRICS:
        print("=" * 80)
        print(f"METRIK: {metric}")
        print("=" * 80)

        # Load data langsung dari 4 JSON
        wide = load_wide_from_jsons(metric)
        print(f"Jumlah pertanyaan lengkap (beririsan di semua teknik) dipakai: {len(wide)}")

        if len(wide) == 0:
            print(f"SKIPPED {metric}: Tidak ada data yang cocok/lengkap.")
            continue

        normality = check_normality(wide)
        print("Shapiro-Wilk p-value per teknik (p < 0.05 => TIDAK normal):")
        for tech, p in normality.items():
            print(f"  {tech}: p={p:.4f}")

        # Hanya jalankan uji jika jumlah sampel mendukung
        if len(wide) >= 3:
            friedman_p = run_friedman(wide, metric)
            if friedman_p < ALPHA:
                run_posthoc_nemenyi(wide, metric)

            pairwise = run_pairwise_wilcoxon(wide, metric)
            all_pairwise_results.extend(pairwise)
        else:
            print("Jumlah data terlalu sedikit (<3) untuk uji statistik.")

    summary = pd.DataFrame(all_pairwise_results)
    out_path = "hasil_uji_signifikansi_ragas.csv"
    if not summary.empty:
        summary.to_csv(out_path, index=False)
        print(f"\n\nRingkasan seluruh hasil uji pairwise disimpan ke: {out_path}")
    else:
        print("\n\nTidak ada uji yang berhasil dijalankan untuk disimpan.")


if __name__ == "__main__":
    main()