import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from pipeline.embedder import get_embeddings, load_faiss
from config import MODEL_NAME, TOP_K, LLM_TEMPERATURE


def get_llm():
    return OllamaLLM(
        model=MODEL_NAME,
        temperature=LLM_TEMPERATURE
    )


def retrieve(query: str, teknik: str) -> list[dict]:
    """
    Ambil chunks paling relevan dari FAISS berdasarkan query.
    Return: list of dict (text + metadata + distance)
    """
    index, docstore = load_faiss(teknik)

    embeddings = get_embeddings()
    query_vector = np.array(
        [embeddings.embed_query(query)], dtype=np.float32
    )

    distances, indices = index.search(query_vector, TOP_K)

    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(docstore):
            entry = docstore[idx].copy()
            entry["distance"] = round(float(distances[0][i]), 4)
            results.append(entry)

    return results


def generate(query: str, context_chunks: list[str]) -> str:
    llm = get_llm()
    context = "\n\n".join(context_chunks)
    prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""[SYSTEM]
You are a professional and objective Academic Chatbot Assistant. Your primary goal is to help students or lecturers understand course materials based strictly on the provided documents.

[STRICT INSTRUCTIONS]
1. Write your entire answer in Bahasa Indonesia. If the context is in English or another language, translate the relevant information into Bahasa Indonesia before answering.
2. Base every claim strictly and only on the information explicitly stated within the <context> tags.
3. If the context contains no relevant information at all, respond with exactly this sentence and nothing else: "Maaf, informasi tersebut tidak tersedia dalam materi pembelajaran yang diberikan."
4. Present each fact directly, as it is literally stated. Treat separate pieces of context as independent facts, without combining them into a new conclusion, interpretation, or summary that is not explicitly written in the context.
5. Present the answer as plain, direct statements only, containing solely the requested factual content. Do not include any extra details, background information, or related facts from the context if they are not explicitly asked by the user.
6. Begin the answer directly with the requested information, and end it immediately once that information is complete.
7. Keep the answer to the requested content only, without citing or pointing the reader toward where the information came from.

[CONTEXT]
<context>
{context}
</context>

[QUESTION]
<question>
{question}
</question>

Read the question again: {question}

FINAL REMINDER before answering — follow every rule under [STRICT INSTRUCTIONS] above:
- Write in Bahasa Indonesia.
- Stay strictly grounded in the context, stated literally.
- Answer directly and completely, then stop.

[ANSWER]
"""
            )
    chain = prompt | llm
    return chain.invoke({"context": context, "question": query})


def run_rag(query: str, teknik: str) -> dict:
    """
    Fungsi utama: jalankan pipeline RAG lengkap.
    Return dict berisi jawaban, konteks, dan sumber dokumen.
    """
    print(f"Query: {query}")
    print(f"Teknik chunking: {teknik}")

    print("Retrieving chunks...")
    retrieved = retrieve(query, teknik)
    print(f"Chunks ditemukan: {len(retrieved)}")

    context_chunks = [r["text"] for r in retrieved]

    print("Generating jawaban...")
    jawaban = generate(query, context_chunks)

    return {
        "query": query,
        "teknik": teknik,
        "jawaban": jawaban,
        "konteks": context_chunks,
        "sumber": [
            {
                "filename": r.get("filename"),
                "matkul": r.get("matkul"),
                "index_chunk": r.get("index_chunk"),
                "panjang_karakter": r.get("panjang_karakter"),
                "distance": r.get("distance")
            }
            for r in retrieved
        ]
    }


if __name__ == "__main__":
    query = "Apa itu basis data relasional?"
    teknik = "fixed"

    hasil = run_rag(query, teknik)

    print("\n=== HASIL RAG ===")
    print(f"Pertanyaan: {hasil['query']}")
    print(f"Jawaban: {hasil['jawaban']}")
    print(f"\nKonteks yang digunakan:")
    for i, (ctx, src) in enumerate(zip(hasil['konteks'], hasil['sumber'])):
        print(f"\n[Chunk {i+1}] {src['filename']} ({src['matkul']})")
        print(f"  Distance: {src['distance']}")
        print(f"  Preview: {ctx[:200]}...")