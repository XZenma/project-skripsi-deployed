from dotenv import load_dotenv
import os

load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "sailor2")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding")
FAISS_PATH = os.getenv("FAISS_PATH", "faiss_db")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
SEMANTIC_THRESHOLD = int(os.getenv("SEMANTIC_THRESHOLD", 95))
TOP_K = int(os.getenv("TOP_K", 3))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.0))

RAW_DATA_PATH = os.getenv("RAW_DATA_PATH", "data/raw")
LOADED_PATH = os.getenv("LOADED_PATH", "data/loaded")
CLEANED_PATH = os.getenv("CLEANED_PATH", "data/cleaned")
DEDUPLICATED_PATH = os.getenv("DEDUPLICATED_PATH", "data/deduplicated")
REMOVED_DUPLICATES_PATH = os.getenv("REMOVED_DUPLICATES_PATH", "data/removed_duplicates")
CHUNKS_PATH = os.getenv("CHUNKS_PATH", "data/chunks")

# bisa single atau all, bila single harus memiliki target matkul
LOAD_MODE = os.getenv("LOAD_MODE", "all") 
TARGET_MATKUL = os.getenv("TARGET_MATKUL", "")

SPACY_MODEL = "xx_sent_ud_sm"

FAISS_PATHS = {
    "fixed":    os.path.join(FAISS_PATH, "fixed"),
    "recursive": os.path.join(FAISS_PATH, "recursive"),
    "sentence": os.path.join(FAISS_PATH, "sentence"),
    "semantic": os.path.join(FAISS_PATH, "semantic"),
}

GROUND_TRUTH_PATH = os.getenv("GROUND_TRUTH_PATH", "Dataset_skripsi.json")
EVAL_RESULTS_PATH = os.getenv("EVAL_RESULTS_PATH", "evaluation_results")