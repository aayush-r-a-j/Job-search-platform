from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "data" / "jobs.csv"
EVAL_FILE = BASE_DIR / "data" / "eval_queries.json"

INDEX_DIR = BASE_DIR / "index"
INDEX_FILE = INDEX_DIR / "jobs.faiss"

MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5
