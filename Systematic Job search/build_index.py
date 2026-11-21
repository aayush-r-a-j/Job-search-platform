import faiss
from sentence_transformers import SentenceTransformer

from config import INDEX_DIR, INDEX_FILE, MODEL_NAME
from jobs_data import job_to_text, load_jobs


def build_index():
    jobs = load_jobs()
    texts = [job_to_text(job) for _, job in jobs.iterrows()]

    model = SentenceTransformer(MODEL_NAME)
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    INDEX_DIR.mkdir(exist_ok=True)
    faiss.write_index(index, str(INDEX_FILE))

    print(f"Indexed {index.ntotal} jobs. Each job is a vector of {vectors.shape[1]} numbers.")
    print(f"Saved index to {INDEX_FILE}")


if __name__ == "__main__":
    build_index()
