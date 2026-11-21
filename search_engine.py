import re
import sys

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from config import INDEX_FILE, MODEL_NAME, TOP_K
from jobs_data import job_to_text, load_jobs
from resume_parser import split_into_chunks


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


class JobSearchEngine:
    def __init__(self):
        if not INDEX_FILE.exists():
            raise FileNotFoundError("Index not found. Run first: python build_index.py")

        self.jobs = load_jobs()
        self.model = SentenceTransformer(MODEL_NAME)
        self.index = faiss.read_index(str(INDEX_FILE))

        if self.index.ntotal != len(self.jobs):
            raise ValueError("jobs.csv has changed since the index was built. Run: python build_index.py")

        texts = [job_to_text(job) for _, job in self.jobs.iterrows()]
        self.bm25 = BM25Okapi([tokenize(text) for text in texts])

    def semantic_search(self, query, k=TOP_K):
        query_vector = self.model.encode([query], normalize_embeddings=True)
        scores, positions = self.index.search(query_vector, k)
        return self._to_results(positions[0], scores[0])

    def keyword_search(self, query, k=TOP_K):
        scores = self.bm25.get_scores(tokenize(query))
        positions = [p for p in np.argsort(scores)[::-1][:k] if scores[p] > 0]
        return self._to_results(positions, [scores[p] for p in positions])

    def match_resume(self, resume_text, k=TOP_K):
        chunks = split_into_chunks(resume_text)
        chunk_vectors = self.model.encode(chunks, normalize_embeddings=True)

        resume_vector = chunk_vectors.mean(axis=0, keepdims=True)
        resume_vector = resume_vector / np.linalg.norm(resume_vector)

        scores, positions = self.index.search(resume_vector.astype("float32"), k)
        return self._to_results(positions[0], scores[0])

    def _to_results(self, positions, scores):
        results = []
        for position, score in zip(positions, scores):
            job = self.jobs.iloc[int(position)].to_dict()
            job["score"] = round(float(score), 3)
            results.append(job)
        return results


def print_results(title, results):
    print(title)
    if not results:
        print("  (no results)")
    for job in results:
        print(f"  {job['score']:>6.3f}  {job['job_id']}  {job['title']} - {job['company']}")
    print()


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or "machine learning internship for freshers"
    engine = JobSearchEngine()

    print(f"\nQuery: {query}\n")
    print_results("Semantic search (meaning):", engine.semantic_search(query))
    print_results("Keyword search (BM25, exact words):", engine.keyword_search(query))
