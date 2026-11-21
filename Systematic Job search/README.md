# Job Search Platform

Find jobs by **meaning**, not just matching words. Search with plain English or upload a resume
to get the best-matching jobs and your skill gap for each one.

Keyword search fails when the user and the job post use different words: *"protect a company
from hackers"* shares no words with *"monitor systems for cyber attacks"*. This project uses
sentence embeddings and vector search to match on meaning, and it measures the improvement
against a BM25 keyword baseline.

## How it works

```
Offline (once):   jobs.csv ─► text per job ─► sentence-transformer ─► 384-d vectors ─► FAISS index
Online (query):   query / resume ─► embedding ─► FAISS top-k (cosine similarity) ─► ranked jobs
```

- **Embeddings:** `all-MiniLM-L6-v2` (sentence-transformers), normalized so inner product = cosine similarity
- **Vector search:** FAISS `IndexFlatIP` (exact search)
- **Keyword baseline:** BM25 (`rank-bm25`)
- **Resume matching:** PDF text extraction (`pypdf`) → 120-word chunks → averaged chunk embeddings → search
- **Skill gap:** whole-word matching of each job's skills against the resume text
- **UI:** Streamlit, with a side-by-side semantic vs keyword comparison
- **Find real openings:** each job card links to a LinkedIn and Indeed search for that role and
  city (the sample companies are fictional). If the data has an `apply_url` column, a direct
  "Apply on company site" button is shown instead.

## Results

Evaluated on 20 hand-labelled queries (`data/eval_queries.json`), top 5 results:

| Method | Recall@5 | MRR |
|---|---|---|
| Keyword (BM25) | 0.86 | 0.95 |
| Semantic (embeddings) | 0.97 | 1.00 |

Semantic search wins on queries that use different words from the job post. For example,
*"protect a company from hackers"*: BM25 finds 0 of the 2 security jobs (it matches the
word "company" instead), while semantic search ranks both at the top.

The test set is small and self-labelled, so treat these numbers as indicative.

## Run it

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt

python build_index.py           # embed all jobs and save the FAISS index
python search_engine.py make mobile apps   # quick terminal test
python evaluate.py              # Recall@5 and MRR, BM25 vs semantic
streamlit run app.py            # web app at http://localhost:8501
```

The first run downloads the embedding model (~90 MB).

## Project structure

| File | Role |
|---|---|
| `config.py` | Paths and settings |
| `jobs_data.py` | Loads jobs, builds the text that gets embedded, builds job-board links |
| `build_index.py` | Embeds all jobs and saves the FAISS index |
| `search_engine.py` | Semantic search, BM25 search, resume matching |
| `resume_parser.py` | PDF text extraction, chunking, skill overlap |
| `evaluate.py` | Recall@5 / MRR evaluation |
| `app.py` | Streamlit UI |
| `data/jobs.csv` | 55 sample job postings (fictional companies) |
| `data/eval_queries.json` | 20 labelled test queries |

## Next steps

- Hybrid search (BM25 + embeddings with Reciprocal Rank Fusion)
- Cross-encoder re-ranking of the top results
- Location and experience filters
- Live job data from a job-listing API, with direct apply links
- An LLM agent that explains each match (Project 2)
