import json

from config import EVAL_FILE
from search_engine import JobSearchEngine

K = 5


def recall_at_k(result_ids, relevant_ids):
    found = len(set(result_ids) & relevant_ids)
    return found / len(relevant_ids)


def reciprocal_rank(result_ids, relevant_ids):
    for rank, job_id in enumerate(result_ids, start=1):
        if job_id in relevant_ids:
            return 1 / rank
    return 0.0


def evaluate():
    with open(EVAL_FILE, encoding="utf-8") as f:
        queries = json.load(f)

    engine = JobSearchEngine()
    methods = {
        "Keyword (BM25)": engine.keyword_search,
        "Semantic": engine.semantic_search,
    }

    summary = {}
    per_query = {item["query"]: {} for item in queries}

    for name, search in methods.items():
        recalls, reciprocal_ranks = [], []
        for item in queries:
            relevant = set(item["relevant"])
            result_ids = [job["job_id"] for job in search(item["query"], K)]
            recall = recall_at_k(result_ids, relevant)
            recalls.append(recall)
            reciprocal_ranks.append(reciprocal_rank(result_ids, relevant))
            per_query[item["query"]][name] = recall
        summary[name] = (sum(recalls) / len(recalls), sum(reciprocal_ranks) / len(reciprocal_ranks))

    print(f"\nRecall@{K} for each query (1.00 = found every relevant job)\n")
    print(f"{'Query':<66}{'BM25':>6}{'Semantic':>10}")
    for query, scores in per_query.items():
        print(f"{query:<66}{scores['Keyword (BM25)']:>6.2f}{scores['Semantic']:>10.2f}")

    print(f"\nOverall on {len(queries)} labelled queries\n")
    for name, (recall, mrr) in summary.items():
        print(f"{name:<16} Recall@{K} = {recall:.2f}    MRR = {mrr:.2f}")


if __name__ == "__main__":
    evaluate()
