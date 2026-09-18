#!/usr/bin/env python3
"""Q2: BM25 lexical candidate generation for MIND-small and EB-NeRD demo."""
from __future__ import annotations
import argparse, json, re
from collections import Counter, defaultdict
from math import log
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm

TOKEN = re.compile(r"\b\w+\b", re.UNICODE)
STOPWORDS = frozenset("a an and are as at be by for from has have in is it of on or that the to was were with".split())

def tokenize(text: str) -> list[str]:
    return [word for word in TOKEN.findall((text or "").lower()) if len(word) > 1 and word not in STOPWORDS]

class InvertedBM25:
    """BM25 backed by postings lists: score only documents sharing query words."""
    def __init__(self, ids: list[str], texts: list[str], k1: float = 1.5, b: float = .75):
        self.ids, self.k1, self.b = ids, k1, b; self.lengths = []; self.postings = defaultdict(list)
        for doc_index, text in enumerate(texts):
            terms = Counter(tokenize(text)); self.lengths.append(sum(terms.values()))
            for term, frequency in terms.items(): self.postings[term].append((doc_index, frequency))
        self.average_length = max(sum(self.lengths) / len(self.lengths), 1); self.n_docs = len(ids)

    def search(self, query: str, k: int, excluded: set[str] | None = None) -> list[str]:
        scores: dict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):
            postings = self.postings.get(term, []); df = len(postings)
            if not df: continue
            idf = log(1 + (self.n_docs - df + .5) / (df + .5))
            for index, tf in postings:
                denominator = tf + self.k1 * (1 - self.b + self.b * self.lengths[index] / self.average_length)
                scores[index] += idf * tf * (self.k1 + 1) / denominator
        excluded = excluded or set()
        return [self.ids[index] for index, _ in sorted(scores.items(), key=lambda item: -item[1]) if self.ids[index] not in excluded][:k]

    def candidate_scores(self, query: str, candidate_ids: list[str]) -> dict[str, float]:
        """Return lexical scores only for articles displayed in an impression."""
        # wanted = {article: i for i, article in enumerate(self.ids) if article in set(candidate_ids)}
        wanted = set(candidate_ids)
        scores = {article: 0.0 for article in candidate_ids}
        max_df = 0.2 * self.n_docs  
        for term in set(tokenize(query)):
            postings = self.postings.get(term, []); df = len(postings)
            if not df or df > max_df: continue
            idf = log(1 + (self.n_docs - df + .5) / (df + .5))
            for index, tf in postings:
                article = self.ids[index]
                if article not in wanted: continue
                denominator = tf + self.k1 * (1 - self.b + self.b * self.lengths[index] / self.average_length)
                scores[article] += idf * tf * (self.k1 + 1) / denominator
        return scores

def query_from_history(history: list[object], title_by_id: dict[str, str], recent: int) -> str:
    return " ".join(title_by_id.get(str(article), "") for article in history[-recent:])

def evaluate(store: Path, dataset: str, split: str, recent: int, limit: int) -> dict[str, object]:
    articles = pd.read_parquet(store / dataset / "articles.parquet")
    impressions = pd.read_parquet(store / dataset / f"{split}_impressions.parquet")
    if limit: impressions = impressions.head(limit)
    ids = articles.article_id.astype(str).tolist(); title_by_id = dict(zip(ids, articles.title.fillna("")))
    text = (articles.title.fillna("") + " " + articles.abstract.fillna("")).tolist(); index = InvertedBM25(ids, text)
    sums = {50: 0.0, 100: 0.0, 200: 0.0}; evaluated = 0
    for _, row in tqdm(impressions.iterrows(), total=len(impressions), desc=f"BM25 {dataset}", unit="impression"):
        clicked = {str(article) for article in row.clicked_ids}
        history = [str(article) for article in row.history_ids]
        query = query_from_history(history, title_by_id, recent)
        if not clicked or not query: continue
        retrieved = index.search(query, 200, excluded=set(history))
        for k in sums: sums[k] += len(clicked.intersection(retrieved[:k])) / len(clicked)
        evaluated += 1
    return {"dataset": dataset, "split": split, "retrieval_corpus_articles": len(ids), "evaluated_impressions": evaluated, "history_articles_used": recent, **{f"recall@{k}": sums[k] / evaluated if evaluated else 0.0 for k in sums}}

def candidate_generator_coverage(store: Path, dataset: str, split: str, recent: int, k: int, limit: int) -> dict[str, object]:
    articles = pd.read_parquet(store / dataset / "articles.parquet")
    impressions = pd.read_parquet(store / dataset / f"{split}_impressions.parquet")
    if limit: impressions = impressions.head(limit)
    ids = articles.article_id.astype(str).tolist()
    title_by_id = dict(zip(ids, articles.title.fillna("")))
    text = (articles.title.fillna("") + " " + articles.abstract.fillna("")).tolist()
    index = InvertedBM25(ids, text)
    covered, total_candidates, evaluated = 0, 0, 0
    for _, row in tqdm(impressions.iterrows(), total=len(impressions), desc=f"Coverage {dataset}", unit="impression"):
        history = [str(article) for article in row.history_ids]
        query = query_from_history(history, title_by_id, recent)
        candidates = {str(c) for c in row.candidate_ids}
        if not query or not candidates: continue
        retrieved = set(index.search(query, k))
        covered += len(candidates & retrieved)
        total_candidates += len(candidates)
        evaluated += 1
    return {
        "dataset": dataset, "split": split, "k": k, "evaluated_impressions": evaluated,
        "candidate_set_coverage": covered / total_candidates if total_candidates else 0.0,
    }

def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--store", type=Path, default=Path("data/processed")); parser.add_argument("--split", default="validation", choices=["validation", "test"]); parser.add_argument("--recent-history", type=int, default=5); parser.add_argument("--k", type=int, default=150); parser.add_argument("--limit", type=int, default=0, help="0 evaluates the complete selected split"); parser.add_argument("--output", type=Path, default=Path("outputs/q2_bm25_metrics.json")); parser.add_argument("--coverage-output", type=Path, default=Path("outputs/q2_candidate_coverage.json")); args = parser.parse_args()
    results = [evaluate(args.store, dataset, args.split, args.recent_history, args.limit) for dataset in ("mind", "ebnerd")]
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(results, indent=2))
    for result in results: print(result)

    coverage = [candidate_generator_coverage(args.store, dataset, args.split, args.recent_history, args.k, args.limit) for dataset in ("mind", "ebnerd")]
    args.coverage_output.parent.mkdir(parents=True, exist_ok=True); args.coverage_output.write_text(json.dumps(coverage, indent=2))
    for result in coverage: print(result)
    
if __name__ == "__main__": main()
