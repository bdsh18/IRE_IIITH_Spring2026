from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from bm25_retrieval import InvertedBM25, query_from_history
from features import (
    FEATURE_NAMES,
    article_metadata,
    candidate_features,
    load_embeddings,
    train_popularity,
    user_session_click_count,
    user_session_order,
)
from reranker import build_dataset_features, train_model


def corpus_and_index_size(store: Path, dataset: str) -> dict:
    articles = pd.read_parquet(store / dataset / "articles.parquet")
    ids = articles.article_id.astype(str).tolist()
    text = (articles.title.fillna("") + " " + articles.abstract.fillna("")).tolist()
    index = InvertedBM25(ids, text)

    def parquet_path(split: str) -> Path:
        return store / dataset / f"{split}_impressions.parquet"

    def count_rows(split: str) -> int:
        path = parquet_path(split)
        return len(pd.read_parquet(path, columns=["impression_id"])) if path.exists() else 0

    def bytes_on_disk(path: Path) -> int:
        return path.stat().st_size if path.exists() else 0

    corpus_text_bytes = sum(len(t.encode("utf-8")) for t in text)
    postings_bytes_approx = sum(len(postings) for postings in index.postings.values()) * 56  # (doc_idx int64, tf float64) + object overhead, rough

    positions, vectors = load_embeddings(store, dataset)
    embeddings_path = store / dataset / "article_embeddings.npz"
    embedding_bytes_on_disk = bytes_on_disk(embeddings_path)
    embedding_bytes_in_memory = int(vectors.nbytes) if vectors is not None else 0

    articles_path = store / dataset / "articles.parquet"
    feature_store_bytes_on_disk = bytes_on_disk(articles_path) + sum(bytes_on_disk(parquet_path(split)) for split in ("train", "validation", "test"))
    from features import build_features  # local import: avoids a hard dependency for callers that only need sizing, not a full feature build
    sample_features = build_features(store, dataset, "validation")
    feature_table_bytes_in_memory_per_1k_rows = int(sample_features.memory_usage(deep=True).sum() / max(len(sample_features), 1) * 1000)

    return {
        "corpus": {
            "catalog_articles": len(ids),
            "raw_title_abstract_bytes": corpus_text_bytes,
            "train_impressions": count_rows("train"),
            "validation_impressions": count_rows("validation"),
            "test_impressions": count_rows("test"),
        },
        "index": {
            "bm25_vocabulary_terms": len(index.postings),
            "bm25_postings_bytes_approx": postings_bytes_approx,
            "embedding_dimensions": int(vectors.shape[1]) if vectors is not None else 0,
            "embedding_index_bytes_in_memory": embedding_bytes_in_memory,
            "embedding_index_bytes_on_disk_compressed": embedding_bytes_on_disk,
        },
        "feature_store": {
            "parquet_bytes_on_disk": feature_store_bytes_on_disk,
            "validation_feature_rows": len(sample_features),
            "feature_table_bytes_in_memory_per_1k_rows": feature_table_bytes_in_memory_per_1k_rows,
        },
    }


def latency_benchmark(store: Path, dataset: str, model, columns: list[str], repeats: int) -> dict:
    articles = pd.read_parquet(store / dataset / "articles.parquet")
    ids = articles.article_id.astype(str).tolist()
    titles = dict(zip(ids, articles.title.fillna("")))
    text = (articles.title.fillna("") + " " + articles.abstract.fillna("")).tolist()
    index = InvertedBM25(ids, text)

    category_by_id, topics_by_id, published_by_id = article_metadata(articles)
    positions, vectors = load_embeddings(store, dataset)

    train_impressions = pd.read_parquet(store / dataset / "train_impressions.parquet")
    popularity = train_popularity(train_impressions)
    max_pop = max(popularity.values(), default=1)

    impressions = pd.read_parquet(store / dataset / "validation_impressions.parquet")
    session_rank = user_session_order(impressions)  # O(1) dict lookup per request below; the sort itself is a batch/offline cost, not per-request
    session_clicks = user_session_click_count(impressions)
    sample = impressions.sample(min(repeats, len(impressions)), random_state=0)

    durations = []
    benchmark_start = time.perf_counter()
    for row in sample.itertuples(index=False):
        start = time.perf_counter()
        history_ids = list(row.history_ids)
        weights = list(row.history_recency_weights)
        query = query_from_history(history_ids, titles, 5)
        candidates = index.search(query, 200, excluded=set(map(str, history_ids)))
        candidate_scores = index.candidate_scores(query, candidates)
        session_count = session_rank.get(str(row.impression_id), 0)
        session_click_count = session_clicks.get(str(row.impression_id), 0)

        feature_rows = [
            candidate_features(
                article, position, history_ids, weights, category_by_id, topics_by_id,
                popularity, max_pop, positions, vectors, published_by_id, row.timestamp,
                session_count, session_click_count,
            ) | {"bm25_score": candidate_scores.get(article, 0.0)}
            for position, article in enumerate(candidates)
        ]
        feature_matrix = pd.DataFrame(feature_rows)
        if len(feature_matrix):
            model.predict(feature_matrix[columns])
        durations.append(time.perf_counter() - start)
    benchmark_wall_seconds = time.perf_counter() - benchmark_start

    durations = np.asarray(durations) if durations else np.asarray([0.0])
    measured_qps = round(len(durations) / benchmark_wall_seconds, 2) if benchmark_wall_seconds > 0 else 0.0
    return {
        "n_requests": len(durations),
        "benchmark_wall_seconds": round(benchmark_wall_seconds, 3),
        "measured_single_process_qps": measured_qps,
        "p50_ms": round(float(np.percentile(durations, 50) * 1000), 3),
        "p95_ms": round(float(np.percentile(durations, 95) * 1000), 3),
        "p99_ms": round(float(np.percentile(durations, 99) * 1000), 3),
        "mean_ms": round(float(durations.mean() * 1000), 3),
    }


def cost_estimate(p99_ms: float, target_sla_ms: float, single_machine_qps: float, machine_cost_per_hour: float, target_qps: int = 1000) -> dict:
    machines_needed = int(max(1, -(-target_qps // max(single_machine_qps, 1))))  # ceil division
    return {
        "measured_p99_ms": p99_ms,
        "target_sla_ms": target_sla_ms,
        "meets_sla_single_machine": p99_ms <= target_sla_ms,
        "assumed_single_machine_qps": single_machine_qps,
        "target_qps": target_qps,
        "machines_needed": machines_needed,
        "estimated_cost_per_hour_usd": round(machines_needed * machine_cost_per_hour, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=Path, default=Path("data/processed"))
    parser.add_argument("--dataset", choices=["mind", "ebnerd"], required=True)
    parser.add_argument("--limit", type=int, default=300, help="requests to benchmark for latency/throughput")
    parser.add_argument("--target-sla-ms", type=float, default=100.0)
    parser.add_argument("--machine-cost-per-hour", type=float, default=0.5, help="e.g. a small cloud VM")
    parser.add_argument("--output", type=Path, default=Path("outputs/q4_serving_scale.json"))
    args = parser.parse_args()

    columns = FEATURE_NAMES + ["bm25_score"]
    train_features = build_dataset_features(args.store, args.dataset, "train", 0)
    model = train_model(train_features, columns)

    size = corpus_and_index_size(args.store, args.dataset)
    latency = latency_benchmark(args.store, args.dataset, model, columns, args.limit)
    cost = cost_estimate(latency["p99_ms"], args.target_sla_ms, latency["measured_single_process_qps"], args.machine_cost_per_hour)

    result = {
        "dataset": args.dataset,
        "corpus_size": size["corpus"],
        "index_size": size["index"],
        "feature_store_size": size["feature_store"],
        "latency": latency,
        "cost_estimate": cost,
        "ten_x_scaling_notes": [
            "BM25 postings and the brute-force cosine embedding index both live in a single process; "
            "at 10x catalog size the embedding index needs FAISS/HNSW ANN rather than a full matrix-vector product per request.",
            "Feature build currently re-reads the full history/popularity tables per process; at 10x traffic "
            "this needs a precomputed, versioned feature store (e.g. Feast/Redis) instead of in-request pandas joins.",
            "The GBDT model.predict call is the main per-request cost once candidates are pooled — batching "
            "requests or moving to a served model endpoint (Triton/TorchServe) amortizes this at higher QPS.",
            "Throughput above is single-process and sequential; real serving needs concurrent workers "
            "(multiprocessing or async I/O) — measured_single_process_qps is a per-worker floor, not a ceiling.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()