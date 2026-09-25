from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from bm25_retrieval import InvertedBM25, query_from_history
from features import (
    article_metadata,
    candidate_features_from_context,
    prepare_candidate_context,
    load_embeddings,
    ranker_feature_names,
    train_popularity,
    weighted_user_vector,
)
from reranker import build_dataset_features, predict_scores, train_model

try:
    import faiss
except ImportError:  # Report the dense fallback only when FAISS is not installed.
    faiss = None

def build_ann_index(vectors: np.ndarray | None):
    if vectors is None or not len(vectors):
        return None, 0, "unavailable"
    if faiss is None:
        return None, int(vectors.nbytes), "dense_exact_fallback (install faiss-cpu for HNSW)"
    index = faiss.IndexHNSWFlat(vectors.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efSearch = 64
    index.add(np.ascontiguousarray(vectors))
    return index, len(faiss.serialize_index(index)), "FAISS IndexHNSWFlat(M=32, efSearch=64)"


def corpus_and_index_size(store: Path, dataset: str) -> dict:
    articles = pd.read_parquet(store / dataset / "articles.parquet")
    ids = articles.article_id.astype(str).tolist()
    text = (articles.title.fillna("") + " " + articles.abstract.fillna("")).tolist()
    index_build_start = time.perf_counter()
    index = InvertedBM25(ids, text)
    bm25_index_seconds = time.perf_counter() - index_build_start

    def parquet_path(split: str) -> Path:
        return store / dataset / f"{split}_impressions.parquet"

    def count_rows(split: str) -> int:
        path = parquet_path(split)
        return len(pd.read_parquet(path, columns=["impression_id"])) if path.exists() else 0

    def bytes_on_disk(path: Path) -> int:
        return path.stat().st_size if path.exists() else 0

    corpus_text_bytes = sum(len(t.encode("utf-8")) for t in text)
    total_postings = sum(len(postings) for postings in index.postings.values())
    postings_bytes_approx = total_postings * 56

    embedding_load_start = time.perf_counter()
    positions, vectors = load_embeddings(store, dataset)
    embedding_load_seconds = time.perf_counter() - embedding_load_start
    embeddings_path = store / dataset / "article_embeddings.npz"
    embedding_bytes_on_disk = bytes_on_disk(embeddings_path)
    embedding_bytes_in_memory = int(vectors.nbytes) if vectors is not None else 0
    _ann, ann_serialized_bytes, ann_type = build_ann_index(vectors)

    # Include the base article/impression Parquet files and cached candidate-level
    # feature tables produced for the re-ranker.
    feature_store_bytes_on_disk = sum(path.stat().st_size for path in (store / dataset).glob("*.parquet"))
    from features import build_features
    sample_features = build_features(store, dataset, "validation")
    feature_table_bytes_in_memory_per_1k_rows = int(sample_features.memory_usage(deep=True).sum() / max(len(sample_features), 1) * 1000)

    return {
        "corpus": {
            "catalog_articles": len(ids),
            "raw_title_abstract_bytes": corpus_text_bytes,
            "bm25_avg_doc_length_tokens": round(index.average_length, 2),
            "train_impressions": count_rows("train"),
            "validation_impressions": count_rows("validation"),
            "test_impressions": count_rows("test"),
        },
        "index": {
            "bm25_vocabulary_terms": len(index.postings),
            "bm25_total_postings": total_postings,
            "bm25_postings_bytes_approx": postings_bytes_approx,
            "bm25_index_build_seconds": round(bm25_index_seconds, 4),
            "bm25_index_throughput_docs_per_sec": round(len(ids) / max(bm25_index_seconds, 1e-9), 1),
            "embedding_dimensions": int(vectors.shape[1]) if vectors is not None else 0,
            "embedding_index_bytes_in_memory": embedding_bytes_in_memory,
            "embedding_index_bytes_on_disk_compressed": embedding_bytes_on_disk,
            "embedding_index_load_seconds": round(embedding_load_seconds, 4),
            "embedding_index_load_throughput_vectors_per_sec": round((vectors.shape[0] if vectors is not None else 0) / max(embedding_load_seconds, 1e-9), 1),
            "ann_index_type": ann_type,
            "ann_index_serialized_bytes": ann_serialized_bytes,
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
    sample = impressions.sample(min(repeats, len(impressions)), random_state=0)

    durations, query_lengths, candidate_counts = [], [], []
    benchmark_start = time.perf_counter()
    for row in tqdm(sample.itertuples(index=False), total=len(sample), desc="Latency benchmark", unit="request"):
        start = time.perf_counter()
        history_ids = list(row.history_ids)
        weights = list(row.history_recency_weights)
        query = query_from_history(history_ids, titles, 5)
        candidates = index.search(query, 200, excluded=set(map(str, history_ids)))
        query_lengths.append(len(query.split()))
        candidate_counts.append(len(candidates))
        candidate_scores = index.candidate_scores(query, candidates)
        # The benchmark is deliberately submission-matched.  Competition test
        # data has no preceding within-session trace, so these values cannot be
        # supplied at request time.
        context = prepare_candidate_context(
            history_ids, weights, category_by_id, topics_by_id,
            popularity, max_pop, positions, vectors,
        )
        feature_rows = [
            candidate_features_from_context(
                article, position, context, category_by_id, topics_by_id,
                published_by_id, row.timestamp, 0, 0, 0.0,
            ) | {"bm25_score": candidate_scores.get(article, 0.0)}
            for position, article in enumerate(candidates)
        ]
        feature_matrix = pd.DataFrame(feature_rows)
        if len(feature_matrix):
            predict_scores(model, feature_matrix[columns])
        durations.append(time.perf_counter() - start)
    benchmark_wall_seconds = time.perf_counter() - benchmark_start

    durations = np.asarray(durations) if durations else np.asarray([0.0])
    measured_qps = round(len(durations) / benchmark_wall_seconds, 2) if benchmark_wall_seconds > 0 else 0.0
    mean_seconds = float(durations.mean())
    predicted_qps_from_littles_law = round(1 / mean_seconds, 2) if mean_seconds > 0 else 0.0
    return {
        "n_requests": len(durations),
        "benchmark_wall_seconds": round(benchmark_wall_seconds, 3),
        "measured_single_process_qps": measured_qps,
        "p50_ms": round(float(np.percentile(durations, 50) * 1000), 3),
        "p95_ms": round(float(np.percentile(durations, 95) * 1000), 3),
        "p99_ms": round(float(np.percentile(durations, 99) * 1000), 3),
        "mean_ms": round(float(mean_seconds * 1000), 3),
        "avg_query_length_tokens": round(float(np.mean(query_lengths)), 2) if query_lengths else 0.0,
        "avg_candidates_retrieved": round(float(np.mean(candidate_counts)), 1) if candidate_counts else 0.0,
        "littles_law_check": {
            "N_concurrent_requests": 1,
            "predicted_qps_1_over_mean_latency": predicted_qps_from_littles_law,
            "measured_qps": measured_qps,
            "holds": measured_qps <= predicted_qps_from_littles_law * 1.05,
        },
    }


def semantic_latency_benchmark(store: Path, dataset: str, repeats: int) -> dict:
    positions, vectors = load_embeddings(store, dataset)
    ann_index, _ann_bytes, ann_type = build_ann_index(vectors)
    impressions = pd.read_parquet(store / dataset / "validation_impressions.parquet")
    sample = impressions.sample(min(repeats, len(impressions)), random_state=0)

    durations = []
    benchmark_start = time.perf_counter()
    for row in tqdm(sample.itertuples(index=False), total=len(sample), desc="Semantic latency", unit="request"):
        start = time.perf_counter()
        history_ids = list(row.history_ids)
        weights = list(row.history_recency_weights)
        if vectors is not None:
            user = weighted_user_vector(history_ids, weights, positions, vectors)
            if user is not None:
                if ann_index is not None:
                    _ = ann_index.search(np.ascontiguousarray(user[None, :].astype(np.float32)), 200)
                else:
                    _ = vectors @ user
        durations.append(time.perf_counter() - start)
    benchmark_wall_seconds = time.perf_counter() - benchmark_start

    durations = np.asarray(durations) if durations else np.asarray([0.0])
    measured_qps = round(len(durations) / benchmark_wall_seconds, 2) if benchmark_wall_seconds > 0 else 0.0
    return {
        "index_type": ann_type,
        "n_requests": len(durations),
        "measured_single_process_qps": measured_qps,
        "p50_ms": round(float(np.percentile(durations, 50) * 1000), 3),
        "p95_ms": round(float(np.percentile(durations, 95) * 1000), 3),
        "p99_ms": round(float(np.percentile(durations, 99) * 1000), 3),
        "mean_ms": round(float(durations.mean() * 1000), 3),
    }


def cost_estimate(p99_ms: float, target_sla_ms: float, single_machine_qps: float, machine_cost_per_hour: float, target_qps: int = 1000) -> dict:
    machines_needed = int(max(1, -(-target_qps // max(single_machine_qps, 1))))
    cost_per_hour = machines_needed * machine_cost_per_hour
    queries_per_hour = target_qps * 3600
    return {
        "measured_p99_ms": p99_ms,
        "target_sla_ms": target_sla_ms,
        "meets_sla_single_machine": p99_ms <= target_sla_ms,
        "assumed_single_machine_qps": single_machine_qps,
        "target_qps": target_qps,
        "machines_needed": machines_needed,
        "estimated_cost_per_hour_usd": round(cost_per_hour, 2),
        "estimated_cost_per_1000_queries_usd": round(cost_per_hour / queries_per_hour * 1000, 5),
        "note": (
            "Adding machines raises throughput but does not lower per-request latency; "
            "the cost figure is only valid at the SLA once single-request p99 is below the target."
        ),
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

    # Measure the same column set used by both large-test submission scripts.
    columns = ranker_feature_names(serving_only=True) + ["bm25_score"]
    train_features = build_dataset_features(args.store, args.dataset, "train", 0)
    model = train_model(train_features, columns)

    size = corpus_and_index_size(args.store, args.dataset)
    latency = latency_benchmark(args.store, args.dataset, model, columns, args.limit)
    semantic_latency = semantic_latency_benchmark(args.store, args.dataset, args.limit)
    cost = cost_estimate(latency["p99_ms"], args.target_sla_ms, latency["measured_single_process_qps"], args.machine_cost_per_hour)

    result = {
        "dataset": args.dataset,
        "feature_policy": "submission_matched_serving_only",
        "feature_columns": columns,
        "corpus_size": size["corpus"],
        "index_size": size["index"],
        "feature_store_size": size["feature_store"],
        "latency_end_to_end_lexical_plus_rerank": latency,
        "latency_semantic_scoring_only": semantic_latency,
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