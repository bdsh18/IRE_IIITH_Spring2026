from __future__ import annotations
import argparse
import math
import multiprocessing as mp
import os
import shutil
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm.auto import tqdm

from bm25_retrieval import InvertedBM25, query_from_history
from build_pipeline import recency_weights
from features import FEATURE_NAMES, SUBMISSION_TIME_UNAVAILABLE, weighted_user_vector
from reranker import build_dataset_features, train_model
from semantic_retrieval import ebnerd_embeddings

_STATE: dict = {}


def _init_worker(category_by_id, topics_by_id, published_by_id, titles, popularity, max_pop, positions, vectors, bm25_index):
    _STATE["category_by_id"] = category_by_id
    _STATE["topics_by_id"] = topics_by_id
    _STATE["published_by_id"] = published_by_id
    _STATE["titles"] = titles
    _STATE["popularity"] = popularity
    _STATE["max_pop"] = max_pop
    _STATE["positions"] = positions
    _STATE["vectors"] = vectors
    _STATE["bm25_index"] = bm25_index


def _score_one_impression(task: tuple[str, object, list[str], list[str]]) -> tuple[str, list[str], list[dict]]:
    impression_id, now, candidates, history_ids = task
    weights = recency_weights(history_ids)
    rows = build_feature_rows(
        history_ids, weights, candidates, now,
        _STATE["category_by_id"], _STATE["topics_by_id"], _STATE["popularity"], _STATE["max_pop"],
        _STATE["positions"], _STATE["vectors"], _STATE["published_by_id"], _STATE["titles"], _STATE["bm25_index"],
    )
    return impression_id, candidates, rows


def extract(archive: Path, member: str, target: Path) -> Path:
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle, bundle.open(member) as source, target.open("wb") as destination:
        shutil.copyfileobj(source, destination, 1024 * 1024)
    return target


def load_article_metadata(articles_path: Path) -> tuple[dict, dict, dict, dict, "InvertedBM25"]:
    art = pq.read_table(articles_path, columns=["article_id", "category_str", "topics", "published_time", "title", "subtitle"])
    category_by_id, topics_by_id, published_by_id, titles = {}, {}, {}, {}
    ids, texts = [], []
    for article, category, article_topics, timestamp, title, subtitle in zip(
        art["article_id"], art["category_str"], art["topics"], art["published_time"], art["title"], art["subtitle"]
    ):
        article_id = str(article.as_py())
        category_by_id[article_id] = category.as_py() or ""
        topics_by_id[article_id] = tuple(article_topics.as_py() or [])
        published_by_id[article_id] = timestamp.as_py()
        title_text = title.as_py() or ""
        titles[article_id] = title_text
        ids.append(article_id)
        texts.append(f"{title_text} {subtitle.as_py() or ''}")
    bm25_index = InvertedBM25(ids, texts)
    return category_by_id, topics_by_id, published_by_id, titles, bm25_index


def load_training_popularity(train_path: Path) -> Counter:
    popularity: Counter = Counter()
    parquet = pq.ParquetFile(train_path)
    for group in tqdm(range(parquet.num_row_groups), desc="Load user histories", unit="row group"):
        for clicked in parquet.read_row_group(group, columns=["article_ids_clicked"])["article_ids_clicked"].to_pylist():
            popularity.update(str(a) for a in (clicked or []))
    return popularity


def load_compact_histories(history_path: Path, keep_last: int = 30) -> dict[int, tuple]:
    preferences: dict[int, tuple] = {}
    parquet = pq.ParquetFile(history_path)
    for group in range(parquet.num_row_groups):
        table = parquet.read_row_group(group, columns=["user_id", "article_id_fixed"])
        for user, items in zip(table["user_id"].to_pylist(), table["article_id_fixed"].to_pylist()):
            preferences[int(user)] = tuple(str(x) for x in (items or [])[-keep_last:])
        print(f"  history group {group + 1}/{parquet.num_row_groups}")
    return preferences


def precompute_history_context(
    history_ids: list,
    weights: list,
    category_by_id: dict[str, str],
    topics_by_id: dict[str, tuple],
    positions: dict[str, int] | None,
    vectors: np.ndarray | None,
) -> dict:
    """Everything below depends only on the user's history, not on any one
    candidate — so compute it exactly once per impression."""
    category_weights: Counter = Counter()
    topic_weights: Counter = Counter()
    for article, weight in zip(history_ids, weights):
        article_key = str(article)
        category_weights[category_by_id.get(article_key, "")] += weight
        for topic in topics_by_id.get(article_key, ()):
            topic_weights[topic] += weight
    total_category_weight = max(sum(category_weights.values()), 1e-9)
    strongest_topic_weight = max(topic_weights.values()) if topic_weights else 0.0
    user_vector = weighted_user_vector(history_ids, weights, positions, vectors)
    return {
        "category_weights": category_weights,
        "total_category_weight": total_category_weight,
        "topic_weights": topic_weights,
        "strongest_topic_weight": strongest_topic_weight,
        "user_vector": user_vector,
        "history_length": len(history_ids),
    }


def fast_candidate_features(
    article: str,
    position: int,
    context: dict,
    category_by_id: dict[str, str],
    topics_by_id: dict[str, tuple],
    popularity: dict,
    max_pop: int,
    positions: dict[str, int] | None,
    vectors: np.ndarray | None,
    published_by_id: dict,
    now,
    session_count: int,
    session_clicks: int,
) -> dict:
    candidate_category = category_by_id.get(article, "")
    candidate_topics = topics_by_id.get(article, ())

    category_affinity = context["category_weights"][candidate_category] / context["total_category_weight"]

    if not candidate_topics or not context["topic_weights"]:
        topic_affinity = 0.0
    else:
        topic_weights = context["topic_weights"]
        topic_affinity = sum(topic_weights[t] for t in candidate_topics) / (context["strongest_topic_weight"] * len(candidate_topics))

    user_vector = context["user_vector"]
    if user_vector is None or positions is None or vectors is None or article not in positions:
        semantic_score = 0.0
    else:
        semantic_score = float(vectors[positions[article]] @ user_vector)

    freshness = 0.0
    published = published_by_id.get(article) if published_by_id else None
    if published is not None and pd.notna(published):
        hours = max(0.0, (now - published).total_seconds() / 3600)
        freshness = math.exp(-hours / (24 * 4))

    return {
        "article_id": article,
        "category_affinity": category_affinity,
        "topic_affinity": topic_affinity,
        "semantic_score": semantic_score,
        "popularity": math.log1p(popularity.get(article, 0)) / math.log1p(max(max_pop, 1)),
        "freshness_hours_inv": freshness,
        "session_prior_impressions": session_count,
        "session_clicks_so_far": session_clicks,
        "position_bias": 1.0 / math.log2(position + 2),
        "history_length": context["history_length"],
    }


def build_feature_rows(history_ids, weights, candidates, now, category_by_id, topics_by_id, popularity, max_pop, positions, vectors, published_by_id, titles, bm25_index):
    query = query_from_history(history_ids, titles, 5)
    candidate_scores = bm25_index.candidate_scores(query, candidates)
    context = precompute_history_context(history_ids, weights, category_by_id, topics_by_id, positions, vectors)
    return [
        fast_candidate_features(
            article, position, context, category_by_id, topics_by_id,
            popularity, max_pop, positions, vectors, published_by_id, now, 0, 0,
        ) | {"bm25_score": candidate_scores.get(article, 0.0)}
        for position, article in enumerate(candidates)
    ]


def score_batch(buffer_meta: list[tuple[str, list[str]]], buffer_rows: list[dict], model, columns: list[str]) -> list[str]:
    if not buffer_meta:
        return []
    frame = pd.DataFrame(buffer_rows)
    scores = model.predict(frame[columns]) if len(frame) else np.array([])
    lines = []
    offset = 0
    for impression_id, candidates in buffer_meta:
        k = len(candidates)
        local_scores = scores[offset:offset + k]
        offset += k
        order = np.argsort(-local_scores)
        ranks = np.empty(k, dtype=int)
        ranks[order] = np.arange(1, k + 1)
        lines.append(f"{impression_id} [{','.join(map(str, ranks.tolist()))}]\n")
    return lines


def write_predictions(test_path: Path, model, columns, category_by_id, topics_by_id, published_by_id, titles, bm25_index, popularity, max_pop, positions, vectors, preferences, prediction_file: Path, limit: int = 0, predict_batch_size: int = 5000, workers: int = 0, chunksize: int = 32) -> int:
    written = 0
    parquet = pq.ParquetFile(test_path)

    pool = None
    if workers and workers > 1:
        ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context()
        pool = ctx.Pool(
            processes=workers,
            initializer=_init_worker,
            initargs=(category_by_id, topics_by_id, published_by_id, titles, popularity, max_pop, positions, vectors, bm25_index),
        )

    try:
        with prediction_file.open("w") as output:
            stop = False
            for group in tqdm(range(parquet.num_row_groups), desc="EB-NeRD predictions", unit="row group"):
                table = parquet.read_row_group(group, columns=["impression_id", "impression_time", "user_id", "article_ids_inview"])
                tasks = []
                for impression_id, now, user, candidates in zip(
                    table["impression_id"].to_pylist(), table["impression_time"].to_pylist(),
                    table["user_id"].to_pylist(), table["article_ids_inview"].to_pylist(),
                ):
                    candidates = [str(c) for c in candidates]
                    history_ids = list(preferences.get(int(user), ()))
                    tasks.append((str(impression_id), now, candidates, history_ids))
                    if limit and written + len(tasks) >= limit:
                        stop = True
                        break

                buffer_meta, buffer_rows = [], []
                group_written = 0
                results_iter = pool.imap(_score_one_impression, tasks, chunksize=chunksize) if pool else map(_score_one_impression_sequential(category_by_id, topics_by_id, published_by_id, titles, popularity, max_pop, positions, vectors, bm25_index), tasks)
                for impression_id, candidates, rows in results_iter:
                    buffer_rows.extend(rows)
                    buffer_meta.append((impression_id, candidates))
                    if len(buffer_meta) >= predict_batch_size:
                        output.writelines(score_batch(buffer_meta, buffer_rows, model, columns))
                        written += len(buffer_meta)
                        group_written += len(buffer_meta)
                        buffer_meta, buffer_rows = [], []

                output.writelines(score_batch(buffer_meta, buffer_rows, model, columns))
                written += len(buffer_meta)
                group_written += len(buffer_meta)
                print(f"  wrote {written:,} rows ({group + 1}/{parquet.num_row_groups}, {group_written:,} in this group)")
                if stop:
                    break
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    return written


def _score_one_impression_sequential(category_by_id, topics_by_id, published_by_id, titles, popularity, max_pop, positions, vectors, bm25_index):
    """Fallback used when --workers is 0/1: same per-impression work, no pool."""
    def run(task):
        impression_id, now, candidates, history_ids = task
        weights = recency_weights(history_ids)
        rows = build_feature_rows(history_ids, weights, candidates, now, category_by_id, topics_by_id, popularity, max_pop, positions, vectors, published_by_id, titles, bm25_index)
        return impression_id, candidates, rows
    return run


def validate(test_path: Path, prediction_file: Path, limit: int = 0) -> int:
    checked = 0
    parquet = pq.ParquetFile(test_path)
    with prediction_file.open() as output:
        for group in range(parquet.num_row_groups):
            table = parquet.read_row_group(group, columns=["impression_id", "article_ids_inview"])
            for impression_id, candidates in tqdm(zip(table["impression_id"].to_pylist(), table["article_ids_inview"].to_pylist()), total=table.num_rows, desc=f"Validate group {group + 1}", unit="impression", leave=False):
                line = output.readline()
                if not line:
                    return checked
                identifier, text = line.rstrip("\n").split(" ", 1)
                ranks = [int(x) for x in text[1:-1].split(",")]
                if identifier != str(impression_id) or sorted(ranks) != list(range(1, len(candidates) + 1)):
                    raise ValueError(f"Invalid row {impression_id}")
                checked += 1
                if limit and checked >= limit:
                    return checked
    return checked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--large", type=Path, default=Path("ebnerd_large.zip"))
    parser.add_argument("--testzip", type=Path, default=Path("ebnerd_testset.zip"))
    parser.add_argument("--store", type=Path, default=Path("data/processed"), help="small-scale processed store used to TRAIN the re-ranker")
    parser.add_argument("--source-root", type=Path, default=Path("."), help="where Ekstra_Bladet_word2vec.zip lives, for semantic_score")
    parser.add_argument("--work", type=Path, default=Path("data/raw/ebnerd_personalized"))
    parser.add_argument("--out", type=Path, default=Path("outputs/ebnerd_personalized_submission"))
    parser.add_argument("--limit", type=int, default=0, help="Smoke-test only; 0 writes all test rows.")
    parser.add_argument("--predict-batch-size", type=int, default=5000, help="flush predictions every N impressions instead of buffering a whole row group")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1) - 1), help="worker processes for feature building; 0 or 1 disables multiprocessing")
    parser.add_argument("--chunksize", type=int, default=32, help="pool.imap chunksize; larger reduces IPC overhead, smaller improves load balance")
    args = parser.parse_args()

    train = extract(args.large, "train/behaviors.parquet", args.work / "train.parquet")
    articles = extract(args.large, "articles.parquet", args.work / "articles.parquet")
    history = extract(args.testzip, "ebnerd_testset/test/history.parquet", args.work / "history.parquet")
    test = extract(args.testzip, "ebnerd_testset/test/behaviors.parquet", args.work / "test.parquet")

    print("Training the Q2 re-ranker on the small-scale processed store...")
    columns = [name for name in FEATURE_NAMES if name not in SUBMISSION_TIME_UNAVAILABLE] + ["bm25_score"]
    train_features = build_dataset_features(args.store, "ebnerd", "train", 0)
    model = train_model(train_features, columns)

    print("Loading large-catalog article metadata...")
    category_by_id, topics_by_id, published_by_id, titles, bm25_index = load_article_metadata(articles)
    print(f"  {len(category_by_id):,} articles indexed")

    print("Loading large-catalog document embeddings for semantic_score...")
    try:
        emb_ids, emb_vectors = ebnerd_embeddings(args.source_root, list(category_by_id.keys()))
        positions = {article_id: i for i, article_id in enumerate(emb_ids)}
        vectors = emb_vectors
        print(f"  {len(positions):,} articles have a document embedding")
    except FileNotFoundError:
        print("  Ekstra_Bladet_word2vec.zip not found under --source-root — semantic_score will be 0.0 for all candidates")
        positions, vectors = None, None

    print("Counting training clicks...")
    popularity = load_training_popularity(train)
    max_pop = max(popularity.values(), default=1)
    print(f"  {len(popularity):,} clicked articles")

    print("Loading compact test-user histories...")
    preferences = load_compact_histories(history)
    print(f"  loaded compact histories for {len(preferences):,} users")

    print(f"Using {args.workers} worker process(es) for feature building (nproc={os.cpu_count()})")

    args.out.mkdir(parents=True, exist_ok=True)
    prediction = args.out / "predictions.txt"
    print("Scoring the EB-NeRD test set with the trained re-ranker...")
    total = write_predictions(test, model, columns, category_by_id, topics_by_id, published_by_id, titles, bm25_index, popularity, max_pop, positions, vectors, preferences, prediction, args.limit, args.predict_batch_size, args.workers, args.chunksize)

    checked = validate(test, prediction, args.limit)
    bundle = args.out / "ebnerd_personalized_submission.zip"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(prediction, arcname="predictions.txt")
    print(f"Ready: {bundle}; validated {checked:,} rows")
    if args.limit:
        print("Smoke-test ZIP only: rerun without --limit before submitting.")


if __name__ == "__main__":
    main()