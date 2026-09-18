from __future__ import annotations
import argparse
import json
import math
import multiprocessing as mp
import os
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from bm25_retrieval import InvertedBM25, query_from_history
from build_pipeline import recency_weights
from features import FEATURE_NAMES, SUBMISSION_TIME_UNAVAILABLE, weighted_user_vector
from reranker import build_dataset_features, train_model

_STATE: dict = {}


def _init_worker(category_by_id, popularity, max_pop, positions, vectors, bm25_index, titles):
    _STATE["category_by_id"] = category_by_id
    _STATE["popularity"] = popularity
    _STATE["max_pop"] = max_pop
    _STATE["positions"] = positions
    _STATE["vectors"] = vectors
    _STATE["bm25_index"] = bm25_index
    _STATE["titles"] = titles


def _score_one_impression(task: tuple[str, list[str], list[str]]) -> tuple[str, list[str], list[dict]]:
    impression_id, history_ids, candidates = task
    weights = recency_weights(history_ids)
    rows = build_feature_rows(
        history_ids, weights, candidates,
        _STATE["category_by_id"], _STATE["popularity"], _STATE["max_pop"],
        _STATE["positions"], _STATE["vectors"], _STATE["bm25_index"], _STATE["titles"],
    )
    return impression_id, candidates, rows


def require_file(directory: Path, name: str) -> Path:
    path = directory / name
    if not path.exists():
        raise FileNotFoundError(
            f"Expected an extracted MIND folder at {directory} containing {name}, "
            f"but {path} does not exist. Point --train/--dev/--test at the "
            f"extracted folder, not the .zip file."
        )
    return path


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as raw:
        return sum(1 for _ in raw)


def load_news(directories: list[Path]) -> dict[str, dict]:
    news: dict[str, dict] = {}
    for directory in directories:
        path = require_file(directory, "news.tsv")
        total = count_lines(path)
        with path.open("r", encoding="utf-8") as raw:
            for line in tqdm(raw, total=total, desc=f"Loading news.tsv ({directory.name})", unit="article"):
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 5:
                    continue
                article_id = fields[0]
                if article_id not in news:
                    news[article_id] = {"category": fields[1], "title": fields[3], "abstract": fields[4]}
    return news


def entity_ids(raw: str) -> list[str]:
    try:
        values = json.loads(raw) if raw else []
    except json.JSONDecodeError:
        return []
    return [item.get("WikidataId") for item in values if item.get("WikidataId")]


def load_entity_vectors(train_dir: Path) -> dict[str, np.ndarray]:
    vectors: dict[str, np.ndarray] = {}
    path = require_file(train_dir, "entity_embedding.vec")
    total = count_lines(path)
    with path.open("r", encoding="utf-8") as raw:
        for line in tqdm(raw, total=total, desc="Loading entity embeddings", unit="entity"):
            fields = line.rstrip().split("\t")
            vectors[fields[0]] = np.asarray(fields[1:], dtype=np.float32)
    return vectors


def load_article_embeddings(directories: list[Path], entity_vectors: dict[str, np.ndarray]) -> tuple[dict[str, int] | None, np.ndarray | None]:
    ids: list[str] = []
    vecs: list[np.ndarray] = []
    seen: set[str] = set()
    for directory in directories:
        path = require_file(directory, "news.tsv")
        total = count_lines(path)
        with path.open("r", encoding="utf-8") as raw:
            for line in tqdm(raw, total=total, desc=f"Building article embeddings ({directory.name})", unit="article"):
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 8:
                    continue
                article_id = fields[0]
                if article_id in seen:
                    continue
                seen.add(article_id)
                entities = entity_ids(fields[6]) + entity_ids(fields[7])
                available = [entity_vectors[e] for e in entities if e in entity_vectors]
                if not available:
                    continue
                vector = np.mean(available, axis=0)
                norm = float(np.linalg.norm(vector))
                if norm > 1e-12:
                    vector = vector / norm
                ids.append(article_id)
                vecs.append(vector)
    if not ids:
        return None, None
    return {article_id: i for i, article_id in enumerate(ids)}, np.vstack(vecs).astype(np.float32)


def click_popularity(train_dir: Path) -> Counter:
    counts: Counter = Counter()
    path = require_file(train_dir, "behaviors.tsv")
    total = count_lines(path)
    with path.open("r", encoding="utf-8") as raw:
        for line in tqdm(raw, total=total, desc="Counting MINDlarge_train clicks", unit="impression"):
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 5:
                continue
            for item in fields[4].split():
                article, sep, label = item.rpartition("-")
                if sep and label == "1":
                    counts[article] += 1
    return counts


def precompute_history_context(
    history_ids: list,
    weights: list,
    category_by_id: dict[str, str],
    positions: dict[str, int] | None,
    vectors: np.ndarray | None,
) -> dict:
    category_weights: Counter = Counter()
    for article, weight in zip(history_ids, weights):
        category_weights[category_by_id.get(str(article), "")] += weight
    total_category_weight = max(sum(category_weights.values()), 1e-9)
    user_vector = weighted_user_vector(history_ids, weights, positions, vectors)
    return {
        "category_weights": category_weights,
        "total_category_weight": total_category_weight,
        "user_vector": user_vector,
        "history_length": len(history_ids),
    }


def fast_candidate_features(
    article: str,
    position: int,
    context: dict,
    category_by_id: dict[str, str],
    popularity: dict,
    max_pop: int,
    positions: dict[str, int] | None,
    vectors: np.ndarray | None,
) -> dict:
    candidate_category = category_by_id.get(article, "")
    category_affinity = context["category_weights"][candidate_category] / context["total_category_weight"]

    user_vector = context["user_vector"]
    if user_vector is None or positions is None or vectors is None or article not in positions:
        semantic_score = 0.0
    else:
        semantic_score = float(vectors[positions[article]] @ user_vector)

    return {
        "article_id": article,
        "category_affinity": category_affinity,
        "topic_affinity": 0.0,
        "semantic_score": semantic_score,
        "popularity": math.log1p(popularity.get(article, 0)) / math.log1p(max(max_pop, 1)),
        "freshness_hours_inv": 0.0,
        "session_prior_impressions": 0,
        "session_clicks_so_far": 0,
        "position_bias": 1.0 / math.log2(position + 2),
        "history_length": context["history_length"],
    }


def build_feature_rows(history_ids, weights, candidates, category_by_id, popularity, max_pop, positions, vectors, bm25_index, titles):
    query = query_from_history(history_ids, titles, 5)
    candidate_scores = bm25_index.candidate_scores(query, candidates)
    context = precompute_history_context(history_ids, weights, category_by_id, positions, vectors)
    return [
        fast_candidate_features(
            article, position, context, category_by_id, popularity, max_pop, positions, vectors,
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


def _read_tasks(test_dir: Path, limit: int):
    path = require_file(test_dir, "behaviors.tsv")
    with path.open("r", encoding="utf-8") as raw:
        count = 0
        for line in raw:
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 5:
                raise ValueError(f"Expected five MIND columns, got {len(fields)}")
            impression_id, _user_id, _timestamp, history_field, candidates_field = fields
            history_ids = history_field.split() if history_field else []
            candidates = candidates_field.split()
            yield (impression_id, history_ids, candidates)
            count += 1
            if limit and count >= limit:
                return


def write_predictions(test_dir: Path, model, columns, category_by_id, popularity, max_pop, positions, vectors, bm25_index, titles, prediction_file: Path, batch_size: int = 5000, limit: int = 0, workers: int = 0, chunksize: int = 32) -> int:
    written = 0
    path = require_file(test_dir, "behaviors.tsv")
    total = count_lines(path)
    if limit:
        total = min(total, limit)

    pool = None
    if workers and workers > 1:
        ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context()
        pool = ctx.Pool(
            processes=workers,
            initializer=_init_worker,
            initargs=(category_by_id, popularity, max_pop, positions, vectors, bm25_index, titles),
        )

    try:
        with prediction_file.open("w", encoding="utf-8") as output:
            buffer_meta, buffer_rows = [], []
            tasks = _read_tasks(test_dir, limit)
            results_iter = pool.imap(_score_one_impression, tasks, chunksize=chunksize) if pool else map(_score_one_impression_sequential(category_by_id, popularity, max_pop, positions, vectors, bm25_index, titles), tasks)
            progress = tqdm(results_iter, total=total, desc="Scoring MINDlarge_test", unit="impression")
            for impression_id, candidates, rows in progress:
                buffer_rows.extend(rows)
                buffer_meta.append((impression_id, candidates))
                if len(buffer_meta) >= batch_size:
                    output.writelines(score_batch(buffer_meta, buffer_rows, model, columns))
                    written += len(buffer_meta)
                    buffer_meta, buffer_rows = [], []
                    progress.set_postfix(written=f"{written:,}")
            output.writelines(score_batch(buffer_meta, buffer_rows, model, columns))
            written += len(buffer_meta)
            progress.set_postfix(written=f"{written:,}")
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    return written


def _score_one_impression_sequential(category_by_id, popularity, max_pop, positions, vectors, bm25_index, titles):
    def run(task):
        impression_id, history_ids, candidates = task
        weights = recency_weights(history_ids)
        rows = build_feature_rows(history_ids, weights, candidates, category_by_id, popularity, max_pop, positions, vectors, bm25_index, titles)
        return impression_id, candidates, rows
    return run


def validate(test_dir: Path, prediction_file: Path, expected_limit: int = 0) -> int:
    checked = 0
    path = require_file(test_dir, "behaviors.tsv")
    total = count_lines(path)
    if expected_limit:
        total = min(total, expected_limit)
    with path.open("r", encoding="utf-8") as raw, prediction_file.open(encoding="utf-8") as output:
        for source, prediction in tqdm(zip(raw, output), total=total, desc="Validate MIND predictions", unit="impression"):
            source_fields = source.rstrip("\n").split("\t")
            expected_id = source_fields[0]
            candidates = source_fields[4].split()
            got_id, ranks_text = prediction.rstrip("\n").split(" ", 1)
            ranks = [int(value) for value in ranks_text.strip()[1:-1].split(",")]
            if got_id != expected_id:
                raise ValueError(f"Impression ID mismatch: {got_id} != {expected_id}")
            if len(ranks) != len(candidates) or sorted(ranks) != list(range(1, len(candidates) + 1)):
                raise ValueError(f"Invalid rank permutation for impression {expected_id}")
            checked += 1
            if expected_limit and checked >= expected_limit:
                break
        if not expected_limit:
            if next(raw, None) is not None or next(output, None) is not None:
                raise ValueError("Prediction line count differs from test impression count")
    return checked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("MIND_data/MINDlarge_train"), help="extracted MINDlarge_train folder (contains news.tsv, behaviors.tsv, entity_embedding.vec)")
    parser.add_argument("--dev", type=Path, default=Path("MIND_data/MINDlarge_dev"), help="extracted MINDlarge_dev folder")
    parser.add_argument("--test", type=Path, default=Path("MIND_data/MINDlarge_test"), help="extracted MINDlarge_test folder")
    parser.add_argument("--store", type=Path, default=Path("data/processed"), help="small-scale processed store used to TRAIN the re-ranker")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/mind_submission"))
    parser.add_argument("--limit", type=int, default=0, help="Use only for a local smoke test; 0 processes all test impressions.")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1) - 1), help="worker processes for feature building; 0 or 1 disables multiprocessing")
    parser.add_argument("--chunksize", type=int, default=32, help="pool.imap chunksize; larger reduces IPC overhead, smaller improves load balance")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction = args.output_dir / "prediction.txt"
    submission_zip = args.output_dir / "mind_submission.zip"

    print("Training the Q2 re-ranker on the small-scale processed store...")
    columns = [name for name in FEATURE_NAMES if name not in SUBMISSION_TIME_UNAVAILABLE] + ["bm25_score"]
    train_features = build_dataset_features(args.store, "mind", "train", 0)
    model = train_model(train_features, columns)

    print("Loading MINDlarge article metadata (train+dev+test news.tsv)...")
    directories = [args.train, args.dev, args.test]
    news = load_news(directories)
    category_by_id = {article_id: entry["category"] for article_id, entry in news.items()}
    titles = {article_id: entry["title"] for article_id, entry in news.items()}
    texts = [f'{entry["title"]} {entry["abstract"]}' for entry in news.values()]
    bm25_index = InvertedBM25(list(news.keys()), texts)
    print(f"  {len(news):,} articles indexed")

    print("Loading MINDlarge entity embeddings for semantic_score...")
    entity_vectors = load_entity_vectors(args.train)
    positions, vectors = load_article_embeddings(directories, entity_vectors)
    print(f"  {len(positions) if positions else 0:,} articles have an entity embedding")

    print("Counting clicks in MINDlarge_train...")
    popularity = click_popularity(args.train)
    max_pop = max(popularity.values(), default=1)
    print(f"  {len(popularity):,} clicked articles")

    print(f"Using {args.workers} worker process(es) for feature building (nproc={os.cpu_count()})")

    print("Scoring MINDlarge_test with the trained re-ranker...")
    total = write_predictions(args.test, model, columns, category_by_id, popularity, max_pop, positions, vectors, bm25_index, titles, prediction, limit=args.limit, workers=args.workers, chunksize=args.chunksize)
    checked = validate(args.test, prediction, args.limit)
    print(f"Validated {checked:,} rank lists.")
    with zipfile.ZipFile(submission_zip, "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(prediction, arcname="prediction.txt")
    print(f"Ready: {submission_zip} ({total:,} predictions)")
    if args.limit:
        print("Smoke-test ZIP only: rerun without --limit before submitting.")


if __name__ == "__main__":
    main()