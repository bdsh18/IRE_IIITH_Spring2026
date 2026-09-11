from __future__ import annotations
import argparse
import json
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from bm25_retrieval import InvertedBM25, query_from_history
from build_pipeline import recency_weights
from features import FEATURE_NAMES, SUBMISSION_TIME_UNAVAILABLE, candidate_features
from mind_personalized_ranker import member, popularity as click_popularity
from reranker import build_dataset_features, train_model

NEWS_MEMBER = "/news.tsv"
BEHAVIORS_MEMBER = "/behaviors.tsv"


def load_news(archives: list[Path]) -> dict[str, dict]:
    news: dict[str, dict] = {}
    for archive in archives:
        with zipfile.ZipFile(archive) as bundle, bundle.open(member(archive, NEWS_MEMBER)) as raw:
            for line in raw:
                fields = line.decode("utf-8").rstrip("\n").split("\t")
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


def load_entity_vectors(train: Path) -> dict[str, np.ndarray]:
    vectors: dict[str, np.ndarray] = {}
    with zipfile.ZipFile(train) as bundle, bundle.open(member(train, "/entity_embedding.vec")) as raw:
        for line in raw:
            fields = line.decode().rstrip().split("\t")
            vectors[fields[0]] = np.asarray(fields[1:], dtype=np.float32)
    return vectors


def load_article_embeddings(archives: list[Path], entity_vectors: dict[str, np.ndarray]) -> tuple[dict[str, int] | None, np.ndarray | None]:
    ids: list[str] = []
    vecs: list[np.ndarray] = []
    seen: set[str] = set()
    for archive in archives:
        with zipfile.ZipFile(archive) as bundle, bundle.open(member(archive, NEWS_MEMBER)) as raw:
            for line in raw:
                fields = line.decode("utf-8").rstrip("\n").split("\t")
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


def build_feature_rows(history_ids, weights, candidates, category_by_id, popularity, max_pop, positions, vectors, bm25_index, titles):
    query = query_from_history(history_ids, titles, 5)
    candidate_scores = bm25_index.candidate_scores(query, candidates)
    return [
        candidate_features(
            article, position, history_ids, weights, category_by_id, {},
            popularity, max_pop, positions, vectors, {}, None, 0, 0,
        ) | {"bm25_score": candidate_scores.get(article, 0.0)}
        for position, article in enumerate(candidates)
    ]


def write_predictions(test_archive: Path, model, columns, category_by_id, popularity, max_pop, positions, vectors, bm25_index, titles, prediction_file: Path, batch_size: int = 5000, limit: int = 0) -> int:
    written = 0
    buffer_meta: list[tuple[str, list[str]]] = []
    buffer_rows: list[dict] = []
    with zipfile.ZipFile(test_archive) as bundle, bundle.open(member(test_archive, BEHAVIORS_MEMBER)) as raw, prediction_file.open("w", encoding="utf-8") as output:
        for line in raw:
            fields = line.decode("utf-8").rstrip("\n").split("\t")
            if len(fields) != 5:
                raise ValueError(f"Expected five MIND columns, got {len(fields)}")
            impression_id, _user_id, _timestamp, history_field, candidates_field = fields
            history_ids = history_field.split() if history_field else []
            weights = recency_weights(history_ids)
            candidates = candidates_field.split()
            buffer_rows.extend(build_feature_rows(history_ids, weights, candidates, category_by_id, popularity, max_pop, positions, vectors, bm25_index, titles))
            buffer_meta.append((impression_id, candidates))
            if len(buffer_meta) >= batch_size:
                output.writelines(score_batch(buffer_meta, buffer_rows, model, columns))
                written += len(buffer_meta)
                buffer_meta, buffer_rows = [], []
                print(f"  wrote {written:,} predictions")
            if limit and written >= limit:
                break
        output.writelines(score_batch(buffer_meta, buffer_rows, model, columns))
        written += len(buffer_meta)
    return written


def validate(test_archive: Path, prediction_file: Path, expected_limit: int = 0) -> int:
    checked = 0
    with zipfile.ZipFile(test_archive) as bundle, bundle.open(member(test_archive, BEHAVIORS_MEMBER)) as raw, prediction_file.open(encoding="utf-8") as output:
        for source, prediction in zip(raw, output):
            source_fields = source.decode("utf-8").rstrip("\n").split("\t")
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
    parser.add_argument("--train", type=Path, default=Path("MIND_data/MINDlarge_train.zip"))
    parser.add_argument("--dev", type=Path, default=Path("MIND_data/MINDlarge_dev.zip"))
    parser.add_argument("--test", type=Path, default=Path("MIND_data/MINDlarge_test.zip"))
    parser.add_argument("--store", type=Path, default=Path("data/processed"), help="small-scale processed store used to TRAIN the re-ranker")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/mind_submission"))
    parser.add_argument("--limit", type=int, default=0, help="Use only for a local smoke test; 0 processes all test impressions.")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction = args.output_dir / "prediction.txt"
    submission_zip = args.output_dir / "mind_submission.zip"

    print("Training the Q2 re-ranker on the small-scale processed store...")
    columns = [name for name in FEATURE_NAMES if name not in SUBMISSION_TIME_UNAVAILABLE] + ["bm25_score"]
    train_features = build_dataset_features(args.store, "mind", "train", 0)
    model = train_model(train_features, columns)

    print("Loading MINDlarge article metadata (train+dev+test news.tsv)...")
    archives = [args.train, args.dev, args.test]
    news = load_news(archives)
    category_by_id = {article_id: entry["category"] for article_id, entry in news.items()}
    titles = {article_id: entry["title"] for article_id, entry in news.items()}
    texts = [f'{entry["title"]} {entry["abstract"]}' for entry in news.values()]
    bm25_index = InvertedBM25(list(news.keys()), texts)
    print(f"  {len(news):,} articles indexed")

    print("Loading MINDlarge entity embeddings for semantic_score...")
    entity_vectors = load_entity_vectors(args.train)
    positions, vectors = load_article_embeddings(archives, entity_vectors)
    print(f"  {len(positions) if positions else 0:,} articles have an entity embedding")

    print("Counting clicks in MINDlarge_train...")
    popularity = click_popularity(args.train)
    max_pop = max(popularity.values(), default=1)
    print(f"  {len(popularity):,} clicked articles")

    print("Scoring MINDlarge_test with the trained re-ranker...")
    total = write_predictions(args.test, model, columns, category_by_id, popularity, max_pop, positions, vectors, bm25_index, titles, prediction, limit=args.limit)
    checked = validate(args.test, prediction, args.limit)
    print(f"Validated {checked:,} rank lists.")
    with zipfile.ZipFile(submission_zip, "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(prediction, arcname="prediction.txt")
    print(f"Ready: {submission_zip} ({total:,} predictions)")
    if args.limit:
        print("Smoke-test ZIP only: rerun without --limit before submitting.")


if __name__ == "__main__":
    main()