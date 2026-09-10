from __future__ import annotations
import argparse
import shutil
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from bm25_retrieval import InvertedBM25, query_from_history
from build_pipeline import recency_weights
from features import FEATURE_NAMES, candidate_features
from reranker import build_dataset_features, train_model
from semantic_retrieval import ebnerd_embeddings


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
    for group in range(parquet.num_row_groups):
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


def build_feature_rows(history_ids, weights, candidates, now, category_by_id, topics_by_id, popularity, max_pop, positions, vectors, published_by_id, titles, bm25_index):
    query = query_from_history(history_ids, titles, 5)
    candidate_scores = bm25_index.candidate_scores(query, candidates)
    return [
        candidate_features(
            article, position, history_ids, weights, category_by_id, topics_by_id,
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


def write_predictions(test_path: Path, model, columns, category_by_id, topics_by_id, published_by_id, titles, bm25_index, popularity, max_pop, positions, vectors, preferences, prediction_file: Path, limit: int = 0) -> int:
    written = 0
    parquet = pq.ParquetFile(test_path)
    with prediction_file.open("w") as output:
        stop = False
        for group in range(parquet.num_row_groups):
            table = parquet.read_row_group(group, columns=["impression_id", "impression_time", "user_id", "article_ids_inview"])
            buffer_meta, buffer_rows = [], []
            for impression_id, now, user, candidates in zip(
                table["impression_id"].to_pylist(), table["impression_time"].to_pylist(),
                table["user_id"].to_pylist(), table["article_ids_inview"].to_pylist(),
            ):
                candidates = [str(c) for c in candidates]
                history_ids = list(preferences.get(int(user), ()))
                weights = recency_weights(history_ids)
                buffer_rows.extend(build_feature_rows(history_ids, weights, candidates, now, category_by_id, topics_by_id, popularity, max_pop, positions, vectors, published_by_id, titles, bm25_index))
                buffer_meta.append((str(impression_id), candidates))
                if limit and written + len(buffer_meta) >= limit:
                    stop = True
                    break
            output.writelines(score_batch(buffer_meta, buffer_rows, model, columns))
            written += len(buffer_meta)
            print(f"  wrote {written:,} rows ({group + 1}/{parquet.num_row_groups})")
            if stop:
                break
    return written


def validate(test_path: Path, prediction_file: Path, limit: int = 0) -> int:
    checked = 0
    parquet = pq.ParquetFile(test_path)
    with prediction_file.open() as output:
        for group in range(parquet.num_row_groups):
            table = parquet.read_row_group(group, columns=["impression_id", "article_ids_inview"])
            for impression_id, candidates in zip(table["impression_id"].to_pylist(), table["article_ids_inview"].to_pylist()):
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
    args = parser.parse_args()

    train = extract(args.large, "train/behaviors.parquet", args.work / "train.parquet")
    articles = extract(args.large, "articles.parquet", args.work / "articles.parquet")
    history = extract(args.testzip, "ebnerd_testset/test/history.parquet", args.work / "history.parquet")
    test = extract(args.testzip, "ebnerd_testset/test/behaviors.parquet", args.work / "test.parquet")

    print("Training the Q2 re-ranker on the small-scale processed store...")
    columns = FEATURE_NAMES + ["bm25_score"]
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

    args.out.mkdir(parents=True, exist_ok=True)
    prediction = args.out / "predictions.txt"
    print("Scoring the EB-NeRD test set with the trained re-ranker...")
    total = write_predictions(test, model, columns, category_by_id, topics_by_id, published_by_id, titles, bm25_index, popularity, max_pop, positions, vectors, preferences, prediction, args.limit)

    checked = validate(test, prediction, args.limit)
    bundle = args.out / "ebnerd_personalized_submission.zip"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(prediction, arcname="predictions.txt")
    print(f"Ready: {bundle}; validated {checked:,} rows")
    if args.limit:
        print("Smoke-test ZIP only: rerun without --limit before submitting.")


if __name__ == "__main__":
    main()