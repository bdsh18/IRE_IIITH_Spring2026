from __future__ import annotations
import argparse
import hashlib
import json
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
from features import (
    candidate_features_from_context,
    prepare_candidate_context,
    ranker_feature_names,
)
from reranker import build_dataset_features, predict_scores, train_model
from semantic_retrieval import ebnerd_embeddings

CHECKPOINT_VERSION = 1


def extract(archive: Path, member: str, target: Path) -> Path:
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle, bundle.open(member) as source, target.open("wb") as destination:
        shutil.copyfileobj(source, destination, 1024 * 1024)
    return target


def load_article_metadata(articles_paths) -> tuple[dict, dict, dict, dict, "InvertedBM25"]:
    if isinstance(articles_paths, (str, Path)):
        articles_paths = [articles_paths]
    category_by_id, topics_by_id, published_by_id, titles = {}, {}, {}, {}
    ids, texts = [], []
    rows = []
    for articles_path in articles_paths:
        art = pq.read_table(articles_path, columns=["article_id", "category_str", "topics", "published_time", "title", "subtitle"])
        rows.extend(zip(art["article_id"], art["category_str"], art["topics"], art["published_time"], art["title"], art["subtitle"]))
    for article, category, article_topics, timestamp, title, subtitle in rows:
        article_id = str(article.as_py())
        if article_id in category_by_id:
            continue
        category_by_id[article_id] = category.as_py() or ""
        topics_by_id[article_id] = tuple(article_topics.as_py() or [])
        published_by_id[article_id] = timestamp.as_py()
        title_text = title.as_py() or ""
        titles[article_id] = title_text
        ids.append(article_id)
        texts.append(f"{title_text} {subtitle.as_py() or ''}")
    bm25_index = InvertedBM25(ids, texts)
    return category_by_id, topics_by_id, published_by_id, titles, bm25_index


def load_training_popularity(behaviour_paths) -> Counter:
    """Count clicks in labelled behaviour logs that precede the test period."""
    if isinstance(behaviour_paths, (str, Path)):
        behaviour_paths = [behaviour_paths]
    popularity: Counter = Counter()
    for train_path in behaviour_paths:
        parquet = pq.ParquetFile(train_path)
        for group in tqdm(range(parquet.num_row_groups), desc=f"Count clicks ({Path(train_path).name})", unit="row group"):
            for clicked in parquet.read_row_group(group, columns=["article_ids_clicked"])["article_ids_clicked"].to_pylist():
                popularity.update(str(a) for a in (clicked or []))
    return popularity


def load_compact_histories(history_path: Path, keep_last: int = 30) -> tuple[dict[int, tuple], dict[int, int]]:
    preferences: dict[int, tuple] = {}
    lengths: dict[int, int] = {}
    parquet = pq.ParquetFile(history_path)
    for group in range(parquet.num_row_groups):
        table = parquet.read_row_group(group, columns=["user_id", "article_id_fixed"])
        for user, items in zip(table["user_id"].to_pylist(), table["article_id_fixed"].to_pylist()):
            items = items or []
            preferences[int(user)] = tuple(str(x) for x in items[-keep_last:])
            lengths[int(user)] = len(items)
        print(f"  history group {group + 1}/{parquet.num_row_groups}")
    return preferences, lengths


def build_feature_rows(history_ids, weights, candidates, now, category_by_id, topics_by_id, popularity, max_pop, positions, vectors, published_by_id, titles, bm25_index, query: str | None = None, context: dict | None = None):
    query = query if query is not None else query_from_history(history_ids, titles, 5)
    candidate_scores = bm25_index.candidate_scores(query, candidates)
    context = context if context is not None else prepare_candidate_context(
        history_ids, weights, category_by_id, topics_by_id, popularity, max_pop, positions, vectors,
    )
    return [
        candidate_features_from_context(
            article, position, context, category_by_id, topics_by_id,
            published_by_id, now, 0, 0, 0.0,
        ) | {"bm25_score": candidate_scores.get(article, 0.0)}
        for position, article in enumerate(candidates)
    ]


def score_batch(buffer_meta: list[tuple[str, list[str]]], buffer_rows: list[dict], model, columns: list[str]) -> list[str]:
    if not buffer_meta:
        return []
    frame = pd.DataFrame(buffer_rows)
    scores = predict_scores(model, frame[columns]) if len(frame) else np.array([])
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


def validate_prediction_line(expected_id: str, candidates: list[str], prediction: str) -> None:
    if not prediction.endswith("\n"):
        raise ValueError(f"Incomplete prediction line for impression {expected_id}")
    identifier, text = prediction.rstrip("\n").split(" ", 1)
    if not text.startswith("[") or not text.endswith("]"):
        raise ValueError(f"Malformed rank list for impression {expected_id}")
    contents = text[1:-1]
    ranks = [] if not contents else [int(value) for value in contents.split(",")]
    if identifier != expected_id or len(ranks) != len(candidates) or sorted(ranks) != list(range(1, len(candidates) + 1)):
        raise ValueError(f"Invalid row {expected_id}")


def source_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_fingerprint(test_path: Path, columns: list[str], batch_size: int, limit: int, store: Path, model) -> str:
    """Describe exactly what a resumable prediction file was generated from."""
    root = Path(__file__).resolve().parent
    test_stat = test_path.stat()
    cache = store / "ebnerd" / "train_features_with_bm25.parquet"
    cache_metadata = None
    if cache.exists():
        stat = cache.stat()
        cache_metadata = {"path": str(cache.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    payload = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "test": {"path": str(test_path.resolve()), "size": test_stat.st_size, "mtime_ns": test_stat.st_mtime_ns},
        "columns": columns,
        "batch_size": batch_size,
        "limit": limit,
        "ranker_backend": getattr(model, "_a2_backend", "lightgbm_lambdarank"),
        "training_cache": cache_metadata,
        "source_hashes": {
            name: source_hash(root / name)
            for name in ("make_ebnerd_personalized_submission.py", "bm25_retrieval.py", "features.py", "reranker.py")
        },
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def write_checkpoint(checkpoint: Path, payload: dict) -> None:
    temporary = checkpoint.with_name(f"{checkpoint.name}.tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(payload, output, indent=2)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, checkpoint)


def iter_test_rows(test_path: Path, batch_size: int, columns: list[str]):
    """Yield bounded, source-ordered test batches without loading row groups wholesale."""
    parquet = pq.ParquetFile(test_path)
    yield from parquet.iter_batches(batch_size=batch_size, columns=columns)


def validate_prefix(test_path: Path, prediction_file: Path, batch_size: int, limit: int = 0) -> int:
    """Validate the complete, source-ordered prefix in a resumable partial file."""
    checked = 0
    columns = ["impression_id", "article_ids_inview"]
    with prediction_file.open(encoding="utf-8") as output:
        for batch in iter_test_rows(test_path, batch_size, columns):
            for impression_id, candidates in zip(batch.column("impression_id").to_pylist(), batch.column("article_ids_inview").to_pylist()):
                if limit and checked >= limit:
                    break
                line = output.readline()
                if not line:
                    if output.readline():
                        raise ValueError("Partial prediction file contains invalid trailing data")
                    return checked
                expected_candidates = [str(candidate) for candidate in (candidates or [])]
                validate_prediction_line(str(impression_id), expected_candidates, line)
                checked += 1
            if limit and checked >= limit:
                break
        if output.readline():
            raise ValueError("Partial prediction file contains rows after its validated source prefix")
    return checked


def resume_count(test_path: Path, partial: Path, checkpoint: Path, fingerprint: str, batch_size: int, limit: int) -> int:
    if not partial.exists() or not checkpoint.exists():
        raise FileNotFoundError("--resume needs both the .partial prediction file and its checkpoint JSON")
    state = json.loads(checkpoint.read_text(encoding="utf-8"))
    if state.get("fingerprint") != fingerprint:
        raise ValueError("Resume checkpoint does not match this test file, model, feature store, or code version")
    completed, byte_offset = int(state["completed"]), int(state["byte_offset"])
    with partial.open("r+b") as output:
        output.seek(0, os.SEEK_END)
        if output.tell() < byte_offset:
            raise ValueError("Partial prediction file is shorter than its checkpoint")
        # A crash can leave a final uncheckpointed write.  Do not trust it.
        output.truncate(byte_offset)
    verified = validate_prefix(test_path, partial, batch_size, limit)
    if verified != completed:
        raise ValueError(f"Resume prefix validation found {verified} rows; checkpoint expects {completed}")
    return completed


def write_predictions(test_path: Path, model, columns, category_by_id, topics_by_id, published_by_id, titles, bm25_index, popularity, max_pop, positions, vectors, preferences, partial: Path, checkpoint: Path, fingerprint: str, batch_size: int = 5000, limit: int = 0, resume: bool = False, context_cache_size: int = 100_000, history_lengths: dict[int, int] | None = None) -> int:
    written = resume_count(test_path, partial, checkpoint, fingerprint, batch_size, limit) if resume else 0
    if limit and written > limit:
        raise ValueError("Checkpoint contains more rows than the requested --limit")
    buffer_meta: list[tuple[str, list[str]]] = []
    buffer_rows: list[dict] = []
    source_columns = ["impression_id", "impression_time", "user_id", "article_ids_inview"]
    rows_to_skip = written
    mode = "ab" if written else "wb"
    # Test histories are fixed per user, while one user commonly has several
    # impressions.  Reusing their query and user-side feature context avoids
    # repeating category/topic aggregation and weighted embedding pooling.
    # FIFO eviction keeps the memory cap explicit on the 13.5M-row test set.
    user_context_cache: dict[int, tuple[tuple, list[float], str, dict]] = {}
    empty_history_entry: tuple[tuple, list[float], str, dict] | None = None
    cache_hits = cache_misses = 0

    def context_for_user(user: int) -> tuple[tuple, list[float], str, dict]:
        nonlocal empty_history_entry, cache_hits, cache_misses
        if context_cache_size:
            cached = user_context_cache.get(user)
            if cached is not None:
                cache_hits += 1
                return cached
        cache_misses += 1
        history_ids = preferences.get(user, ())
        if not history_ids and empty_history_entry is not None:
            return empty_history_entry
        weights = recency_weights(history_ids)
        query = query_from_history(history_ids, titles, 5)
        context = prepare_candidate_context(
            history_ids, weights, category_by_id, topics_by_id,
            popularity, max_pop, positions, vectors,
        )
        if history_lengths is not None:
            context["history_length"] = history_lengths.get(user, len(history_ids))
        entry = (history_ids, weights, query, context)
        if not history_ids:
            empty_history_entry = entry
        elif context_cache_size:
            if len(user_context_cache) >= context_cache_size:
                user_context_cache.pop(next(iter(user_context_cache)))
            user_context_cache[user] = entry
        return entry

    with partial.open(mode) as output:
        def flush() -> None:
            nonlocal written, buffer_meta, buffer_rows
            if not buffer_meta:
                return
            lines = score_batch(buffer_meta, buffer_rows, model, columns)
            output.write("".join(lines).encode("utf-8"))
            output.flush()
            os.fsync(output.fileno())
            written += len(buffer_meta)
            write_checkpoint(checkpoint, {
                "version": CHECKPOINT_VERSION, "fingerprint": fingerprint,
                "completed": written, "byte_offset": output.tell(),
                "last_impression_id": buffer_meta[-1][0],
            })
            print(f"  checkpointed {written:,} predictions")
            buffer_meta, buffer_rows = [], []

        for batch in tqdm(iter_test_rows(test_path, batch_size, source_columns), desc="EB-NeRD predictions", unit="batch"):
            impression_ids = batch.column("impression_id").to_pylist()
            times = batch.column("impression_time").to_pylist()
            users = batch.column("user_id").to_pylist()
            candidate_lists = batch.column("article_ids_inview").to_pylist()
            start = min(rows_to_skip, len(impression_ids))
            rows_to_skip -= start
            if start == len(impression_ids):
                continue
            stop = False
            for impression_id, now, user, candidates_raw in zip(
                impression_ids[start:], times[start:], users[start:], candidate_lists[start:]
            ):
                if limit and written + len(buffer_meta) >= limit:
                    stop = True
                    break
                candidates = [str(candidate) for candidate in (candidates_raw or [])]
                history_ids, weights, query, context = context_for_user(int(user))
                buffer_rows.extend(build_feature_rows(
                    history_ids, weights, candidates, now, category_by_id, topics_by_id,
                    popularity, max_pop, positions, vectors, published_by_id, titles, bm25_index,
                    query=query, context=context,
                ))
                buffer_meta.append((str(impression_id), candidates))
                if len(buffer_meta) >= batch_size:
                    flush()
            if stop:
                break
        if rows_to_skip:
            raise ValueError("Resume checkpoint exceeds the number of test impressions")
        flush()
    if cache_hits or cache_misses:
        total_lookups = cache_hits + cache_misses
        print(f"  user-context cache: {cache_hits:,}/{total_lookups:,} hits; {len(user_context_cache):,} cached users")
    return written


def validate(test_path: Path, prediction_file: Path, batch_size: int, limit: int = 0) -> int:
    checked = 0
    columns = ["impression_id", "article_ids_inview"]
    with prediction_file.open(encoding="utf-8") as output:
        for batch in tqdm(iter_test_rows(test_path, batch_size, columns), desc="Validate EB-NeRD predictions", unit="batch"):
            for impression_id, candidates in zip(batch.column("impression_id").to_pylist(), batch.column("article_ids_inview").to_pylist()):
                if limit and checked >= limit:
                    break
                line = output.readline()
                if not line:
                    raise ValueError("Prediction line count is smaller than the expected test impression count")
                validate_prediction_line(str(impression_id), [str(candidate) for candidate in (candidates or [])], line)
                checked += 1
            if limit and checked >= limit:
                break
        if limit and checked != limit:
            raise ValueError(f"Expected {limit} predictions but validated {checked}")
        if output.readline():
            raise ValueError("Prediction line count is larger than the expected test impression count")
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
    parser.add_argument("--batch-size", type=int, default=5000, help="impressions per durable checkpoint")
    parser.add_argument("--context-cache-size", type=int, default=100_000, help="maximum reusable non-empty user contexts in memory; 0 disables the bounded cache")
    parser.add_argument("--resume", action="store_true", help="resume a matching interrupted .partial prediction run")
    args = parser.parse_args()

    train = extract(args.large, "train/behaviors.parquet", args.work / "train.parquet")
    behaviour_logs = [train]
    try:
        # The validation week is labelled and precedes the test period.
        behaviour_logs.append(extract(args.large, "validation/behaviors.parquet", args.work / "validation.parquet"))
    except KeyError:
        print("  ebnerd_large has no validation/behaviors.parquet; popularity uses the train week only")
    articles = extract(args.large, "articles.parquet", args.work / "articles.parquet")
    article_tables = [articles]
    try:
        # Test-period articles are only in the test-set bundle; list it first.
        article_tables.insert(0, extract(args.testzip, "ebnerd_testset/articles.parquet", args.work / "test_articles.parquet"))
    except KeyError:
        print("  ebnerd_testset has no articles.parquet; using ebnerd_large articles only")
    history = extract(args.testzip, "ebnerd_testset/test/history.parquet", args.work / "history.parquet")
    test = extract(args.testzip, "ebnerd_testset/test/behaviors.parquet", args.work / "test.parquet")

    print("Training the Q2 re-ranker on the small-scale processed store...")
    # Match the production schema used by the serving-parity evaluation.
    # Test data does not provide prior within-session aggregates.
    columns = ranker_feature_names(serving_only=True) + ["bm25_score"]
    train_features = build_dataset_features(args.store, "ebnerd", "train", 0)
    model = train_model(train_features, columns)

    print("Loading large-catalog article metadata...")
    category_by_id, topics_by_id, published_by_id, titles, bm25_index = load_article_metadata(article_tables)
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
    popularity = load_training_popularity(behaviour_logs)
    max_pop = max(popularity.values(), default=1)
    print(f"  {len(popularity):,} clicked articles")

    print("Loading compact test-user histories...")
    preferences, history_lengths = load_compact_histories(history)
    print(f"  loaded compact histories for {len(preferences):,} users")

    args.out.mkdir(parents=True, exist_ok=True)
    prediction = args.out / "predictions.txt"
    partial = args.out / "predictions.txt.partial"
    checkpoint = args.out / "predictions.txt.resume.json"
    print("Scoring the EB-NeRD test set with the trained re-ranker...")
    fingerprint = run_fingerprint(test, columns, args.batch_size, args.limit, args.store, model)
    total = write_predictions(
        test, model, columns, category_by_id, topics_by_id, published_by_id, titles,
        bm25_index, popularity, max_pop, positions, vectors, preferences, partial,
        checkpoint, fingerprint, args.batch_size, args.limit, args.resume, args.context_cache_size,
        history_lengths,
    )
    if args.limit and total != args.limit:
        raise ValueError(f"Smoke test requested {args.limit} predictions but wrote {total}")

    os.replace(partial, prediction)
    checked = validate(test, prediction, args.batch_size, args.limit)
    bundle = args.out / "ebnerd_personalized_submission.zip"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        z.write(prediction, arcname="predictions.txt")
    checkpoint.unlink(missing_ok=True)
    print(f"Ready: {bundle}; validated {checked:,} rows")
    if args.limit:
        print("Smoke-test ZIP only: rerun without --limit before submitting.")


if __name__ == "__main__":
    main()