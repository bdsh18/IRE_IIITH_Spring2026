from __future__ import annotations
import argparse
import hashlib
import json
import os
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from bm25_retrieval import InvertedBM25, query_from_history
from build_pipeline import recency_weights
from features import (
    candidate_features_from_context,
    prepare_candidate_context,
    ranker_feature_names,
)
from mind_personalized_ranker import member, popularity as click_popularity
from reranker import build_dataset_features, predict_scores, train_model

NEWS_MEMBER = "/news.tsv"
BEHAVIORS_MEMBER = "/behaviors.tsv"
CHECKPOINT_VERSION = 1


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


def load_entity_vectors(archives: list[Path]) -> dict[str, np.ndarray]:
    """Union of entity_embedding.vec across splits (dev/test add new entities)."""
    vectors: dict[str, np.ndarray] = {}
    for archive in archives:
        try:
            vec_member = member(archive, "/entity_embedding.vec")
        except StopIteration:
            continue
        with zipfile.ZipFile(archive) as bundle, bundle.open(vec_member) as raw:
            for line in tqdm(raw, desc=f"Entity vectors ({archive.name})", unit="entity"):
                fields = line.decode().rstrip().split("\t")
                if len(fields) > 1 and fields[0] not in vectors:
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


def build_feature_rows(history_ids, weights, candidates, category_by_id, popularity, max_pop, positions, vectors, bm25_index, titles):
    query = query_from_history(history_ids, titles, 5)
    candidate_scores = bm25_index.candidate_scores(query, candidates)
    context = prepare_candidate_context(
        history_ids, weights, category_by_id, {}, popularity, max_pop, positions, vectors,
    )
    return [
        candidate_features_from_context(
            article, position, context, category_by_id, {}, {}, None, 0, 0, 0.0,
        ) | {"bm25_score": candidate_scores.get(article, 0.0)}
        for position, article in enumerate(candidates)
    ]


def validate_prediction_line(expected_id: str, candidates: list[str], prediction: str) -> None:
    if not prediction.endswith("\n"):
        raise ValueError(f"Incomplete prediction line for impression {expected_id}")
    got_id, ranks_text = prediction.rstrip("\n").split(" ", 1)
    ranks = [int(value) for value in ranks_text.strip()[1:-1].split(",")]
    if got_id != expected_id:
        raise ValueError(f"Impression ID mismatch: {got_id} != {expected_id}")
    if len(ranks) != len(candidates) or sorted(ranks) != list(range(1, len(candidates) + 1)):
        raise ValueError(f"Invalid rank permutation for impression {expected_id}")


def validate_prefix(test_archive: Path, prediction_file: Path, limit: int = 0) -> int:
    """Validate the complete, source-ordered prefix in a resumable partial file."""
    checked = 0
    with zipfile.ZipFile(test_archive) as bundle, bundle.open(member(test_archive, BEHAVIORS_MEMBER)) as raw, prediction_file.open(encoding="utf-8") as output:
        for source in raw:
            if limit and checked >= limit:
                break
            prediction = output.readline()
            if not prediction:
                break
            source_fields = source.decode("utf-8").rstrip("\n").split("\t")
            validate_prediction_line(source_fields[0], source_fields[4].split(), prediction)
            checked += 1
        if output.readline():
            raise ValueError("Partial prediction file contains rows after its validated source prefix")
    return checked


def archive_member_fingerprint(archive: Path, requested_member: str) -> dict:
    actual_member = member(archive, requested_member)
    with zipfile.ZipFile(archive) as bundle:
        info = bundle.getinfo(actual_member)
    stat = archive.stat()
    return {
        "archive": str(archive.resolve()), "archive_size": stat.st_size,
        "archive_mtime_ns": stat.st_mtime_ns, "member": actual_member,
        "member_crc": info.CRC, "member_size": info.file_size,
    }


def source_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_fingerprint(test_archive: Path, columns: list[str], batch_size: int, limit: int, store: Path, model) -> str:
    root = Path(__file__).resolve().parent
    cache = store / "mind" / "train_features_with_bm25.parquet"
    cache_metadata = None
    if cache.exists():
        stat = cache.stat()
        cache_metadata = {"path": str(cache.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    payload = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "test": archive_member_fingerprint(test_archive, BEHAVIORS_MEMBER),
        "columns": columns, "batch_size": batch_size, "limit": limit,
        "ranker_backend": getattr(model, "_a2_backend", "lightgbm_lambdarank"),
        "training_cache": cache_metadata,
        "source_hashes": {
            name: source_hash(root / name)
            for name in ("make_mind_submission.py", "bm25_retrieval.py", "features.py", "reranker.py")
        },
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def write_checkpoint(checkpoint: Path, payload: dict) -> None:
    temporary = checkpoint.with_name(f"{checkpoint.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    os.replace(temporary, checkpoint)


def resume_count(test_archive: Path, partial: Path, checkpoint: Path, fingerprint: str, limit: int) -> int:
    if not partial.exists() or not checkpoint.exists():
        raise FileNotFoundError("--resume needs both the .partial prediction file and its checkpoint JSON")
    state = json.loads(checkpoint.read_text())
    if state.get("fingerprint") != fingerprint:
        raise ValueError("Resume checkpoint does not match this test archive, model, feature store, or code version")
    completed, byte_offset = int(state["completed"]), int(state["byte_offset"])
    with partial.open("r+b") as output:
        output.seek(0, os.SEEK_END)
        if output.tell() < byte_offset:
            raise ValueError("Partial prediction file is shorter than its checkpoint")
        # Discard an uncheckpointed final write after an interruption.
        output.truncate(byte_offset)
    verified = validate_prefix(test_archive, partial, limit)
    if verified != completed:
        raise ValueError(f"Resume prefix validation found {verified} rows; checkpoint expects {completed}")
    return completed


def write_predictions(test_archive: Path, model, columns, category_by_id, popularity, max_pop, positions, vectors, bm25_index, titles, partial: Path, checkpoint: Path, fingerprint: str, batch_size: int = 5000, limit: int = 0, resume: bool = False) -> int:
    written = resume_count(test_archive, partial, checkpoint, fingerprint, limit) if resume else 0
    if limit and written > limit:
        raise ValueError("Checkpoint contains more rows than the requested --limit")
    buffer_meta: list[tuple[str, list[str]]] = []
    buffer_rows: list[dict] = []
    mode = "ab" if written else "wb"
    with zipfile.ZipFile(test_archive) as bundle, bundle.open(member(test_archive, BEHAVIORS_MEMBER)) as raw, partial.open(mode) as output:
        for _ in range(written):
            if not raw.readline():
                raise ValueError("Resume checkpoint exceeds the number of test impressions")

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
            if len(buffer_meta) >= batch_size or (limit and written + len(buffer_meta) >= limit):
                flush()
            if limit and written >= limit:
                break
        flush()
    return written


def validate(test_archive: Path, prediction_file: Path, expected_limit: int = 0) -> int:
    checked = 0
    with zipfile.ZipFile(test_archive) as bundle, bundle.open(member(test_archive, BEHAVIORS_MEMBER)) as raw, prediction_file.open(encoding="utf-8") as output:
        for source in tqdm(raw, desc="Validate MIND predictions", unit="impression"):
            if expected_limit and checked >= expected_limit:
                break
            prediction = output.readline()
            if not prediction:
                raise ValueError("Prediction line count is smaller than the expected test impression count")
            source_fields = source.decode("utf-8").rstrip("\n").split("\t")
            validate_prediction_line(source_fields[0], source_fields[4].split(), prediction)
            checked += 1
        if expected_limit and checked != expected_limit:
            raise ValueError(f"Expected {expected_limit} predictions but validated {checked}")
        if output.readline():
            raise ValueError("Prediction line count is larger than the expected test impression count")
    return checked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("MIND_data/MINDlarge_train.zip"))
    parser.add_argument("--dev", type=Path, default=Path("MIND_data/MINDlarge_dev.zip"))
    parser.add_argument("--test", type=Path, default=Path("MIND_data/MINDlarge_test.zip"))
    parser.add_argument("--store", type=Path, default=Path("data/processed"), help="small-scale processed store used to TRAIN the re-ranker")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/mind_submission"))
    parser.add_argument("--limit", type=int, default=0, help="Use only for a local smoke test; 0 processes all test impressions.")
    parser.add_argument("--batch-size", type=int, default=5000, help="impressions per durable checkpoint")
    parser.add_argument("--resume", action="store_true", help="resume a matching interrupted .partial prediction run")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction = args.output_dir / "prediction.txt"
    partial = args.output_dir / "prediction.txt.partial"
    checkpoint = args.output_dir / "prediction.txt.resume.json"
    submission_zip = args.output_dir / "mind_submission.zip"

    print("Training the Q2 re-ranker on the small-scale processed store...")
    # Keep the production model schema identical to the serving-parity
    # evaluation.  Test impressions do not expose prior session aggregates.
    columns = ranker_feature_names(serving_only=True) + ["bm25_score"]
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
    entity_vectors = load_entity_vectors(archives)
    positions, vectors = load_article_embeddings(archives, entity_vectors)
    print(f"  {len(positions) if positions else 0:,} articles have an entity embedding")

    print("Counting clicks in MINDlarge_train + MINDlarge_dev (both precede the test week)...")
    popularity = click_popularity(args.train) + click_popularity(args.dev)
    max_pop = max(popularity.values(), default=1)
    print(f"  {len(popularity):,} clicked articles")

    print("Scoring MINDlarge_test with the trained re-ranker...")
    fingerprint = run_fingerprint(args.test, columns, args.batch_size, args.limit, args.store, model)
    total = write_predictions(
        args.test, model, columns, category_by_id, popularity, max_pop, positions, vectors,
        bm25_index, titles, partial, checkpoint, fingerprint, args.batch_size, args.limit, args.resume,
    )
    if args.limit and total != args.limit:
        raise ValueError(f"Smoke test requested {args.limit} predictions but wrote {total}")
    os.replace(partial, prediction)
    checked = validate(args.test, prediction, args.limit)
    print(f"Validated {checked:,} rank lists.")
    with zipfile.ZipFile(submission_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as bundle:
        bundle.write(prediction, arcname="prediction.txt")
    checkpoint.unlink(missing_ok=True)
    print(f"Ready: {submission_zip} ({total:,} predictions)")
    if args.limit:
        print("Smoke-test ZIP only: rerun without --limit before submitting.")


if __name__ == "__main__":
    main()