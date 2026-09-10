#!/usr/bin/env python3
"""Create a validated EB-NeRD large-test Codabench submission ZIP.

Ranks every test impression's shown articles by click popularity in ebnerd_large
training data. Parquet files are processed one row group at a time.
"""
from __future__ import annotations
import argparse, shutil, zipfile
from collections import Counter
from pathlib import Path
import pyarrow.parquet as pq

TRAIN_MEMBER = "train/behaviors.parquet"
TEST_MEMBER = "ebnerd_testset/test/behaviors.parquet"

def extract_member(archive: Path, member: str, target: Path) -> Path:
    if target.exists(): return target
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle, bundle.open(member) as source, target.open("wb") as destination:
        shutil.copyfileobj(source, destination, length=1024 * 1024)
    return target

def training_popularity(path: Path) -> Counter:
    popularity: Counter = Counter(); parquet = pq.ParquetFile(path)
    for group in range(parquet.num_row_groups):
        for clicked in parquet.read_row_group(group, columns=["article_ids_clicked"]).column(0).to_pylist():
            if clicked:
                popularity.update(clicked)
        print(f"  read training group {group + 1}/{parquet.num_row_groups}")
    return popularity

def write_predictions(path: Path, popularity: Counter, prediction: Path, limit: int = 0) -> int:
    parquet = pq.ParquetFile(path); written = 0
    with prediction.open("w", encoding="utf-8") as output:
        for group in range(parquet.num_row_groups):
            table = parquet.read_row_group(group, columns=["impression_id", "article_ids_inview"])
            for impression_id, candidates in zip(table.column("impression_id").to_pylist(), table.column("article_ids_inview").to_pylist()):
                order = sorted(range(len(candidates)), key=lambda i: (-popularity[candidates[i]], i))
                ranks = [0] * len(candidates)
                for rank, position in enumerate(order, start=1): ranks[position] = rank
                output.write(f"{impression_id} [{','.join(map(str, ranks))}]\n"); written += 1
                if limit and written >= limit: return written
            print(f"  wrote {written:,} predictions (test group {group + 1}/{parquet.num_row_groups})")
    return written

def validate(path: Path, prediction: Path, limit: int = 0) -> int:
    parquet = pq.ParquetFile(path); checked = 0
    with prediction.open(encoding="utf-8") as output:
        for group in range(parquet.num_row_groups):
            table = parquet.read_row_group(group, columns=["impression_id", "article_ids_inview"])
            for impression_id, candidates in zip(table.column("impression_id").to_pylist(), table.column("article_ids_inview").to_pylist()):
                line = output.readline()
                if not line: raise ValueError("Prediction file ends before test data")
                output_id, rank_text = line.rstrip("\n").split(" ", 1); ranks = [int(value) for value in rank_text[1:-1].split(",")]
                if output_id != str(impression_id): raise ValueError(f"Impression order mismatch: {output_id} != {impression_id}")
                if len(ranks) != len(candidates) or sorted(ranks) != list(range(1, len(candidates) + 1)): raise ValueError(f"Invalid ranks for {impression_id}")
                checked += 1
                if limit and checked >= limit: return checked
        if output.readline(): raise ValueError("Prediction file contains extra rows")
    return checked

def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--large", type=Path, default=Path("ebnerd_large.zip")); p.add_argument("--test", type=Path, default=Path("ebnerd_testset.zip")); p.add_argument("--work-dir", type=Path, default=Path("data/raw/ebnerd_submission")); p.add_argument("--output-dir", type=Path, default=Path("outputs/ebnerd_submission")); p.add_argument("--limit", type=int, default=0, help="Smoke-test only; 0 writes all test rows."); a = p.parse_args()
    train = extract_member(a.large, TRAIN_MEMBER, a.work_dir / "train_behaviors.parquet"); test = extract_member(a.test, TEST_MEMBER, a.work_dir / "test_behaviors.parquet")
    a.output_dir.mkdir(parents=True, exist_ok=True); prediction = a.output_dir / "predictions.txt"; bundle = a.output_dir / "ebnerd_submission.zip"
    print("Counting training clicks..."); popularity = training_popularity(train); print(f"Found {len(popularity):,} clicked articles.")
    total = write_predictions(test, popularity, prediction, a.limit); checked = validate(test, prediction, a.limit); print(f"Validated {checked:,} rank lists.")
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as output: output.write(prediction, arcname="predictions.txt")
    print(f"Ready: {bundle} ({total:,} predictions)")
    if a.limit: print("Smoke-test ZIP only - rerun without --limit before submitting.")
if __name__ == "__main__": main()
