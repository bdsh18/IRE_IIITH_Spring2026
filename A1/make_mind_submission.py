#!/usr/bin/env python3
"""Create and validate a memory-efficient MIND large-test prediction ZIP.

The baseline ranks shown candidates by their click counts in MINDlarge_train.
It streams both ZIP archives, so it never extracts or loads millions of rows.
"""
from __future__ import annotations
import argparse, zipfile
from collections import Counter
from pathlib import Path

TRAIN_MEMBER = "MINDlarge_train/behaviors.tsv"
TEST_MEMBER = "MINDlarge_test/behaviors.tsv"

def click_popularity(archive: Path) -> Counter:
    counts: Counter = Counter()
    with zipfile.ZipFile(archive) as bundle, bundle.open(TRAIN_MEMBER) as raw:
        for line in raw:
            fields = line.decode("utf-8").rstrip("\n").split("\t")
            if len(fields) != 5: continue
            for item in fields[4].split():
                article, separator, label = item.rpartition("-")
                if separator and label == "1": counts[article] += 1
    return counts

def make_predictions(test_archive: Path, popularity: Counter, prediction_file: Path, limit: int = 0) -> int:
    written = 0
    with zipfile.ZipFile(test_archive) as bundle, bundle.open(TEST_MEMBER) as raw, prediction_file.open("w", encoding="utf-8") as output:
        for line in raw:
            fields = line.decode("utf-8").rstrip("\n").split("\t")
            if len(fields) != 5: raise ValueError(f"Expected five MIND columns, got {len(fields)}")
            impression_id, candidates = fields[0], fields[4].split()
            order = sorted(range(len(candidates)), key=lambda i: (-popularity[candidates[i]], i))
            ranks = [0] * len(candidates)
            for rank, position in enumerate(order, start=1): ranks[position] = rank
            output.write(f"{impression_id} [{','.join(map(str, ranks))}]\n")
            written += 1
            if written % 100_000 == 0: print(f"  wrote {written:,} predictions")
            if limit and written >= limit: break
    return written

def validate(test_archive: Path, prediction_file: Path, expected_limit: int = 0) -> int:
    checked = 0
    with zipfile.ZipFile(test_archive) as bundle, bundle.open(TEST_MEMBER) as raw, prediction_file.open(encoding="utf-8") as output:
        for source, prediction in zip(raw, output):
            source_fields = source.decode("utf-8").rstrip("\n").split("\t"); expected_id = source_fields[0]; candidates = source_fields[4].split()
            got_id, ranks_text = prediction.rstrip("\n").split(" ", 1)
            ranks = [int(value) for value in ranks_text.strip()[1:-1].split(",")]
            if got_id != expected_id: raise ValueError(f"Impression ID mismatch: {got_id} != {expected_id}")
            if len(ranks) != len(candidates) or sorted(ranks) != list(range(1, len(candidates) + 1)):
                raise ValueError(f"Invalid rank permutation for impression {expected_id}")
            checked += 1
            if expected_limit and checked >= expected_limit: break
        if not expected_limit:
            if next(raw, None) is not None or next(output, None) is not None: raise ValueError("Prediction line count differs from test impression count")
    return checked

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("MIND_data/MINDlarge_train.zip"))
    parser.add_argument("--test", type=Path, default=Path("MIND_data/MINDlarge_test.zip"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/mind_submission"))
    parser.add_argument("--limit", type=int, default=0, help="Use only for a local smoke test; 0 processes all test impressions.")
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction = args.output_dir / "prediction.txt"; submission_zip = args.output_dir / "mind_submission.zip"
    print("Counting clicks in MINDlarge_train..."); popularity = click_popularity(args.train); print(f"Found {len(popularity):,} clicked articles.")
    print("Writing ranks for MINDlarge_test..."); total = make_predictions(args.test, popularity, prediction, args.limit)
    checked = validate(args.test, prediction, args.limit); print(f"Validated {checked:,} rank lists.")
    with zipfile.ZipFile(submission_zip, "w", zipfile.ZIP_DEFLATED) as bundle: bundle.write(prediction, arcname="prediction.txt")
    print(f"Ready: {submission_zip} ({total:,} predictions)")
    if args.limit: print("Smoke-test ZIP only: rerun without --limit before submitting.")
if __name__ == "__main__": main()
