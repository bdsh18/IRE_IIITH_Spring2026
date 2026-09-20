from __future__ import annotations
import argparse
import json
from pathlib import Path

from ablation import METRIC_NAMES, paired_bootstrap, per_impression_metric_series
from features import SUBMISSION_TIME_UNAVAILABLE, ranker_feature_names
from reranker import build_dataset_features, score_validation, train_model

def score_with_columns(train_features, valid_features, columns: list[str]) -> dict:
    model = train_model(train_features, columns)
    return score_validation(model, valid_features, columns)


def serving_availability_check(store: Path, dataset: str, limit: int) -> dict:
    # This comparison must reproduce the feature schema in both competition
    # submission generators.  Popularity and semantic score are valid serving
    # inputs because they come from versioned, batch-refreshed artifacts;
    # within-session aggregates are absent from the released test files.
    full_columns = ranker_feature_names(serving_only=False) + ["bm25_score"]
    serving_columns = ranker_feature_names(serving_only=True) + ["bm25_score"]

    train_features = build_dataset_features(store, dataset, "train", 0)
    valid_features = build_dataset_features(store, dataset, "validation", limit)

    with_all_features = score_with_columns(train_features, valid_features, full_columns)
    serving_only = score_with_columns(train_features, valid_features, serving_columns)

    result = {
        "dataset": dataset,
        "comparison": "offline_full_vs_submission_matched_serving_only",
        "offline_full_columns": full_columns,
        "serving_only_columns": serving_columns,
        "dropped_features": list(SUBMISSION_TIME_UNAVAILABLE),
        "reason": (
            "competition test impressions do not include preceding impressions, "
            "preceding session clicks, or a preceding dwell-time trace. Both "
            "submission generators therefore set these fields to zero and train "
            "with them excluded. Popularity and semantic score are retained from "
            "time-safe, versioned batch artifacts."
        ),
        "metrics": {},
    }
    for index, name in enumerate(METRIC_NAMES):
        offline_full = per_impression_metric_series(with_all_features, "model_score", index)
        submission_matched = per_impression_metric_series(serving_only, "model_score", index)
        result["metrics"][name] = paired_bootstrap(submission_matched, offline_full)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=Path, default=Path("data/processed"))
    parser.add_argument("--dataset", choices=["mind", "ebnerd"], required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("outputs/q9_serving_availability.json"))
    args = parser.parse_args()

    result = serving_availability_check(args.store, args.dataset, args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
