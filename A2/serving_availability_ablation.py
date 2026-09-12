from __future__ import annotations
import argparse
import json
from pathlib import Path

from ablation import METRIC_NAMES, paired_bootstrap, per_impression_metric_series
from features import FEATURE_NAMES
from reranker import build_dataset_features, score_validation, train_model

SERVING_RISK_FEATURES = ["popularity", "semantic_score"]
SERVING_SAFE_FEATURES = [name for name in FEATURE_NAMES if name not in SERVING_RISK_FEATURES]


def score_with_columns(train_features, valid_features, columns: list[str]) -> dict:
    model = train_model(train_features, columns)
    return score_validation(model, valid_features, columns)


def serving_availability_check(store: Path, dataset: str, limit: int) -> dict:
    full_columns = FEATURE_NAMES + ["bm25_score"]
    safe_columns = SERVING_SAFE_FEATURES + ["bm25_score"]

    train_features = build_dataset_features(store, dataset, "train", 0)
    valid_features = build_dataset_features(store, dataset, "validation", limit)

    with_all_features = score_with_columns(train_features, valid_features, full_columns)
    serving_safe_only = score_with_columns(train_features, valid_features, safe_columns)

    result = {
        "dataset": dataset,
        "dropped_features": SERVING_RISK_FEATURES,
        "reason": (
            "both depend on a batch-refreshed index (train click popularity, "
            "article_embeddings.npz) that lags for brand-new articles until "
            "the next offline refresh job runs"
        ),
        "metrics": {},
    }
    for index, name in enumerate(METRIC_NAMES):
        with_risk_features = per_impression_metric_series(with_all_features, "model_score", index)
        serving_safe = per_impression_metric_series(serving_safe_only, "model_score", index)
        result["metrics"][name] = paired_bootstrap(serving_safe, with_risk_features)
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
