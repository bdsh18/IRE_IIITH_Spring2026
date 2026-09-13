from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from features import FEATURE_NAMES
from mind_personalized_ranker import metrics
from reranker import build_dataset_features, rank_from_scores, score_validation, train_model

METRIC_NAMES = ["auc", "mrr", "ndcg@5", "ndcg@10"]
IMPROVEMENT_FEATURE = "freshness_hours_inv"


def per_impression_metric_series(per_impression: dict, score_key: str, metric_index: int) -> dict[str, float]:
    return {
        impression_id: metrics(rank_from_scores(entry[score_key]), entry["labels"])[metric_index]
        for impression_id, entry in per_impression.items()
    }


def paired_bootstrap(baseline: dict[str, float], improved: dict[str, float], samples: int = 2000, seed: int = 42) -> dict:
    keys = list(baseline)
    deltas = np.array([improved[k] - baseline[k] for k in keys])
    rng = np.random.default_rng(seed)
    means = [rng.choice(deltas, len(deltas), replace=True).mean() for _ in range(samples)]
    low, high = float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))
    return {
        "baseline_mean": round(float(np.mean([baseline[k] for k in keys])), 6),
        "improved_mean": round(float(np.mean([improved[k] for k in keys])), 6),
        "delta_mean": round(float(deltas.mean()), 6),
        "ci95_low": round(low, 6),
        "ci95_high": round(high, 6),
        "excludes_zero": bool(low > 0 or high < 0),
    }

def score_with_columns(train_features, valid_features, columns: list[str]) -> dict:
    model = train_model(train_features, columns)
    return score_validation(model, valid_features, columns)


def ablate_from_scores(per_impression: dict) -> dict:
    result = {"impressions": len(per_impression), "metrics": {}}
    for index, name in enumerate(METRIC_NAMES):
        baseline = per_impression_metric_series(per_impression, "bm25_score", index)
        improved = per_impression_metric_series(per_impression, "model_score", index)
        result["metrics"][name] = paired_bootstrap(baseline, improved)
    return result


def improvement_ablation_from_scores(full_scores: dict, without_feature_scores: dict) -> dict:
    result = {"improvement_feature": IMPROVEMENT_FEATURE, "metrics": {}}
    for index, name in enumerate(METRIC_NAMES):
        baseline = per_impression_metric_series(without_feature_scores, "model_score", index)
        improved = per_impression_metric_series(full_scores, "model_score", index)
        result["metrics"][name] = paired_bootstrap(baseline, improved)
    return result

def popularity_baseline_scores(valid_features: pd.DataFrame) -> dict[str, dict]:
    per_impression = {}
    for impression_id, group in valid_features.groupby("impression_id", sort=False):
        per_impression[impression_id] = {"labels": group.label.tolist(), "popularity_score": group.popularity.tolist()}
    return per_impression


def official_baseline_ablation(valid_features: pd.DataFrame, full_scores: dict) -> dict:
    popularity_scores = popularity_baseline_scores(valid_features)
    result = {"baseline": "most_popular", "metrics": {}}
    for index, name in enumerate(METRIC_NAMES):
        baseline = per_impression_metric_series(popularity_scores, "popularity_score", index)
        improved = per_impression_metric_series(full_scores, "model_score", index)
        result["metrics"][name] = paired_bootstrap(baseline, improved)
    return result

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=Path, default=Path("data/processed"))
    parser.add_argument("--dataset", choices=["mind", "ebnerd"], required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("outputs/q3_ablation.json"))
    args = parser.parse_args()

    full_columns = FEATURE_NAMES + ["bm25_score"]
    without_feature_columns = [name for name in FEATURE_NAMES if name != IMPROVEMENT_FEATURE] + ["bm25_score"]

    train_features = build_dataset_features(args.store, args.dataset, "train", 0)
    valid_features = build_dataset_features(args.store, args.dataset, "validation", args.limit)

    full_scores = score_with_columns(train_features, valid_features, full_columns)
    without_feature_scores = score_with_columns(train_features, valid_features, without_feature_columns)

    result = {
        "dataset": args.dataset,
        "reranker_vs_bm25": {"dataset": args.dataset, **ablate_from_scores(full_scores)},
        "reranker_vs_official_baseline": {"dataset": args.dataset, **official_baseline_ablation(valid_features, full_scores)},
        "feature_ablation": {"dataset": args.dataset, **improvement_ablation_from_scores(full_scores, without_feature_scores)},
    }
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()