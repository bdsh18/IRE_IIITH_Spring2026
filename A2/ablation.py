from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

from mind_personalized_ranker import metrics
from reranker import build_dataset_features, rank_from_scores, run

METRIC_NAMES = ["auc", "mrr", "ndcg@5", "ndcg@10"]


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


def ablate(store: Path, dataset: str, limit: int) -> dict:
    from reranker import FEATURE_NAMES, score_validation, train_model

    columns = FEATURE_NAMES + ["bm25_score"]
    train_features = build_dataset_features(store, dataset, "train", 0)
    valid_features = build_dataset_features(store, dataset, "validation", limit)
    model = train_model(train_features, columns)
    per_impression = score_validation(model, valid_features, columns)

    result = {"dataset": dataset, "impressions": len(per_impression), "metrics": {}}
    for index, name in enumerate(METRIC_NAMES):
        baseline = per_impression_metric_series(per_impression, "bm25_score", index)
        improved = per_impression_metric_series(per_impression, "model_score", index)
        result["metrics"][name] = paired_bootstrap(baseline, improved)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=Path, default=Path("data/processed"))
    parser.add_argument("--dataset", choices=["mind", "ebnerd"], required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("outputs/q3_ablation.json"))
    args = parser.parse_args()

    result = ablate(args.store, args.dataset, args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()