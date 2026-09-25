from __future__ import annotations
import argparse
import json
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd

from features import ranker_feature_names
from mind_personalized_ranker import metrics
from reranker import LGBMRanker, build_dataset_features, rank_from_scores, score_validation, train_model

METRIC_NAMES = ["auc", "mrr", "ndcg@5", "ndcg@10"]
# Available on both datasets, unlike freshness which is absent in MIND.
IMPROVEMENT_FEATURE = "category_affinity"


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


def improvement_ablation_from_scores(full_scores: dict, without_feature_scores: dict, feature: str = IMPROVEMENT_FEATURE) -> dict:
    result = {"improvement_feature": feature, "metrics": {}}
    for index, name in enumerate(METRIC_NAMES):
        baseline = per_impression_metric_series(without_feature_scores, "model_score", index)
        improved = per_impression_metric_series(full_scores, "model_score", index)
        result["metrics"][name] = paired_bootstrap(baseline, improved)
    return result

def temporal_popularity_baseline_scores(valid_features: pd.DataFrame) -> dict[str, dict]:
    """Score candidates with the causal popularity feature only.

    This is an internal, time-safe reference baseline.  It is deliberately
    named for what it does rather than being presented as a repository/starter
    baseline, because its implementation lives in this project.
    """
    per_impression = {}
    for impression_id, group in valid_features.groupby("impression_id", sort=False):
        per_impression[str(impression_id)] = {
            "labels": group.label.tolist(),
            "popularity_score": group.popularity.tolist(),
        }
    return per_impression


def temporal_popularity_baseline_ablation(valid_features: pd.DataFrame, full_scores: dict) -> dict:
    popularity_scores = temporal_popularity_baseline_scores(valid_features)
    result = {
        "baseline": "causal_temporal_popularity",
        "baseline_source": "internal_feature_store",
        "baseline_description": (
            "Ranks with the project’s causal popularity feature computed from "
            "behaviour strictly before each impression."
        ),
        "metrics": {},
    }
    for index, name in enumerate(METRIC_NAMES):
        baseline = per_impression_metric_series(popularity_scores, "popularity_score", index)
        improved = per_impression_metric_series(full_scores, "model_score", index)
        result["metrics"][name] = paired_bootstrap(baseline, improved)
    return result


def _prediction_lines(path: Path):
    """Yield lines from a plain prediction text file or a submission ZIP.

    Both supported competitions use the same ``impression_id [ranks]`` syntax,
    while their archives differ only in whether the member is called
    ``prediction.txt`` or ``predictions.txt``.
    """
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            candidates = [
                name for name in archive.namelist()
                if name.rsplit("/", 1)[-1] in {"prediction.txt", "predictions.txt"}
            ]
            if len(candidates) != 1:
                raise ValueError(
                    f"{path} must contain exactly one prediction.txt or predictions.txt member; "
                    f"found {candidates}"
                )
            with archive.open(candidates[0]) as source:
                for raw_line in source:
                    yield raw_line.decode("utf-8")
        return

    with path.open(encoding="utf-8") as source:
        yield from source


def starter_prediction_scores(prediction_path: Path, valid_features: pd.DataFrame) -> dict[str, dict]:
    """Parse a starter-system rank file against the current validation rows.

    The parser is intentionally strict: every validation impression must occur
    once, with a full permutation of ranks.  That prevents an accidental
    comparison between different validation subsets or candidate orders.
    """
    if not prediction_path.exists():
        raise FileNotFoundError(f"Starter prediction file was not found: {prediction_path}")

    expected: dict[str, list[int]] = {
        str(impression_id): group.label.astype(int).tolist()
        for impression_id, group in valid_features.groupby("impression_id", sort=False)
    }
    parsed: dict[str, dict] = {}
    for line_number, line in enumerate(_prediction_lines(prediction_path), start=1):
        stripped = line.strip()
        if not stripped:
            raise ValueError(f"Blank prediction line at {line_number} in {prediction_path}")
        try:
            impression_id, ranks_text = stripped.split(maxsplit=1)
            ranks = json.loads(ranks_text)
        except (ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"Malformed rank list at line {line_number} in {prediction_path}") from exc
        impression_id = str(impression_id)
        if impression_id not in expected:
            raise ValueError(f"Unexpected impression ID {impression_id!r} at line {line_number} in {prediction_path}")
        if impression_id in parsed:
            raise ValueError(f"Duplicate impression ID {impression_id!r} in {prediction_path}")
        if not isinstance(ranks, list) or any(isinstance(rank, bool) or not isinstance(rank, int) for rank in ranks):
            raise ValueError(f"Ranks for {impression_id!r} must be a JSON list of integers")
        expected_count = len(expected[impression_id])
        if len(ranks) != expected_count or sorted(ranks) != list(range(1, expected_count + 1)):
            raise ValueError(
                f"Ranks for {impression_id!r} must be a permutation of 1..{expected_count}; got {ranks}"
            )
        # Smaller rank means a better position, whereas the common scoring
        # convention used by ``rank_from_scores`` sorts larger values first.
        parsed[impression_id] = {
            "labels": expected[impression_id],
            "starter_score": [-float(rank) for rank in ranks],
        }

    missing = list(set(expected).difference(parsed))
    if missing:
        preview = ", ".join(sorted(missing)[:5])
        raise ValueError(
            f"Starter predictions are missing {len(missing)} validation impressions "
            f"(for example: {preview})"
        )
    return parsed


def starter_prediction_ablation(
    prediction_path: Path,
    valid_features: pd.DataFrame,
    full_scores: dict,
) -> dict:
    """Compare supplied repository/starter predictions with the full ranker."""
    starter_scores = starter_prediction_scores(prediction_path, valid_features)
    result = {
        "baseline": "external_starter_predictions",
        "baseline_source": "user_supplied_prediction_file",
        "prediction_file": str(prediction_path),
        "metrics": {},
    }
    for index, name in enumerate(METRIC_NAMES):
        baseline = per_impression_metric_series(starter_scores, "starter_score", index)
        improved = per_impression_metric_series(full_scores, "model_score", index)
        result["metrics"][name] = paired_bootstrap(baseline, improved)
    return result

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=Path, default=Path("data/processed"))
    parser.add_argument("--dataset", choices=["mind", "ebnerd"], required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--serving-only",
        action="store_true",
        help="train/evaluate only with the feature columns used by the competition submission generators",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/q3_ablation.json"))
    parser.add_argument(
        "--improvement-feature",
        default=IMPROVEMENT_FEATURE,
        help="feature removed in the controlled ablation (category_affinity exists in both datasets; "
             "freshness_hours_inv is EB-NeRD only)",
    )
    parser.add_argument(
        "--starter-predictions",
        type=Path,
        help=(
            "optional plain prediction text file or ZIP emitted by an upstream starter system; "
            "it is strictly aligned to this validation split before comparison"
        ),
    )
    args = parser.parse_args()

    ranker_features = ranker_feature_names(serving_only=args.serving_only)
    if args.improvement_feature not in ranker_features:
        parser.error(f"--improvement-feature must be one of {ranker_features}")
    full_columns = ranker_features + ["bm25_score"]
    without_feature_columns = [name for name in ranker_features if name != args.improvement_feature] + ["bm25_score"]

    train_features = build_dataset_features(args.store, args.dataset, "train", 0)
    valid_features = build_dataset_features(args.store, args.dataset, "validation", args.limit)

    full_scores = score_with_columns(train_features, valid_features, full_columns)
    without_feature_scores = score_with_columns(train_features, valid_features, without_feature_columns)

    result = {
        "dataset": args.dataset,
        "feature_policy": "submission_matched_serving_only" if args.serving_only else "offline_full",
        "ranker_backend": "lightgbm_lambdarank" if LGBMRanker is not None else "sklearn_hist_gradient_boosting",
        "feature_columns": full_columns,
        "impressions": len(full_scores),
        "reranker_vs_bm25": {"dataset": args.dataset, **ablate_from_scores(full_scores)},
        "reranker_vs_temporal_popularity_baseline": {
            "dataset": args.dataset,
            **temporal_popularity_baseline_ablation(valid_features, full_scores),
        },
        "feature_ablation": {
            "dataset": args.dataset,
            **improvement_ablation_from_scores(full_scores, without_feature_scores, args.improvement_feature),
        },
        "external_starter_baseline": {
            "status": "not_supplied",
            "note": (
                "No external starter prediction file was provided. Run with "
                "--starter-predictions PATH after generating rankings on the identical validation split."
            ),
        },
    }
    if args.starter_predictions:
        result["reranker_vs_external_starter_baseline"] = {
            "dataset": args.dataset,
            **starter_prediction_ablation(args.starter_predictions, valid_features, full_scores),
        }
        result["external_starter_baseline"] = {
            "status": "evaluated",
            "prediction_file": str(args.starter_predictions),
        }
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()