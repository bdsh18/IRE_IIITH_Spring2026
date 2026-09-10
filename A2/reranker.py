from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bm25_retrieval import InvertedBM25, query_from_history
from features import FEATURE_NAMES, build_features
from mind_personalized_ranker import metrics

try:
    from lightgbm import LGBMRanker
except ImportError as exc:  # pragma: no cover
    raise SystemExit("This script needs LightGBM: pip install lightgbm --break-system-packages") from exc


def attach_bm25_scores(store: Path, dataset: str, split: str, features: pd.DataFrame) -> pd.DataFrame:
    articles = pd.read_parquet(store / dataset / "articles.parquet")
    ids = articles.article_id.astype(str).tolist()
    titles = dict(zip(ids, articles.title.fillna("")))
    text = (articles.title.fillna("") + " " + articles.abstract.fillna("")).tolist()
    index = InvertedBM25(ids, text)

    impressions = pd.read_parquet(store / dataset / f"{split}_impressions.parquet")
    impressions["impression_id"] = impressions.impression_id.astype(str)
    history_by_impression = dict(zip(impressions.impression_id, impressions.history_ids))

    records = []
    for impression_id, group in features.groupby("impression_id", sort=False):
        history = history_by_impression.get(impression_id, [])
        query = query_from_history(list(history), titles, 5)
        candidate_scores = index.candidate_scores(query, group.article_id.tolist())
        for article_id in group.article_id:
            records.append((impression_id, article_id, candidate_scores.get(article_id, 0.0)))
    bm25_frame = pd.DataFrame(records, columns=["impression_id", "article_id", "bm25_score"])
    return features.merge(bm25_frame, on=["impression_id", "article_id"], how="left")


def rank_from_scores(scores) -> list[int]:
    order = np.argsort(-np.asarray(scores, dtype=float))
    ranks = np.empty(len(order), dtype=int)
    ranks[order] = np.arange(1, len(order) + 1)
    return ranks.tolist()


def train_model(train_features: pd.DataFrame, columns: list[str]) -> LGBMRanker:
    groups = train_features.groupby("impression_id", sort=False).size().tolist()
    model = LGBMRanker(objective="lambdarank", n_estimators=200, num_leaves=31, learning_rate=0.05, verbosity=-1)
    model.fit(train_features[columns], train_features.label, group=groups)
    return model


def score_validation(model: LGBMRanker, valid_features: pd.DataFrame, columns: list[str]) -> dict[str, dict]:
    """Returns {impression_id: {"labels": [...], "bm25_score": [...], "model_score": [...]}}"""
    per_impression = {}
    for impression_id, group in valid_features.groupby("impression_id", sort=False):
        per_impression[impression_id] = {
            "labels": group.label.tolist(),
            "bm25_score": group.bm25_score.tolist(),
            "model_score": model.predict(group[columns]).tolist(),
        }
    return per_impression


def summarize(per_impression: dict[str, dict]) -> dict:
    totals = {"before_bm25_only": [0.0] * 4, "after_reranked": [0.0] * 4}
    n = 0
    for entry in per_impression.values():
        before = metrics(rank_from_scores(entry["bm25_score"]), entry["labels"])
        after = metrics(rank_from_scores(entry["model_score"]), entry["labels"])
        for name, values in [("before_bm25_only", before), ("after_reranked", after)]:
            totals[name] = [a + b for a, b in zip(totals[name], values)]
        n += 1
    metric_names = ["auc", "mrr", "ndcg@5", "ndcg@10"]
    return {name: dict(zip(metric_names, [round(v / max(n, 1), 6) for v in values])) for name, values in totals.items()} | {"impressions": n}


def build_dataset_features(store: Path, dataset: str, split: str, limit: int) -> pd.DataFrame:
    features = build_features(store, dataset, split)
    if limit:
        keep = features.impression_id.drop_duplicates().head(limit)
        features = features[features.impression_id.isin(keep)]
    return attach_bm25_scores(store, dataset, split, features)


def run(store: Path, dataset: str, limit: int) -> tuple[dict, LGBMRanker, list[str]]:
    columns = FEATURE_NAMES + ["bm25_score"]
    train_features = build_dataset_features(store, dataset, "train", 0)
    valid_features = build_dataset_features(store, dataset, "validation", limit)
    model = train_model(train_features, columns)
    per_impression = score_validation(model, valid_features, columns)
    result = {"dataset": dataset, **summarize(per_impression)}
    return result, model, columns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=Path, default=Path("data/processed"))
    parser.add_argument("--dataset", choices=["mind", "ebnerd"], required=True)
    parser.add_argument("--limit", type=int, default=0, help="cap validation impressions for a quick smoke test")
    parser.add_argument("--output", type=Path, default=Path("outputs/q2_reranker_metrics.json"))
    args = parser.parse_args()

    result, _model, _columns = run(args.store, args.dataset, args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()