from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from bm25_retrieval import InvertedBM25, query_from_history, retrieve_from_history
from features import FEATURE_NAMES, build_features, ranker_feature_names
from mind_personalized_ranker import metrics

try:
    from lightgbm import LGBMRanker
except (ImportError, OSError):
    # LightGBM on macOS needs libomp.  A portable GBDT fallback keeps the
    # coursework pipeline runnable while preserving the same feature inputs.
    LGBMRanker = None
from sklearn.ensemble import HistGradientBoostingClassifier


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
    groups = features.groupby("impression_id", sort=False)
    for impression_id, group in tqdm(groups, total=groups.ngroups, desc=f"BM25 features {dataset}/{split}", unit="impression"):
        history = history_by_impression.get(impression_id, [])
        query = query_from_history(list(history), titles, 5)
        candidate_scores = index.candidate_scores(query, group.article_id.tolist())
        for article_id in group.article_id:
            records.append((impression_id, article_id, candidate_scores.get(article_id, 0.0)))
    bm25_frame = pd.DataFrame(records, columns=["impression_id", "article_id", "bm25_score"])
    return features.merge(bm25_frame, on=["impression_id", "article_id"], how="left")


def rank_from_scores(scores) -> list[int]:
    # Stable ties are important for retrieval-gated evaluation: candidates that
    # do not survive stage 1 all receive a bottom score and must retain their
    # original in-view order as the deterministic fallback.
    order = np.argsort(-np.asarray(scores, dtype=float), kind="stable")
    ranks = np.empty(len(order), dtype=int)
    ranks[order] = np.arange(1, len(order) + 1)
    return ranks.tolist()


def train_model(train_features: pd.DataFrame, columns: list[str]) -> LGBMRanker:
    if LGBMRanker is None:
        model = HistGradientBoostingClassifier(
            max_iter=200,
            max_leaf_nodes=31,
            learning_rate=0.05,
            l2_regularization=1.0,
            random_state=42,
        )
        model.fit(train_features[columns], train_features.label)
        model._a2_backend = "sklearn_hist_gradient_boosting"
        return model
    groups = train_features.groupby("impression_id", sort=False).size().tolist()
    model = LGBMRanker(objective="lambdarank", n_estimators=200, num_leaves=31, learning_rate=0.05, verbosity=-1)
    model.fit(train_features[columns], train_features.label, group=groups)
    model._a2_backend = "lightgbm_lambdarank"
    return model


def predict_scores(model, frame: pd.DataFrame) -> np.ndarray:
    """Return click-likelihood scores for either supported ranker backend."""
    if getattr(model, "_a2_backend", "lightgbm_lambdarank") == "lightgbm_lambdarank":
        return model.predict(frame)
    return model.predict_proba(frame)[:, 1]


def apply_stage1_gate(
    article_ids: list[str],
    scores: list[float],
    stage1_candidates: set[str],
    bottom_score: float = -1e12,
) -> list[float]:
    """Keep stage-2 scores only for catalog-retrieved articles.

    Competition submission files must rank every shown article, so this helper
    is deliberately used only by the *offline strict two-stage diagnostic*.
    Candidates that did not survive catalog top-K retrieval receive a stable
    bottom fallback score; they are not silently removed from the impression.
    """
    return [float(score) if article_id in stage1_candidates else bottom_score for article_id, score in zip(article_ids, scores)]


def score_validation(
    model: LGBMRanker,
    valid_features: pd.DataFrame,
    columns: list[str],
    stage1_candidates: dict[str, set[str]] | None = None,
) -> dict[str, dict]:
    per_impression = {}
    groups = valid_features.groupby("impression_id", sort=False)
    for impression_id, group in tqdm(groups, total=groups.ngroups, desc="Scoring validation", unit="impression"):
        article_ids = group.article_id.astype(str).tolist()
        bm25_scores = group.bm25_score.astype(float).tolist()
        model_scores = predict_scores(model, group[columns]).tolist()
        if stage1_candidates is not None:
            gate = stage1_candidates.get(str(impression_id), set())
            bm25_scores = apply_stage1_gate(article_ids, bm25_scores, gate)
            model_scores = apply_stage1_gate(article_ids, model_scores, gate)
        per_impression[str(impression_id)] = {
            "article_ids": article_ids,
            "labels": group.label.tolist(),
            "bm25_score": bm25_scores,
            "model_score": model_scores,
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
    cache_path = store / dataset / f"{split}_features_with_bm25.parquet"
    if limit == 0 and cache_path.exists():
        cached = pd.read_parquet(cache_path)
        required = set(FEATURE_NAMES + ["bm25_score", "label", "impression_id", "article_id"])
        if required.issubset(cached.columns):
            return cached
        # Schema changed (for example, a newly added behavioural feature).
        # Rebuild instead of silently training without it.
        cache_path.unlink()
    features = build_features(store, dataset, split)
    if limit:
        keep = features.impression_id.drop_duplicates().head(limit)
        features = features[features.impression_id.isin(keep)]
    result = attach_bm25_scores(store, dataset, split, features)
    if limit == 0:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(cache_path, index=False)
    return result


def catalog_stage1_candidates(
    store: Path,
    dataset: str,
    split: str,
    top_k: int,
    limit: int,
    recent_history: int = 5,
) -> tuple[dict[str, set[str]], dict[str, object]]:
    """Retrieve catalog top-K once and reuse it for all A2 stage-1 reports.

    The returned mapping contains *catalog* candidates.  The strict diagnostic
    later intersects this set with the logged in-view candidates, because only
    those labels are observable in the public datasets.
    """
    articles = pd.read_parquet(store / dataset / "articles.parquet")
    impressions = pd.read_parquet(store / dataset / f"{split}_impressions.parquet")
    if limit:
        impressions = impressions.head(limit)
    ids = articles.article_id.astype(str).tolist()
    titles = dict(zip(ids, articles.title.fillna("")))
    text = (articles.title.fillna("") + " " + articles.abstract.fillna("")).tolist()
    index = InvertedBM25(ids, text)

    candidate_sets: dict[str, set[str]] = {}
    recall_sums = {50: 0.0, 100: 0.0, 200: 0.0}
    recall_evaluated = coverage_evaluated = 0
    covered = total_inview = 0
    retrieve_k = max(top_k, 200)

    for row in tqdm(impressions.itertuples(index=False), total=len(impressions), desc=f"Catalog BM25 {dataset}/{split}", unit="impression"):
        impression_id = str(row.impression_id)
        history = [str(article) for article in row.history_ids]
        clicked = {str(article) for article in row.clicked_ids}
        inview = {str(article) for article in row.candidate_ids}
        query = query_from_history(history, titles, recent_history)
        retrieved = retrieve_from_history(index, history, titles, recent_history, retrieve_k) if query else []
        candidate_sets[impression_id] = set(retrieved[:top_k])

        if clicked and query:
            for k in recall_sums:
                recall_sums[k] += len(clicked.intersection(retrieved[:k])) / len(clicked)
            recall_evaluated += 1
        if inview and query:
            covered += len(inview.intersection(candidate_sets[impression_id]))
            total_inview += len(inview)
            coverage_evaluated += 1

    recall = {
        "dataset": dataset,
        "split": split,
        "retrieval_corpus_articles": len(ids),
        "evaluated_impressions": recall_evaluated,
        "history_articles_used": recent_history,
        **{f"recall@{k}": recall_sums[k] / recall_evaluated if recall_evaluated else 0.0 for k in recall_sums},
    }
    coverage = {
        "dataset": dataset,
        "split": split,
        "k": top_k,
        "evaluated_impressions": coverage_evaluated,
        "candidate_set_coverage": covered / total_inview if total_inview else 0.0,
    }
    return candidate_sets, {"generator": "BM25(title + abstract)", "top_k": top_k, "clicked_article_recall": recall, "shown_candidate_coverage": coverage}


def run(
    store: Path,
    dataset: str,
    limit: int,
    top_k: int = 200,
    serving_only: bool = False,
    strict_two_stage: bool = False,
) -> tuple[dict, LGBMRanker, list[str]]:
    columns = ranker_feature_names(serving_only) + ["bm25_score"]
    train_features = build_dataset_features(store, dataset, "train", 0)
    valid_features = build_dataset_features(store, dataset, "validation", limit)
    model = train_model(train_features, columns)
    per_impression = score_validation(model, valid_features, columns)
    stage1_sets, stage1 = catalog_stage1_candidates(store, dataset, "validation", top_k, limit)
    stage2 = {
        "description": (
            "LightGBM LambdaRank over submission-available behavioural, semantic, lexical, and article features"
            if serving_only
            else "LightGBM LambdaRank over behavioural, semantic, lexical, session, and article features"
        ),
        "in_view_reranker_metrics": summarize(per_impression),
    }
    if strict_two_stage:
        gated = score_validation(model, valid_features, columns, stage1_candidates=stage1_sets)
        stage2["retrieval_gated_inview_metrics"] = {
            "protocol": (
                "Catalog BM25 top-K gates the re-ranker. Non-retrieved logged candidates receive a stable bottom score; "
                "only in-view labels are observable, so this is an impression-grounded strict two-stage diagnostic."
            ),
            "metrics": summarize(gated),
        }
    # The top-level values intentionally retain the normal in-view ranking
    # protocol used by Codabench.  The strict retrieval-gated result is nested
    # separately rather than being misrepresented as the competition metric.
    result = {
        "dataset": dataset,
        "ranker_backend": getattr(model, "_a2_backend", "lightgbm_lambdarank"),
        "feature_policy": "serving_only" if serving_only else "offline_full",
        "feature_columns": columns,
        "two_stage": {"stage1": stage1, "stage2": stage2},
        **summarize(per_impression),
    }
    return result, model, columns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=Path, default=Path("data/processed"))
    parser.add_argument("--dataset", choices=["mind", "ebnerd"], required=True)
    parser.add_argument("--limit", type=int, default=0, help="cap validation impressions for a quick smoke test")
    parser.add_argument("--top-k", type=int, default=200, help="stage-1 BM25 candidate count")
    parser.add_argument("--serving-only", action="store_true", help="train/evaluate only with the feature columns available in both test-set submission generators")
    parser.add_argument("--strict-two-stage", action="store_true", help="also report catalog-top-K-gated in-view re-ranking metrics; submissions still rank every shown candidate")
    parser.add_argument("--output", type=Path, default=Path("outputs/q2_reranker_metrics.json"))
    args = parser.parse_args()

    result, _model, _columns = run(
        args.store, args.dataset, args.limit, args.top_k,
        serving_only=args.serving_only, strict_two_stage=args.strict_two_stage,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
