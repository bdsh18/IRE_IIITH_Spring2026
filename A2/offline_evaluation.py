from __future__ import annotations
import argparse, json, math
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from bm25_retrieval import InvertedBM25, query_from_history
from features import ranker_feature_names, weighted_user_vector
from reranker import (
    apply_stage1_gate,
    build_dataset_features,
    catalog_stage1_candidates,
    predict_scores,
    train_model,
)

def auc(scores, labels):
    p = [s for s, y in zip(scores, labels) if y]; n = [s for s, y in zip(scores, labels) if not y]
    if not p or not n: return float("nan")
    return sum(1 if x > y else .5 if x == y else 0 for x in p for y in n) / (len(p) * len(n))

def mrr(ranked, clicked): return next((1 / (i + 1) for i, item in enumerate(ranked) if item in clicked), 0.0)
def ndcg(ranked, clicked, k):
    observed = sum(1 / math.log2(i + 2) for i, item in enumerate(ranked[:k]) if item in clicked)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(clicked))))
    return observed / ideal if ideal else 0.0

def ci(values, samples=500, seed=42):
    data = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if not len(data): return {"mean": None, "ci95_low": None, "ci95_high": None}
    rng = np.random.default_rng(seed); means = [rng.choice(data, len(data), replace=True).mean() for _ in range(samples)]
    return {"mean": round(float(data.mean()), 6), "ci95_low": round(float(np.quantile(means, .025)), 6), "ci95_high": round(float(np.quantile(means, .975)), 6)}

def coverage_ci(recommendations, catalog_size, samples=500, seed=42):
    if not recommendations: return {"mean": None, "ci95_low": None, "ci95_high": None}
    rng = np.random.default_rng(seed); point = len(set(item for rec in recommendations for item in rec)) / catalog_size
    values = [len(set(item for index in rng.integers(0, len(recommendations), len(recommendations)) for item in recommendations[index])) / catalog_size for _ in range(samples)]
    return {"mean": round(float(np.mean(values)), 6), "ci95_low": round(float(np.quantile(values, .025)), 6), "ci95_high": round(float(np.quantile(values, .975)), 6), "observed_full_sample": round(point, 6)}

def catalog_popularity_median(popularity: Counter, catalog_ids: list[str]) -> float:
    """Median over articles that were actually clicked at least once, not the
    whole catalog. Most catalog articles on MIND-small/EB-NeRD demo have zero
    train-period clicks, so a median over ALL articles collapses to 0 — every
    impression's clicked-article popularity would then be >= the threshold,
    and the tail_popularity slice in run_method would end up empty or tiny."""
    values = [count for count in popularity.values() if count > 0]
    return float(np.median(values)) if values else 0.0

def run_method(name, rows, score_fn, vectors, positions, popularity, catalog_size, popularity_median):
    metric_values = {key: [] for key in ["auc", "mrr", "ndcg@5", "ndcg@10", "diversity@10", "novelty@10"]}; recommendations = []
    slice_names = ["cold_history_under_5", "warm_history_5_or_more", "head_popularity", "tail_popularity"]
    slice_values = {slice_name: {key: [] for key in metric_values} for slice_name in slice_names}
    total_clicks = max(sum(popularity.values()), 1)
    for row in tqdm(rows.itertuples(index=False), total=len(rows), desc=f"Evaluate {name}", unit="impression"):
        candidates = [str(x) for x in row.candidate_ids]; clicked = {str(x) for x in row.clicked_ids}
        if not clicked: continue
        scores = score_fn(row, candidates); ranked = sorted(candidates, key=lambda x: -scores.get(x, float("-inf"))); top = ranked[:10]
        labels = [int(x in clicked) for x in candidates]; values = {"auc": auc([scores.get(x, float("-inf")) for x in candidates], labels), "mrr": mrr(ranked, clicked), "ndcg@5": ndcg(ranked, clicked, 5), "ndcg@10": ndcg(ranked, clicked, 10)}
        known = [positions[x] for x in top if x in positions]
        values["diversity@10"] = float(np.mean([1 - float(vectors[a] @ vectors[b]) for i, a in enumerate(known) for b in known[i + 1:]])) if len(known) > 1 else 0.0
        values["novelty@10"] = float(np.mean([-math.log2(max(popularity[x], 1) / total_clicks) for x in top])) if top else 0.0
        user_bucket = "cold_history_under_5" if len(row.history_ids) < 5 else "warm_history_5_or_more"
        clicked_popularity = float(np.mean([popularity.get(c, 0) for c in clicked]))
        item_bucket = "head_popularity" if clicked_popularity >= popularity_median else "tail_popularity"
        for key, value in values.items():
            metric_values[key].append(value)
            slice_values[user_bucket][key].append(value)
            slice_values[item_bucket][key].append(value)
        recommendations.append(top)
    answer = {"method": name, "evaluated_impressions": len(recommendations), **{key: ci(values) for key, values in metric_values.items()}, "coverage@10": coverage_ci(recommendations, catalog_size)}
    answer["slices"] = {slice_name: {key: ci(values) for key, values in data.items()} for slice_name, data in slice_values.items()}
    return answer

def reranker_scores(
    store: Path,
    dataset: str,
    limit: int = 0,
    serving_only: bool = False,
) -> dict[str, dict[str, float]]:
    columns = ranker_feature_names(serving_only=serving_only) + ["bm25_score"]
    train_features = build_dataset_features(store, dataset, "train", 0)
    valid_features = build_dataset_features(store, dataset, "validation", limit)
    model = train_model(train_features, columns)
    valid_features = valid_features.assign(score=predict_scores(model, valid_features[columns]))
    scores: dict[str, dict[str, float]] = {}
    for row in valid_features.itertuples(index=False):
        scores.setdefault(row.impression_id, {})[row.article_id] = row.score
    return scores

def evaluate_dataset(
    store: Path,
    dataset: str,
    limit: int,
    top_k: int = 200,
    serving_only: bool = False,
    strict_two_stage: bool = False,
):
    articles = pd.read_parquet(store / dataset / "articles.parquet"); train = pd.read_parquet(store / dataset / "train_impressions.parquet"); rows = pd.read_parquet(store / dataset / "validation_impressions.parquet")
    if limit: rows = rows.head(limit)
    ids = articles.article_id.astype(str).tolist(); titles = dict(zip(ids, articles.title.fillna(""))); text = (articles.title.fillna("") + " " + articles.abstract.fillna("")).tolist(); bm25 = InvertedBM25(ids, text)
    saved = np.load(store / dataset / "article_embeddings.npz"); emb_ids = saved["article_ids"].astype(str).tolist(); vectors = saved["vectors"].astype(np.float32); positions = {article: i for i, article in enumerate(emb_ids)}
    popularity = Counter(str(x) for history in train.clicked_ids for x in history)
    popularity_median = catalog_popularity_median(popularity, ids)
    def lexical(row, candidates): return bm25.candidate_scores(query_from_history(row.history_ids, titles, 5), candidates)
    def semantic(row, candidates):
        user = weighted_user_vector(row.history_ids, row.history_recency_weights, positions, vectors)
        if user is None: return {x: 0.0 for x in candidates}
        return {x: float(vectors[positions[x]] @ user) if x in positions else 0.0 for x in candidates}
    reranker_lookup = reranker_scores(store, dataset, limit, serving_only=serving_only)
    feature_policy = "submission_matched_serving_only" if serving_only else "offline_full"
    def reranked(row, candidates):
        per_impression = reranker_lookup.get(str(row.impression_id), {})
        return {x: per_impression.get(x, float("-inf")) for x in candidates}
    methods = [
        run_method("bm25", rows, lexical, vectors, positions, popularity, len(ids), popularity_median),
        run_method("semantic", rows, semantic, vectors, positions, popularity, len(ids), popularity_median),
        run_method(f"in_view_reranker_{feature_policy}", rows, reranked, vectors, positions, popularity, len(ids), popularity_median),
    ]
    stage1_sets, stage1 = catalog_stage1_candidates(store, dataset, "validation", top_k, limit)
    if strict_two_stage:
        def retrieval_gated_reranked(row, candidates):
            per_impression = reranker_lookup.get(str(row.impression_id), {})
            raw_scores = [per_impression.get(x, float("-inf")) for x in candidates]
            gated_scores = apply_stage1_gate(
                candidates,
                raw_scores,
                stage1_sets.get(str(row.impression_id), set()),
            )
            return dict(zip(candidates, gated_scores))

        methods.append(run_method(
            f"retrieval_gated_in_view_reranker_{feature_policy}",
            rows,
            retrieval_gated_reranked,
            vectors,
            positions,
            popularity,
            len(ids),
            popularity_median,
        ))
    return {
        "dataset": dataset,
        "catalog_articles": len(ids),
        "feature_policy": feature_policy,
        "two_stage_pipeline": {
            "stage1": stage1,
            "stage2_ranker": (
                "LightGBM LambdaRank with submission-available behavioural, lexical, semantic, and article features"
                if serving_only
                else "LightGBM LambdaRank with behavioural, lexical, semantic, session, and article features"
            ),
            "evaluation_protocol": (
                "The in-view reranker scores every logged candidate, matching Codabench's required rank-all-candidates format. "
                "When --strict-two-stage is set, a separate retrieval-gated method keeps model scores only for catalog BM25 top-K "
                "articles and assigns a stable bottom score to the other logged candidates."
            ),
            "strict_two_stage_reported": strict_two_stage,
        },
        "methods": methods,
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--store", type=Path, default=Path("data/processed"))
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--top-k", type=int, default=200)
    p.add_argument("--serving-only", action="store_true", help="evaluate only with the feature columns available to the submission generators")
    p.add_argument("--strict-two-stage", action="store_true", help="add a catalog-top-K-gated in-view re-ranking diagnostic")
    p.add_argument("--output", type=Path, default=Path("outputs/q5_evaluation.json"))
    a = p.parse_args()
    results = [
        evaluate_dataset(a.store, d, a.limit, a.top_k, a.serving_only, a.strict_two_stage)
        for d in ["mind", "ebnerd"]
    ]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
if __name__ == "__main__": main()
