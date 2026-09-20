from __future__ import annotations
import argparse
import bisect
from collections import Counter, defaultdict
from pathlib import Path
import math
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

FEATURE_NAMES = [
    "category_affinity",
    "topic_affinity",
    "semantic_score",
    "popularity",
    "freshness_hours_inv",
    "session_prior_impressions",
    "session_clicks_so_far",
    "session_mean_dwell_time",
    "position_bias",
    "history_length",
]

# The large competition test files do not expose prior impressions inside a
# session, prior session clicks, or a previous dwell-time trace.  Both
# submission generators therefore supply zero for all three fields.  Keep this
# list as the single source of truth for the submission model and for the
# serving-parity evaluation.
SUBMISSION_TIME_UNAVAILABLE = [
    "session_prior_impressions",
    "session_clicks_so_far",
    "session_mean_dwell_time",
]


def ranker_feature_names(serving_only: bool = False) -> list[str]:
    """Return behavioural feature columns available to a ranker.

    ``serving_only=True`` exactly mirrors the feature schema used by the
    Codabench generators.  Popularity and semantic score remain available:
    they are served from versioned, batch-refreshed artifacts, so they are not
    silently treated as unavailable session signals.
    """
    if not serving_only:
        return list(FEATURE_NAMES)
    unavailable = set(SUBMISSION_TIME_UNAVAILABLE)
    return [name for name in FEATURE_NAMES if name not in unavailable]

def optional_column(frame: pd.DataFrame, name: str, default):
    if name in frame.columns:
        return frame[name]
    return pd.Series([default] * len(frame), index=frame.index)


def train_popularity(train_impressions: pd.DataFrame) -> Counter:
    counts: Counter = Counter()
    for row in train_impressions.itertuples(index=False):
        counts.update(str(a) for a in row.clicked_ids)
    return counts


def causal_click_times(impressions: pd.DataFrame) -> dict[str, list]:
    click_times: dict[str, list] = defaultdict(list)
    for row in tqdm(impressions.itertuples(index=False), total=len(impressions), desc="Collect click times", unit="impression"):
        for article in row.clicked_ids:
            click_times[str(article)].append(row.timestamp)
    for times in click_times.values():
        times.sort()
    return click_times


def causal_popularity_snapshot(candidates: list[str], click_times: dict[str, list], as_of) -> dict[str, int]:
    return {article: bisect.bisect_left(click_times.get(article, []), as_of) for article in candidates}


def user_session_order(impressions: pd.DataFrame) -> dict[str, int]:
    ordered = impressions.sort_values("timestamp")
    order: dict[str, int] = {}
    seen: dict[tuple[str, str], int] = defaultdict(int)
    for row in ordered.itertuples(index=False):
        session_id = str(getattr(row, "session_id", "") or "")
        if not session_id:
            order[str(row.impression_id)] = 0
            continue
        key = (str(row.user_id), session_id)
        order[str(row.impression_id)] = seen[key]
        seen[key] += 1
    return order


def user_session_click_count(impressions: pd.DataFrame) -> dict[str, int]:
    ordered = impressions.sort_values("timestamp")
    counts: dict[str, int] = {}
    running: dict[tuple[str, str], int] = defaultdict(int)
    for row in ordered.itertuples(index=False):
        session_id = str(getattr(row, "session_id", "") or "")
        if not session_id:
            counts[str(row.impression_id)] = 0
            continue
        key = (str(row.user_id), session_id)
        counts[str(row.impression_id)] = running[key]
        running[key] += len(row.clicked_ids)
    return counts

def user_session_mean_dwell(impressions: pd.DataFrame) -> dict[str, float]:
    """Prior mean dwell time in the same session; zero when unavailable."""
    means, running = {}, {}
    for row in impressions.sort_values("timestamp").itertuples(index=False):
        session_id = str(getattr(row, "session_id", "") or "")
        if not session_id:
            means[str(row.impression_id)] = 0.0
            continue
        key = (str(row.user_id), session_id); total, count = running.get(key, (0.0, 0))
        means[str(row.impression_id)] = total / count if count else 0.0
        running[key] = (total + max(0.0, float(getattr(row, "dwell_time", 0.0) or 0.0)), count + 1)
    return means


def category_affinity(history_ids: list, history_weights: list, category_by_id: dict[str, str], candidate_category: str) -> float:
    weights: Counter = Counter()
    for article, weight in zip(history_ids, history_weights):
        weights[category_by_id.get(str(article), "")] += weight
    total = max(sum(weights.values()), 1e-9)
    return weights[candidate_category] / total


def topic_affinity(history_ids: list, history_weights: list, topics_by_id: dict[str, tuple], candidate_topics: tuple) -> float:
    if not candidate_topics:
        return 0.0
    topic_weights: Counter = Counter()
    for article, weight in zip(history_ids, history_weights):
        for topic in topics_by_id.get(str(article), ()):
            topic_weights[topic] += weight
    if not topic_weights:
        return 0.0
    strongest = max(topic_weights.values())
    return sum(topic_weights[t] for t in candidate_topics) / (strongest * len(candidate_topics))


def weighted_user_vector(history_ids: list, history_weights: list, positions: dict[str, int] | None, vectors: np.ndarray | None) -> np.ndarray | None:
    if positions is None or vectors is None:
        return None
    indices, article_weights = [], []
    for article, weight in zip(history_ids, history_weights):
        position = positions.get(str(article))
        if position is not None:
            indices.append(position); article_weights.append(weight)
    if not indices:
        return None
    user_vector = (vectors[indices] * np.asarray(article_weights)[:, None]).sum(axis=0)
    norm = float(np.linalg.norm(user_vector))
    if norm < 1e-12:
        return None
    return user_vector / norm


def semantic_score(history_ids: list, history_weights: list, positions: dict[str, int] | None, vectors: np.ndarray | None, candidate_id: str) -> float:
    if positions is None or vectors is None or candidate_id not in positions:
        return 0.0
    user_vector = weighted_user_vector(history_ids, history_weights, positions, vectors)
    if user_vector is None:
        return 0.0
    return float(vectors[positions[candidate_id]] @ user_vector)


def prepare_candidate_context(
    history_ids: list,
    weights: list,
    category_by_id: dict[str, str],
    topics_by_id: dict[str, tuple],
    popularity: dict,
    max_pop: int,
    positions: dict[str, int] | None,
    vectors: np.ndarray | None,
) -> dict:
    """Compute user-side signals once per impression, not once per candidate.

    Submission impressions often contain several candidates.  Category/topic
    preference and the weighted user vector are invariant across that list;
    caching them here removes repeated history scans and vector pooling while
    preserving the feature definitions used in offline evaluation.
    """
    category_weights: Counter = Counter()
    topic_weights: Counter = Counter()
    for article, weight in zip(history_ids, weights):
        article_id = str(article)
        category_weights[category_by_id.get(article_id, "")] += weight
        for topic in topics_by_id.get(article_id, ()):
            topic_weights[topic] += weight
    return {
        "category_weights": category_weights,
        "category_total": max(sum(category_weights.values()), 1e-9),
        "topic_weights": topic_weights,
        "topic_strongest": max(topic_weights.values()) if topic_weights else 0.0,
        "user_vector": weighted_user_vector(history_ids, weights, positions, vectors),
        "positions": positions,
        "vectors": vectors,
        "popularity": popularity,
        "max_pop": max_pop,
        "history_length": len(history_ids),
    }


def candidate_features_from_context(
    article: str,
    position: int,
    context: dict,
    category_by_id: dict[str, str],
    topics_by_id: dict[str, tuple],
    published_by_id: dict,
    now,
    session_count: int,
    session_clicks: int,
    session_mean_dwell: float,
) -> dict:
    """Build candidate features from a precomputed per-impression context."""
    candidate_category = category_by_id.get(article, "")
    candidate_topics = topics_by_id.get(article, ())
    category_value = context["category_weights"][candidate_category] / context["category_total"]
    topic_value = 0.0
    if candidate_topics and context["topic_weights"] and context["topic_strongest"]:
        topic_value = sum(context["topic_weights"][topic] for topic in candidate_topics) / (
            context["topic_strongest"] * len(candidate_topics)
        )
    semantic_value = 0.0
    user_vector = context["user_vector"]
    positions = context.get("positions")
    vectors = context.get("vectors")
    if user_vector is not None and positions is not None and vectors is not None:
        index = positions.get(article)
        if index is not None:
            semantic_value = float(vectors[index] @ user_vector)
    freshness = 0.0
    published = published_by_id.get(article) if published_by_id else None
    if published is not None and pd.notna(published):
        hours = max(0.0, (now - published).total_seconds() / 3600)
        freshness = math.exp(-hours / (24 * 4))
    return {
        "article_id": article,
        "category_affinity": category_value,
        "topic_affinity": topic_value,
        "semantic_score": semantic_value,
        "popularity": math.log1p(context["popularity"].get(article, 0)) / math.log1p(max(context["max_pop"], 1)),
        "freshness_hours_inv": freshness,
        "session_prior_impressions": session_count,
        "session_clicks_so_far": session_clicks,
        "session_mean_dwell_time": session_mean_dwell,
        "position_bias": 1.0 / math.log2(position + 2),
        "history_length": context["history_length"],
    }


def article_metadata(articles: pd.DataFrame) -> tuple[dict[str, str], dict[str, tuple], dict[str, object]]:
    category_by_id = dict(zip(articles.article_id.astype(str), articles.category.fillna("")))
    topics_series = optional_column(articles, "topics", None)
    topics_by_id = {
        article_id: tuple(topics) if topics is not None and hasattr(topics, "__iter__") and not isinstance(topics, str) else ()
        for article_id, topics in zip(articles.article_id.astype(str), topics_series)
    }
    published_series = optional_column(articles, "published_time", pd.NaT)
    published_by_id = dict(zip(articles.article_id.astype(str), published_series))
    return category_by_id, topics_by_id, published_by_id


def load_embeddings(store: Path, dataset: str) -> tuple[dict[str, int] | None, np.ndarray | None]:
    path = store / dataset / "article_embeddings.npz"
    if not path.exists():
        return None, None
    saved = np.load(path)
    ids = saved["article_ids"].astype(str).tolist()
    return {article: i for i, article in enumerate(ids)}, saved["vectors"].astype(np.float32)


def candidate_features(
    article: str,
    position: int,
    history_ids: list,
    weights: list,
    category_by_id: dict[str, str],
    topics_by_id: dict[str, tuple],
    popularity: dict,
    max_pop: int,
    positions: dict[str, int] | None,
    vectors: np.ndarray | None,
    published_by_id: dict,
    now,
    session_count: int,
    session_clicks: int,
    session_mean_dwell: float,
) -> dict:
    context = prepare_candidate_context(
        history_ids, weights, category_by_id, topics_by_id,
        popularity, max_pop, positions, vectors,
    )
    return candidate_features_from_context(
        article, position, context, category_by_id, topics_by_id,
        published_by_id, now, session_count, session_clicks, session_mean_dwell,
    )


def build_features(store: Path, dataset: str, split: str) -> pd.DataFrame:
    articles = pd.read_parquet(store / dataset / "articles.parquet")
    category_by_id, topics_by_id, published_by_id = article_metadata(articles)

    positions, vectors = load_embeddings(store, dataset)

    train_impressions = pd.read_parquet(store / dataset / "train_impressions.parquet")
    full_popularity = train_popularity(train_impressions)
    max_pop = max(full_popularity.values(), default=1)
    click_times = causal_click_times(train_impressions) if split == "train" else None

    impressions = pd.read_parquet(store / dataset / f"{split}_impressions.parquet")
    session_rank = user_session_order(impressions)
    session_clicks = user_session_click_count(impressions)
    session_dwell = user_session_mean_dwell(impressions)

    rows = []
    for row in impressions.itertuples(index=False):
        history_ids = list(row.history_ids)
        weights = list(row.history_recency_weights)
        candidates = [str(c) for c in row.candidate_ids]
        clicked = {str(c) for c in row.clicked_ids}
        session_count = session_rank.get(str(row.impression_id), 0)
        session_click_count = session_clicks.get(str(row.impression_id), 0)
        session_mean_dwell = session_dwell.get(str(row.impression_id), 0.0)

        popularity = causal_popularity_snapshot(candidates, click_times, row.timestamp) if split == "train" else full_popularity
        context = prepare_candidate_context(
            history_ids, weights, category_by_id, topics_by_id,
            popularity, max_pop, positions, vectors,
        )

        for position, article in enumerate(candidates):
            features = candidate_features_from_context(
                article, position, context, category_by_id, topics_by_id, published_by_id, row.timestamp,
                session_count, session_click_count, session_mean_dwell,
            )
            rows.append({
                "dataset": dataset,
                "impression_id": str(row.impression_id),
                "label": int(article in clicked),
                **features,
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=Path, default=Path("data/processed"))
    parser.add_argument("--dataset", choices=["mind", "ebnerd"], required=True)
    parser.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    features = build_features(args.store, args.dataset, args.split)
    output = args.output or (args.store / args.dataset / f"{args.split}_features.parquet")
    output.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output, index=False)
    print(f"{len(features):,} rows -> {output}")


if __name__ == "__main__":
    main()
