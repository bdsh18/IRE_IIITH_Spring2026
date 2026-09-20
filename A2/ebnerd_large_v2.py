#!/usr/bin/env python3
"""Large-data, causal-safe EB-NeRD LambdaRank pipeline.

This is intentionally separate from ``make_ebnerd_personalized_submission.py``.
The latter remains the reproducible Assignment-2 baseline submission.  This
script is the more expensive *v2 experiment*: it learns from the labelled
``ebnerd_large`` training split and validates on its supplied later validation
split before it is allowed to score the 13.5M-row hidden test split.

The script is designed for a laptop or Colab runtime with limited RAM:

* parquet behaviour files are scanned in batches; only a deterministic sample
  of train impressions is materialised as features;
* every sampled impression stays intact, so LightGBM receives valid LambdaRank
  groups rather than randomly sampled candidate rows;
* train popularity is calculated in hourly bins and, for training rows, uses
  only clicks from *earlier* hours.  It therefore cannot look ahead to the
  current interaction's click label;
* history comes from the bundle's split-specific ``history.parquet``.  It is
  treated as the provided pre-impression history and never augmented with a
  validation/test click label;
* final prediction writes and validates a root-level ``predictions.txt`` and
  then creates a Codabench-ready ZIP.  It supports durable checkpoints.

Typical use (run these from the A2 directory):

    # Build sampled train/validation features, train, and report validation.
    .venv/bin/python ebnerd_large_v2.py --mode all \
      --large /path/to/ebnerd_large.zip \
      --embeddings /path/to/Ekstra_Bladet_word2vec.zip

    # Only after checking data/v2/ebnerd_large/validation_metrics.json:
    .venv/bin/python ebnerd_large_v2.py --mode submit \
      --large /path/to/ebnerd_large.zip \
      --testzip /path/to/ebnerd_testset.zip \
      --embeddings /path/to/Ekstra_Bladet_word2vec.zip

For a smoke test, append ``--limit 1000`` to the submit command.  A smoke ZIP
is deliberately named ``ebnerd_v2_smoke.zip`` and must not be uploaded.
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import os
import pickle
import shutil
import zipfile
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm

from bm25_retrieval import InvertedBM25, query_from_history, tokenize

try:
    import lightgbm as lgb
except (ImportError, OSError) as exc:  # pragma: no cover - environment-specific
    lgb = None
    LIGHTGBM_ERROR = exc
else:
    LIGHTGBM_ERROR = None


# All columns here are available at serving time.  In particular, this list
# intentionally excludes read_time, scroll_percentage, next_read_time, and
# article_ids_clicked: they are post-impression or label-derived signals.
FEATURE_COLUMNS = [
    "bm25_score",
    "category_affinity",
    "subcategory_affinity",
    "topic_affinity",
    "entity_overlap",
    "entity_jaccard",
    "semantic_mean",
    "semantic_max",
    "popularity",
    "freshness",
    "history_length",
    "candidate_title_tokens",
    "candidate_premium",
    "candidate_sentiment",
    "position_bias",
    "device_type",
    "is_sso_user",
    "is_subscriber",
    "age_bucket",
]

FEATURE_SCHEMA = pa.schema(
    [
        pa.field("impression_id", pa.string()),
        pa.field("group_order", pa.int64()),
        pa.field("label", pa.int8()),
        *[pa.field(column, pa.float32()) for column in FEATURE_COLUMNS],
    ]
)


def default_source(filename: str) -> Path:
    """Find the data layout used in this coursework without requiring it."""
    here = Path(__file__).resolve()
    candidates = [Path(filename)]
    # .../IRA/IRE_IIITH_Spring2026/A2/ebnerd_large_v2.py -> .../IRA
    if len(here.parents) >= 3:
        candidates.append(here.parents[2] / "Assighnment1" / filename)
    return next((candidate for candidate in candidates if candidate.exists()), candidates[0])


def listify(value: object) -> tuple[str, ...]:
    """Return an immutable, null-safe ID sequence from Arrow/Pandas values."""
    if value is None:
        return ()
    if isinstance(value, float) and math.isnan(value):
        return ()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple, np.ndarray)):
        return tuple(str(item) for item in value if item is not None)
    return (str(value),)


def number(value: object, default: float = 0.0) -> float:
    """Convert nullable Arrow scalars to finite float features."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def hour_key(value: datetime) -> int:
    """A timezone-agnostic, sortable hour identifier for temporal popularity."""
    return value.year * 8_928 + value.timetuple().tm_yday * 24 + value.hour


def stable_selected(impression_id: object, modulus: int) -> bool:
    """Deterministically retain an entire impression with probability 1/modulus."""
    if modulus <= 1:
        return True
    digest = hashlib.blake2b(str(impression_id).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % modulus == 0


def ranks_from_scores(scores: Sequence[float]) -> list[int]:
    """Convert scores into Codabench's 1=best rank convention, stably on ties."""
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
    ranks = np.empty(len(order), dtype=np.int32)
    ranks[order] = np.arange(1, len(order) + 1)
    return ranks.tolist()


def find_member(archive: Path, suffix: str) -> str:
    with zipfile.ZipFile(archive) as bundle:
        matches = [
            name for name in bundle.namelist()
            if name.endswith(suffix) and not name.startswith("__MACOSX/") and "/._" not in name
        ]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one member ending {suffix!r} in {archive}; found {matches}")
    return matches[0]


def extract_member(archive: Path, member: str, target: Path) -> Path:
    """Copy one archive member atomically into this v2 work directory."""
    if target.exists() and target.stat().st_size:
        return target
    if not archive.exists():
        raise FileNotFoundError(f"Missing archive: {archive}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.partial")
    with zipfile.ZipFile(archive) as bundle, bundle.open(member) as source, temporary.open("wb") as destination:
        shutil.copyfileobj(source, destination, length=1024 * 1024)
    os.replace(temporary, target)
    return target


def require_lightgbm() -> None:
    if lgb is None:
        raise RuntimeError(
            "LightGBM could not be imported. Install libomp and then `pip install lightgbm`; "
            f"original error: {LIGHTGBM_ERROR}"
        )


@dataclass
class ArticleStore:
    """Compact catalogue metadata needed at both training and serving time."""

    category: dict[str, str]
    subcategories: dict[str, tuple[str, ...]]
    topics: dict[str, tuple[str, ...]]
    entities: dict[str, tuple[str, ...]]
    published: dict[str, datetime | None]
    title: dict[str, str]
    title_tokens: dict[str, int]
    premium: dict[str, float]
    sentiment: dict[str, float]
    bm25: InvertedBM25 | None


def load_articles(path: Path, use_bm25: bool) -> ArticleStore:
    """Load only feature columns; never load article bodies into RAM."""
    requested = [
        "article_id", "category_str", "subcategory", "topics", "ner_clusters",
        "published_time", "title", "subtitle", "premium", "sentiment_score",
    ]
    available = set(pq.ParquetFile(path).schema_arrow.names)
    missing = {"article_id", "title"} - available
    if missing:
        raise ValueError(f"Article catalogue lacks required columns: {sorted(missing)}")
    table = pq.read_table(path, columns=[column for column in requested if column in available])
    values = table.to_pydict()
    ids = [str(value) for value in values["article_id"]]
    category: dict[str, str] = {}
    subcategories: dict[str, tuple[str, ...]] = {}
    topics: dict[str, tuple[str, ...]] = {}
    entities: dict[str, tuple[str, ...]] = {}
    published: dict[str, datetime | None] = {}
    titles: dict[str, str] = {}
    title_tokens: dict[str, int] = {}
    premium: dict[str, float] = {}
    sentiment: dict[str, float] = {}
    texts: list[str] = []

    def column(name: str, default: object = None) -> list[object]:
        return values.get(name, [default] * len(ids))

    for article_id, cat, subs, article_topics, ner, timestamp, title, subtitle, is_premium, sent in zip(
        ids, column("category_str", ""), column("subcategory"), column("topics"),
        column("ner_clusters"), column("published_time"), column("title", ""),
        column("subtitle", ""), column("premium", False), column("sentiment_score", 0.0),
    ):
        title = str(title or "")
        category[article_id] = str(cat or "")
        subcategories[article_id] = listify(subs)
        topics[article_id] = listify(article_topics)
        # Named entities (rather than generic entity types) give a more useful
        # overlap signal. Empty fields are deliberately represented as ().
        entities[article_id] = listify(ner)
        published[article_id] = timestamp if isinstance(timestamp, datetime) else None
        titles[article_id] = title
        title_tokens[article_id] = float(len(tokenize(title)))
        premium[article_id] = float(bool(is_premium))
        sentiment[article_id] = number(sent)
        texts.append(f"{title} {str(subtitle or '')}")

    bm25 = InvertedBM25(ids, texts) if use_bm25 else None
    return ArticleStore(category, subcategories, topics, entities, published, titles, title_tokens, premium, sentiment, bm25)


def extract_embedding_file(embedding_archive: Path, work: Path) -> Path:
    """Extract EB's provided document vectors into the private v2 cache."""
    target = work / "embeddings" / "document_vector.parquet"
    if target.exists() and target.stat().st_size:
        return target
    return extract_member(embedding_archive, find_member(embedding_archive, "document_vector.parquet"), target)


def load_embeddings(embedding_archive: Path | None, work: Path, article_ids: set[str]) -> tuple[dict[str, int] | None, np.ndarray | None]:
    """Load L2-normalised provided article vectors, or gracefully disable them."""
    if embedding_archive is None or not embedding_archive.exists():
        print("Embedding archive not found: semantic features will be zero.")
        return None, None
    path = extract_embedding_file(embedding_archive, work)
    frame = pd.read_parquet(path)
    id_column = next((column for column in ("article_id", "article_id_fixed") if column in frame.columns), None)
    if id_column is None:
        raise ValueError(f"Cannot find article ID column in {path}")
    vector_column = None
    for column in frame.columns:
        if column == id_column:
            continue
        nonnull = frame[column].dropna()
        if len(nonnull) and hasattr(nonnull.iloc[0], "__len__"):
            vector_column = column
            break
    if vector_column is None:
        raise ValueError(f"Cannot find vector column in {path}")
    frame[id_column] = frame[id_column].astype(str)
    frame = frame[frame[id_column].isin(article_ids)]
    if frame.empty:
        return None, None
    vectors = np.vstack(frame[vector_column].to_list()).astype(np.float32)
    # The released matrix is finite, but normalising an explicit cleaned copy
    # makes the serving path robust to a future artifact containing NaN/Inf.
    np.nan_to_num(vectors, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    ids = frame[id_column].tolist()
    return {article_id: index for index, article_id in enumerate(ids)}, vectors


@dataclass
class HourlyPopularity:
    """Hourly click counts with a strict earlier-hour lookup for train rows."""

    timelines: dict[str, tuple[list[int], list[int]]]
    totals: Counter
    maximum: int

    def count(self, article_id: str, timestamp: datetime | None, causal: bool) -> int:
        if not causal or timestamp is None:
            return int(self.totals.get(article_id, 0))
        timeline = self.timelines.get(article_id)
        if timeline is None:
            return 0
        hours, prefixes = timeline
        # bisect_left excludes the full current hour, including this row.
        position = bisect.bisect_left(hours, hour_key(timestamp))
        return prefixes[position - 1] if position else 0


def popularity_cache_path(work: Path) -> Path:
    return work / "cache" / "train_hourly_popularity.pkl"


def build_hourly_popularity(train_path: Path, work: Path, force: bool = False) -> HourlyPopularity:
    """Create a reusable causal popularity index with one streaming train pass."""
    cache = popularity_cache_path(work)
    if cache.exists() and not force:
        with cache.open("rb") as source:
            return pickle.load(source)
    by_article: dict[str, Counter] = defaultdict(Counter)
    parquet = pq.ParquetFile(train_path)
    columns = ["impression_time", "article_ids_clicked"]
    for batch in tqdm(parquet.iter_batches(batch_size=65_536, columns=columns), desc="Building hourly popularity", unit="batch"):
        times = batch.column("impression_time").to_pylist()
        clicks = batch.column("article_ids_clicked").to_pylist()
        for timestamp, articles in zip(times, clicks):
            if timestamp is None:
                continue
            hour = hour_key(timestamp)
            for article_id in listify(articles):
                by_article[article_id][hour] += 1
    totals: Counter = Counter()
    timelines: dict[str, tuple[list[int], list[int]]] = {}
    for article_id, values in by_article.items():
        running = 0
        hours: list[int] = []
        prefixes: list[int] = []
        for hour, count in sorted(values.items()):
            running += count
            hours.append(hour)
            prefixes.append(running)
        timelines[article_id] = (hours, prefixes)
        totals[article_id] = running
    result = HourlyPopularity(timelines, totals, max(totals.values(), default=1))
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_suffix(".partial")
    with temporary.open("wb") as destination:
        pickle.dump(result, destination, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, cache)
    return result


def iter_behaviour_batches(path: Path, columns: list[str], batch_size: int = 65_536) -> Iterator[dict[str, list[object]]]:
    parquet = pq.ParquetFile(path)
    available = set(parquet.schema_arrow.names)
    selected = [column for column in columns if column in available]
    missing_required = {"impression_id", "impression_time", "article_ids_inview", "user_id"} - set(selected)
    if missing_required:
        raise ValueError(f"{path} lacks required behaviour columns {sorted(missing_required)}")
    for batch in parquet.iter_batches(batch_size=batch_size, columns=selected):
        yield {column: batch.column(column).to_pylist() for column in selected}


def row_value(batch: dict[str, list[object]], name: str, index: int, default: object = None) -> object:
    values = batch.get(name)
    return values[index] if values is not None else default


def selected_user_ids(behaviour_path: Path, modulus: int, maximum: int) -> set[int]:
    """First pass: keep history only for users who occur in sampled groups."""
    selected: set[int] = set()
    kept = 0
    for batch in tqdm(iter_behaviour_batches(behaviour_path, ["impression_id", "impression_time", "article_ids_inview", "user_id"]), desc=f"Selecting {behaviour_path.stem} users", unit="batch"):
        for impression_id, user_id in zip(batch["impression_id"], batch["user_id"]):
            if not stable_selected(impression_id, modulus):
                continue
            selected.add(int(user_id))
            kept += 1
            if maximum and kept >= maximum:
                return selected
    return selected


def load_histories(history_path: Path, wanted_users: set[int] | None, keep_last: int) -> dict[int, tuple[str, ...]]:
    """Read histories in batches; ``wanted_users`` prevents a huge train map."""
    history = pq.ParquetFile(history_path)
    id_column = "article_id_fixed" if "article_id_fixed" in history.schema_arrow.names else "article_id"
    result: dict[int, tuple[str, ...]] = {}
    for batch in tqdm(history.iter_batches(batch_size=65_536, columns=["user_id", id_column]), desc=f"Loading {history_path.parent.name} histories", unit="batch"):
        users = batch.column("user_id").to_pylist()
        articles = batch.column(id_column).to_pylist()
        for user_id, history_ids in zip(users, articles):
            key = int(user_id)
            if wanted_users is None or key in wanted_users:
                result[key] = listify(history_ids)[-keep_last:]
    return result


@dataclass
class UserProfile:
    """Per-history aggregations computed once, then reused for every candidate."""

    history: tuple[str, ...]
    category_weights: Counter
    subcategory_weights: Counter
    topic_weights: Counter
    entity_weights: Counter
    entity_set: set[str]
    user_vector: np.ndarray | None
    history_vectors: np.ndarray | None


def affinity(values: Sequence[str], weights: Counter, denominator: float) -> float:
    if not values or denominator <= 0:
        return 0.0
    return float(sum(weights[value] for value in values) / (denominator * len(values)))


def profile_from_history(
    history: tuple[str, ...],
    articles: ArticleStore,
    positions: dict[str, int] | None,
    vectors: np.ndarray | None,
    decay: float,
) -> UserProfile:
    category_weights: Counter = Counter()
    subcategory_weights: Counter = Counter()
    topic_weights: Counter = Counter()
    entity_weights: Counter = Counter()
    vector_indices: list[int] = []
    vector_weights: list[float] = []
    for reverse_age, article_id in enumerate(reversed(history)):
        weight = decay ** reverse_age
        category_weights[articles.category.get(article_id, "")] += weight
        for subcategory in articles.subcategories.get(article_id, ()):
            subcategory_weights[subcategory] += weight
        for topic in articles.topics.get(article_id, ()):
            topic_weights[topic] += weight
        for entity in articles.entities.get(article_id, ()):
            entity_weights[entity] += weight
        if positions is not None and vectors is not None and article_id in positions:
            vector_indices.append(positions[article_id])
            vector_weights.append(weight)
    user_vector: np.ndarray | None = None
    history_vectors: np.ndarray | None = None
    if vector_indices and vectors is not None:
        history_vectors = vectors[np.asarray(vector_indices)]
        raw_vector = (history_vectors * np.asarray(vector_weights, dtype=np.float32)[:, None]).sum(axis=0)
        norm = float(np.linalg.norm(raw_vector))
        if norm > 1e-12:
            user_vector = raw_vector / norm
    return UserProfile(
        history=history,
        category_weights=category_weights,
        subcategory_weights=subcategory_weights,
        topic_weights=topic_weights,
        entity_weights=entity_weights,
        entity_set=set(entity_weights),
        user_vector=user_vector,
        history_vectors=history_vectors,
    )


def candidate_features(
    article_id: str,
    position: int,
    profile: UserProfile,
    now: datetime | None,
    articles: ArticleStore,
    popularity: HourlyPopularity,
    causal_popularity: bool,
    embedding_positions: dict[str, int] | None,
    vectors: np.ndarray | None,
    bm25_score: float,
    device_type: object,
    is_sso_user: object,
    is_subscriber: object,
    age: object,
) -> dict[str, float]:
    candidate_entities = articles.entities.get(article_id, ())
    entity_overlap = affinity(candidate_entities, profile.entity_weights, sum(profile.entity_weights.values()))
    candidate_entity_set = set(candidate_entities)
    union = candidate_entity_set | profile.entity_set
    entity_jaccard = len(candidate_entity_set & profile.entity_set) / len(union) if union else 0.0
    semantic_mean = semantic_max = 0.0
    if embedding_positions is not None and vectors is not None:
        candidate_index = embedding_positions.get(article_id)
        if candidate_index is not None:
            candidate_vector = vectors[candidate_index]
            if profile.user_vector is not None:
                semantic_mean = float(candidate_vector @ profile.user_vector)
            if profile.history_vectors is not None and len(profile.history_vectors):
                # On macOS ARM, NumPy/Accelerate can emit spurious overflow
                # warnings for a finite float32 matrix-vector ``@`` product.
                # ``einsum`` is numerically equivalent here, avoids that
                # backend path, and keeps semantic values safely bounded.
                similarities = np.einsum(
                    "ij,j->i", profile.history_vectors, candidate_vector, optimize=False
                )
                semantic_max = float(np.clip(
                    np.nan_to_num(similarities, nan=0.0, posinf=1.0, neginf=-1.0).max(),
                    -1.0,
                    1.0,
                ))
    published = articles.published.get(article_id)
    freshness = 0.0
    if published is not None and now is not None:
        hours_old = max(0.0, (now - published).total_seconds() / 3600.0)
        freshness = math.exp(-hours_old / (24.0 * 4.0))
    category_total = sum(profile.category_weights.values())
    subcategory_total = sum(profile.subcategory_weights.values())
    topic_total = sum(profile.topic_weights.values())
    popularity_count = popularity.count(article_id, now, causal_popularity)
    age_value = number(age, -1.0)
    age_bucket = 0.0 if age_value < 0 else min(9.0, math.floor(age_value / 10.0)) / 9.0
    return {
        "bm25_score": number(bm25_score),
        "category_affinity": affinity((articles.category.get(article_id, ""),), profile.category_weights, category_total),
        "subcategory_affinity": affinity(articles.subcategories.get(article_id, ()), profile.subcategory_weights, subcategory_total),
        "topic_affinity": affinity(articles.topics.get(article_id, ()), profile.topic_weights, topic_total),
        "entity_overlap": entity_overlap,
        "entity_jaccard": entity_jaccard,
        "semantic_mean": semantic_mean,
        "semantic_max": semantic_max,
        "popularity": math.log1p(popularity_count) / math.log1p(max(popularity.maximum, 1)),
        "freshness": freshness,
        "history_length": float(len(profile.history)),
        "candidate_title_tokens": float(articles.title_tokens.get(article_id, 0.0)),
        "candidate_premium": articles.premium.get(article_id, 0.0),
        "candidate_sentiment": articles.sentiment.get(article_id, 0.0),
        "position_bias": 1.0 / math.log2(position + 2.0),
        "device_type": number(device_type),
        "is_sso_user": float(bool(is_sso_user)),
        "is_subscriber": float(bool(is_subscriber)),
        "age_bucket": age_bucket,
    }


def feature_rows_for_impression(
    impression_id: object,
    candidates_raw: object,
    clicked_raw: object,
    user_id: object,
    now: datetime | None,
    histories: dict[int, tuple[str, ...]],
    articles: ArticleStore,
    popularity: HourlyPopularity,
    causal_popularity: bool,
    embedding_positions: dict[str, int] | None,
    vectors: np.ndarray | None,
    decay: float,
    group_order: int,
    device_type: object = 0,
    is_sso_user: object = False,
    is_subscriber: object = False,
    age: object = None,
    cached_profile: UserProfile | None = None,
) -> tuple[list[dict[str, object]], UserProfile]:
    """Create feature records for one complete candidate group."""
    candidates = listify(candidates_raw)
    history = histories.get(int(user_id), ())
    profile = cached_profile or profile_from_history(history, articles, embedding_positions, vectors, decay)
    query = query_from_history(list(profile.history), articles.title, 8)
    bm25_scores = articles.bm25.candidate_scores(query, list(candidates)) if articles.bm25 else {}
    clicked = set(listify(clicked_raw))
    records: list[dict[str, object]] = []
    for position, article_id in enumerate(candidates):
        features = candidate_features(
            article_id, position, profile, now, articles, popularity, causal_popularity,
            embedding_positions, vectors, bm25_scores.get(article_id, 0.0), device_type,
            is_sso_user, is_subscriber, age,
        )
        records.append({
            "impression_id": str(impression_id),
            "group_order": group_order,
            "label": int(article_id in clicked),
            **features,
        })
    return records, profile


def feature_path(work: Path, split: str, modulus: int, maximum: int) -> Path:
    suffix = f"mod{modulus}" + (f"_first{maximum}" if maximum else "")
    return work / "features" / f"{split}_{suffix}.parquet"


def build_feature_cache(
    behaviour_path: Path,
    history_path: Path,
    split: str,
    modulus: int,
    maximum: int,
    work: Path,
    articles: ArticleStore,
    popularity: HourlyPopularity,
    positions: dict[str, int] | None,
    vectors: np.ndarray | None,
    history_size: int,
    decay: float,
    force: bool,
) -> Path:
    """Materialise only deterministic full groups into a compact Parquet cache."""
    target = feature_path(work, split, modulus, maximum)
    if target.exists() and not force:
        return target
    wanted_users = selected_user_ids(behaviour_path, modulus, maximum)
    histories = load_histories(history_path, wanted_users, history_size)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.partial")
    writer = pq.ParquetWriter(temporary, FEATURE_SCHEMA, compression="zstd")
    records: list[dict[str, object]] = []
    group_order = 0
    selected_groups = 0
    source_columns = [
        "impression_id", "impression_time", "article_ids_inview", "article_ids_clicked", "user_id",
        "device_type", "is_sso_user", "is_subscriber", "age",
    ]

    def flush() -> None:
        nonlocal records
        if records:
            writer.write_table(pa.Table.from_pylist(records, schema=FEATURE_SCHEMA))
            records = []

    try:
        for batch in tqdm(iter_behaviour_batches(behaviour_path, source_columns), desc=f"Building {split} v2 features", unit="batch"):
            row_count = len(batch["impression_id"])
            for index in range(row_count):
                impression_id = row_value(batch, "impression_id", index)
                if not stable_selected(impression_id, modulus):
                    continue
                if maximum and selected_groups >= maximum:
                    break
                candidates = row_value(batch, "article_ids_inview", index)
                clicked = row_value(batch, "article_ids_clicked", index, ())
                # LambdaRank gains nothing from an all-negative train group.
                if split == "train" and not listify(clicked):
                    continue
                rows, _profile = feature_rows_for_impression(
                    impression_id=impression_id,
                    candidates_raw=candidates,
                    clicked_raw=clicked,
                    user_id=row_value(batch, "user_id", index),
                    now=row_value(batch, "impression_time", index),
                    histories=histories,
                    articles=articles,
                    popularity=popularity,
                    causal_popularity=(split == "train"),
                    embedding_positions=positions,
                    vectors=vectors,
                    decay=decay,
                    group_order=group_order,
                    device_type=row_value(batch, "device_type", index),
                    is_sso_user=row_value(batch, "is_sso_user", index),
                    is_subscriber=row_value(batch, "is_subscriber", index),
                    age=row_value(batch, "age", index),
                )
                if not rows:
                    continue
                records.extend(rows)
                group_order += 1
                selected_groups += 1
                if len(records) >= 50_000:
                    flush()
            if maximum and selected_groups >= maximum:
                break
        flush()
    finally:
        writer.close()
    os.replace(temporary, target)
    print(f"{selected_groups:,} full {split} impressions -> {target}")
    return target


def read_feature_frame(path: Path, require_positive: bool = True) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing feature cache {path}. Run --mode prepare or --mode all first.")
    frame = pd.read_parquet(path)
    missing = set(FEATURE_COLUMNS + ["label", "group_order", "impression_id"]) - set(frame.columns)
    if missing:
        raise ValueError(f"Feature cache {path} is missing {sorted(missing)}")
    frame = frame.sort_values("group_order", kind="stable").reset_index(drop=True)
    if require_positive and not frame.empty:
        positive_groups = frame.groupby("group_order", sort=False).label.transform("sum") > 0
        frame = frame[positive_groups].reset_index(drop=True)
    return frame


def group_sizes(frame: pd.DataFrame) -> list[int]:
    if frame.empty:
        raise ValueError("No ranked impressions available. Increase the sample size or check the data paths.")
    return frame.groupby("group_order", sort=False).size().astype(int).tolist()


def model_paths(work: Path) -> tuple[Path, Path]:
    return work / "models" / "ebnerd_v2_lambdarank.txt", work / "models" / "ebnerd_v2_lambdarank.json"


def train_model(train_features: Path, validation_features: Path, work: Path, threads: int) -> Path:
    """Train a real LightGBM LambdaRank model, with temporal validation early-stop."""
    require_lightgbm()
    train = read_feature_frame(train_features)
    validation = read_feature_frame(validation_features)
    train_groups, validation_groups = group_sizes(train), group_sizes(validation)
    params = {
        "objective": "lambdarank",
        "metric": ["auc", "ndcg"],
        "ndcg_at": [5, 10],
        "learning_rate": 0.04,
        "num_leaves": 63,
        "min_data_in_leaf": 40,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l2": 2.0,
        "lambdarank_truncation_level": 20,
        "verbosity": -1,
        "seed": 42,
        "num_threads": threads,
    }
    train_set = lgb.Dataset(train[FEATURE_COLUMNS], label=train.label, group=train_groups, feature_name=FEATURE_COLUMNS)
    validation_set = lgb.Dataset(validation[FEATURE_COLUMNS], label=validation.label, group=validation_groups, reference=train_set, feature_name=FEATURE_COLUMNS)
    booster = lgb.train(
        params,
        train_set,
        num_boost_round=1_200,
        valid_sets=[validation_set],
        valid_names=["temporal_validation"],
        callbacks=[lgb.early_stopping(80, verbose=True), lgb.log_evaluation(50)],
    )
    model, metadata = model_paths(work)
    model.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(model), num_iteration=booster.best_iteration)
    metadata.write_text(json.dumps({
        "feature_columns": FEATURE_COLUMNS,
        "backend": "lightgbm_lambdarank",
        "best_iteration": booster.best_iteration,
        "train_rows": int(len(train)),
        "train_groups": len(train_groups),
        "validation_rows": int(len(validation)),
        "validation_groups": len(validation_groups),
        "parameters": params,
        "created_utc": datetime.utcnow().isoformat() + "Z",
    }, indent=2), encoding="utf-8")
    print(f"Saved LambdaRank model: {model}")
    return model


def metric_values(rank_list: Sequence[int], labels: Sequence[int]) -> tuple[float, float, float, float]:
    order = np.argsort(np.asarray(rank_list))
    positives = [position for position, label in enumerate(labels) if label]
    negatives = [position for position, label in enumerate(labels) if not label]
    if not positives or not negatives:
        auc = 0.0
    else:
        auc = sum(rank_list[positive] < rank_list[negative] for positive in positives for negative in negatives) / (len(positives) * len(negatives))
    mrr = next((1.0 / (index + 1) for index, position in enumerate(order) if labels[position]), 0.0)

    def ndcg(k: int) -> float:
        dcg = sum(1.0 / math.log2(index + 2) for index, position in enumerate(order[:k]) if labels[position])
        ideal = sum(1.0 / math.log2(index + 2) for index in range(min(k, sum(labels))))
        return dcg / ideal if ideal else 0.0

    return float(auc), mrr, ndcg(5), ndcg(10)


def evaluate_model(validation_features: Path, work: Path) -> dict[str, object]:
    """Compare learned ranker with the lexical feature on exactly the same groups."""
    require_lightgbm()
    model_path, metadata_path = model_paths(work)
    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("Model is missing. Run --mode train after --mode prepare.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("feature_columns") != FEATURE_COLUMNS:
        raise ValueError("Model feature schema differs from this script; rebuild v2 features and model.")
    frame = read_feature_frame(validation_features, require_positive=False)
    booster = lgb.Booster(model_file=str(model_path))
    model_scores = booster.predict(frame[FEATURE_COLUMNS], num_iteration=booster.best_iteration)
    totals = {"bm25_only": np.zeros(4), "v2_lambdarank": np.zeros(4)}
    evaluated = 0
    offset = 0
    for _group, group in frame.groupby("group_order", sort=False):
        length = len(group)
        labels = group.label.astype(int).tolist()
        # Codabench's behaviour metric is only meaningful where there is a
        # positive item and at least one non-clicked candidate.
        if not any(labels) or all(labels):
            offset += length
            continue
        totals["bm25_only"] += metric_values(ranks_from_scores(group.bm25_score), labels)
        totals["v2_lambdarank"] += metric_values(ranks_from_scores(model_scores[offset:offset + length]), labels)
        offset += length
        evaluated += 1
    names = ["auc", "mrr", "ndcg@5", "ndcg@10"]
    result = {
        "validation_impressions": evaluated,
        "feature_cache": str(validation_features),
        "model": str(model_path),
        "bm25_only": dict(zip(names, (totals["bm25_only"] / max(evaluated, 1)).round(6).tolist())),
        "v2_lambdarank": dict(zip(names, (totals["v2_lambdarank"] / max(evaluated, 1)).round(6).tolist())),
        "delta_v2_minus_bm25": dict(zip(names, ((totals["v2_lambdarank"] - totals["bm25_only"]) / max(evaluated, 1)).round(6).tolist())),
    }
    output = work / "validation_metrics.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def test_fingerprint(test_path: Path, model_path: Path, batch_size: int, limit: int) -> str:
    payload = {
        "version": 2,
        "test": {"path": str(test_path.resolve()), "size": test_path.stat().st_size, "mtime_ns": test_path.stat().st_mtime_ns},
        "model": {"path": str(model_path.resolve()), "size": model_path.stat().st_size, "mtime_ns": model_path.stat().st_mtime_ns},
        "features": FEATURE_COLUMNS,
        "batch_size": batch_size,
        "limit": limit,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def valid_prediction_line(expected_id: str, candidates: Sequence[str], line: str) -> bool:
    try:
        identifier, ranks_text = line.rstrip("\n").split(" ", 1)
        ranks = [int(value) for value in ranks_text[1:-1].split(",")] if ranks_text != "[]" else []
        return identifier == expected_id and ranks_text.startswith("[") and ranks_text.endswith("]") and sorted(ranks) == list(range(1, len(candidates) + 1))
    except (ValueError, IndexError):
        return False


def validate_prediction_file(test_path: Path, prediction: Path, limit: int = 0) -> int:
    """Strictly verify IDs, order, rank permutations, and row count before ZIP."""
    checked = 0
    with prediction.open("r", encoding="utf-8") as source:
        for batch in tqdm(iter_behaviour_batches(test_path, ["impression_id", "impression_time", "article_ids_inview", "user_id"]), desc="Validating EB-NeRD v2 prediction", unit="batch"):
            for impression_id, candidates in zip(batch["impression_id"], batch["article_ids_inview"]):
                if limit and checked >= limit:
                    break
                line = source.readline()
                candidate_ids = listify(candidates)
                if not line or not valid_prediction_line(str(impression_id), candidate_ids, line):
                    raise ValueError(f"Invalid prediction at source row {checked + 1}, impression {impression_id}")
                checked += 1
            if limit and checked >= limit:
                break
        if source.readline():
            raise ValueError("Prediction file has more rows than the requested test prefix")
    if limit and checked != limit:
        raise ValueError(f"Expected {limit} prediction rows, found {checked}")
    return checked


def make_submission(
    test_path: Path,
    test_history_path: Path,
    articles: ArticleStore,
    popularity: HourlyPopularity,
    positions: dict[str, int] | None,
    vectors: np.ndarray | None,
    work: Path,
    output_dir: Path,
    batch_size: int,
    context_cache_size: int,
    history_size: int,
    decay: float,
    limit: int,
    resume: bool,
) -> Path:
    """Stream model scores for test impressions and package a root-only ZIP."""
    require_lightgbm()
    model_path, metadata_path = model_paths(work)
    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("Model is missing. Do not submit until --mode train and --mode validate succeed.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("feature_columns") != FEATURE_COLUMNS:
        raise ValueError("Model feature schema differs from this script; rebuild v2 model.")
    booster = lgb.Booster(model_file=str(model_path))
    histories = load_histories(test_history_path, None, history_size)
    print(f"Loaded compact test histories for {len(histories):,} users")

    destination = output_dir / ("smoke" if limit else "final")
    destination.mkdir(parents=True, exist_ok=True)
    partial = destination / "predictions.txt.partial"
    checkpoint = destination / "predictions.resume.json"
    prediction = destination / "predictions.txt"
    fingerprint = test_fingerprint(test_path, model_path, batch_size, limit)
    completed = 0
    offset = 0
    if resume:
        if not partial.exists() or not checkpoint.exists():
            raise FileNotFoundError("--resume requires predictions.txt.partial and predictions.resume.json")
        state = json.loads(checkpoint.read_text(encoding="utf-8"))
        if state.get("fingerprint") != fingerprint:
            raise ValueError("Resume state does not match this test/model/configuration")
        completed, offset = int(state["completed"]), int(state["byte_offset"])
        with partial.open("r+b") as destination_file:
            destination_file.seek(0, os.SEEK_END)
            if destination_file.tell() < offset:
                raise ValueError("Partial prediction is shorter than its checkpoint")
            destination_file.truncate(offset)
        print(f"Resuming after {completed:,} checked rows")
    elif partial.exists() or checkpoint.exists():
        raise FileExistsError(f"A partial run exists at {destination}. Re-run with --resume or use a new --out directory.")
    elif prediction.exists():
        raise FileExistsError(
            f"{prediction} already exists. Keep it as an artifact and choose a new --out directory for another v2 run."
        )

    profile_cache: OrderedDict[int, UserProfile] = OrderedDict()

    def cached_profile(user_id: int) -> UserProfile:
        profile = profile_cache.get(user_id)
        if profile is not None:
            profile_cache.move_to_end(user_id)
            return profile
        profile = profile_from_history(histories.get(user_id, ()), articles, positions, vectors, decay)
        if context_cache_size:
            profile_cache[user_id] = profile
            if len(profile_cache) > context_cache_size:
                profile_cache.popitem(last=False)
        return profile

    meta: list[tuple[str, tuple[str, ...]]] = []
    feature_rows: list[dict[str, object]] = []
    skipped = 0
    source_columns = [
        "impression_id", "impression_time", "article_ids_inview", "user_id", "device_type",
        "is_sso_user", "is_subscriber", "age",
    ]
    mode = "ab" if completed else "wb"
    written = completed
    with partial.open(mode) as destination_file:
        def flush() -> None:
            nonlocal meta, feature_rows, written
            if not meta:
                return
            frame = pd.DataFrame(feature_rows)
            scores = booster.predict(frame[FEATURE_COLUMNS], num_iteration=booster.best_iteration)
            lines: list[str] = []
            start = 0
            for impression_id, candidates in meta:
                count = len(candidates)
                ranks = ranks_from_scores(scores[start:start + count])
                start += count
                lines.append(f"{impression_id} [{','.join(map(str, ranks))}]\n")
            destination_file.write("".join(lines).encode("utf-8"))
            destination_file.flush()
            os.fsync(destination_file.fileno())
            written += len(meta)
            temporary = checkpoint.with_suffix(".tmp")
            temporary.write_text(json.dumps({
                "fingerprint": fingerprint,
                "completed": written,
                "byte_offset": destination_file.tell(),
                "last_impression_id": meta[-1][0],
            }, indent=2), encoding="utf-8")
            os.replace(temporary, checkpoint)
            print(f"  checkpointed {written:,} predictions")
            meta, feature_rows = [], []

        for batch in tqdm(iter_behaviour_batches(test_path, source_columns), desc="Scoring EB-NeRD v2 test", unit="batch"):
            row_count = len(batch["impression_id"])
            for index in range(row_count):
                if skipped < completed:
                    skipped += 1
                    continue
                if limit and written + len(meta) >= limit:
                    break
                impression_id = row_value(batch, "impression_id", index)
                candidates = listify(row_value(batch, "article_ids_inview", index))
                user_id = int(row_value(batch, "user_id", index))
                rows, _profile = feature_rows_for_impression(
                    impression_id=impression_id,
                    candidates_raw=candidates,
                    clicked_raw=(),
                    user_id=user_id,
                    now=row_value(batch, "impression_time", index),
                    histories=histories,
                    articles=articles,
                    popularity=popularity,
                    causal_popularity=False,
                    embedding_positions=positions,
                    vectors=vectors,
                    decay=decay,
                    group_order=0,
                    device_type=row_value(batch, "device_type", index),
                    is_sso_user=row_value(batch, "is_sso_user", index),
                    is_subscriber=row_value(batch, "is_subscriber", index),
                    age=row_value(batch, "age", index),
                    cached_profile=cached_profile(user_id),
                )
                feature_rows.extend(rows)
                meta.append((str(impression_id), candidates))
                if len(meta) >= batch_size:
                    flush()
            if limit and written + len(meta) >= limit:
                break
        if skipped != completed:
            raise ValueError("Resume checkpoint exceeds the number of test impressions")
        flush()
    if limit and written != limit:
        raise ValueError(f"Smoke limit was {limit}, but {written} rows were written")
    os.replace(partial, prediction)
    checked = validate_prediction_file(test_path, prediction, limit=limit)
    bundle = destination / ("ebnerd_v2_smoke.zip" if limit else "ebnerd_v2_submission.zip")
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        archive.write(prediction, arcname="predictions.txt")
    checkpoint.unlink(missing_ok=True)
    print(f"Ready: {bundle} ({checked:,} validated rows; ZIP contains only predictions.txt)")
    if limit:
        print("This is a smoke-test ZIP. Re-run without --limit before Codabench upload.")
    return bundle


@dataclass
class Paths:
    articles: Path
    train_behaviours: Path
    train_history: Path
    validation_behaviours: Path
    validation_history: Path
    test_behaviours: Path | None = None
    test_history: Path | None = None


def prepare_large_paths(large: Path, testzip: Path | None, work: Path, include_test: bool) -> Paths:
    raw = work / "raw"
    paths = Paths(
        articles=extract_member(large, "articles.parquet", raw / "articles.parquet"),
        train_behaviours=extract_member(large, "train/behaviors.parquet", raw / "train_behaviors.parquet"),
        train_history=extract_member(large, "train/history.parquet", raw / "train_history.parquet"),
        validation_behaviours=extract_member(large, "validation/behaviors.parquet", raw / "validation_behaviors.parquet"),
        validation_history=extract_member(large, "validation/history.parquet", raw / "validation_history.parquet"),
    )
    if include_test:
        if testzip is None:
            raise ValueError("--testzip is required for --mode submit")
        paths.test_behaviours = extract_member(testzip, find_member(testzip, "/test/behaviors.parquet"), raw / "test_behaviors.parquet")
        paths.test_history = extract_member(testzip, find_member(testzip, "/test/history.parquet"), raw / "test_history.parquet")
    return paths


def run_prepare(args: argparse.Namespace, paths: Paths, articles: ArticleStore, popularity: HourlyPopularity, positions: dict[str, int] | None, vectors: np.ndarray | None) -> tuple[Path, Path]:
    train = build_feature_cache(
        paths.train_behaviours, paths.train_history, "train", args.train_sample_mod,
        args.train_max_impressions, args.work, articles, popularity, positions, vectors,
        args.history_size, args.decay, args.force,
    )
    validation = build_feature_cache(
        paths.validation_behaviours, paths.validation_history, "validation", args.validation_sample_mod,
        args.validation_max_impressions, args.work, articles, popularity, positions, vectors,
        args.history_size, args.decay, args.force,
    )
    return train, validation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["prepare", "train", "validate", "all", "submit"], default="all")
    parser.add_argument("--large", type=Path, default=default_source("ebnerd_large.zip"))
    parser.add_argument("--testzip", type=Path, default=default_source("ebnerd_testset.zip"))
    parser.add_argument("--embeddings", type=Path, default=default_source("Ekstra_Bladet_word2vec.zip"))
    parser.add_argument("--work", type=Path, default=Path("data/v2/ebnerd_large"), help="private v2 cache; does not touch baseline data/raw")
    parser.add_argument("--out", type=Path, default=Path("outputs/ebnerd_v2_submission"))
    parser.add_argument("--train-sample-mod", type=int, default=80, help="keep 1/N complete train groups; lower N = larger/better model")
    parser.add_argument("--validation-sample-mod", type=int, default=20, help="keep 1/N validation groups for early stopping/reporting")
    parser.add_argument("--train-max-impressions", type=int, default=0, help="cap sampled train groups only for a quick run")
    parser.add_argument("--validation-max-impressions", type=int, default=0, help="cap sampled validation groups only for a quick run")
    parser.add_argument("--history-size", type=int, default=40)
    parser.add_argument("--decay", type=float, default=0.85, help="per-click recency decay in (0, 1]")
    parser.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--prediction-batch-size", type=int, default=3_000)
    parser.add_argument("--context-cache-size", type=int, default=10_000, help="bounded test user-profile LRU cache")
    parser.add_argument("--limit", type=int, default=0, help="test-prefix smoke limit; 0 means the complete test set")
    parser.add_argument("--no-bm25", action="store_true", help="disable lexical feature to reduce RAM (not recommended for final model)")
    parser.add_argument("--resume", action="store_true", help="resume an interrupted submit run with matching model/config")
    parser.add_argument("--force", action="store_true", help="rebuild exactly this v2 feature/popularity cache")
    args = parser.parse_args()
    if args.train_sample_mod < 1 or args.validation_sample_mod < 1:
        parser.error("sample moduli must be >= 1")
    if args.history_size < 1 or not 0 < args.decay <= 1:
        parser.error("--history-size must be positive and --decay must be in (0, 1]")
    if args.mode in {"train", "validate", "all", "submit"}:
        require_lightgbm()
    include_test = args.mode == "submit"
    paths = prepare_large_paths(args.large, args.testzip if include_test else None, args.work, include_test)
    print("Loading v2 article metadata" + (" + BM25" if not args.no_bm25 else " (without BM25)"))
    articles = load_articles(paths.articles, use_bm25=not args.no_bm25)
    print(f"  {len(articles.title):,} articles")
    positions, vectors = load_embeddings(args.embeddings, args.work, set(articles.title))
    print(f"  semantic vectors: {0 if positions is None else len(positions):,}")
    popularity = build_hourly_popularity(paths.train_behaviours, args.work, force=args.force)
    print(f"  train popularity: {len(popularity.totals):,} clicked articles")

    train_features = feature_path(args.work, "train", args.train_sample_mod, args.train_max_impressions)
    validation_features = feature_path(args.work, "validation", args.validation_sample_mod, args.validation_max_impressions)
    if args.mode in {"prepare", "all"}:
        train_features, validation_features = run_prepare(args, paths, articles, popularity, positions, vectors)
    if args.mode in {"train", "all"}:
        train_model(train_features, validation_features, args.work, args.threads)
    if args.mode in {"validate", "all"}:
        evaluate_model(validation_features, args.work)
    if args.mode == "submit":
        assert paths.test_behaviours is not None and paths.test_history is not None
        make_submission(
            paths.test_behaviours, paths.test_history, articles, popularity, positions, vectors,
            args.work, args.out, args.prediction_batch_size, args.context_cache_size,
            args.history_size, args.decay, args.limit, args.resume,
        )


if __name__ == "__main__":
    main()
