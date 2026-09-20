#!/usr/bin/env python3
"""Large-data MIND v2 ranker.

This is deliberately separate from ``make_mind_submission.py``.  The original
script is retained as the Assignment-2 reproducibility baseline; this module
trains a stronger model from MINDlarge's labelled interactions and writes a
second, independently resumable submission.

The design avoids the two common large-MIND mistakes:

* it never materialises all ~83M candidate rows; training uses a deterministic
  group-preserving sample and bounded negatives per impression;
* validation is the official future MINDlarge_dev split, not a random sample.

Typical workflow (from the A2 directory):

  python mind_large_v2.py train --train /.../MINDlarge_train.zip \
      --dev /.../MINDlarge_dev.zip
  python mind_large_v2.py evaluate --dev /.../MINDlarge_dev.zip
  python mind_large_v2.py submit --test /.../MINDlarge_test.zip

Only run ``submit`` after its ``evaluate`` result is better than the previous
submission.  The resulting ZIP contains exactly one root-level prediction.txt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import re
import sys
import zipfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
from tqdm.auto import tqdm

try:
    from lightgbm import LGBMRanker
except (ImportError, OSError):
    LGBMRanker = None
from sklearn.ensemble import HistGradientBoostingClassifier


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE_ROOT = ROOT.parent.parent / "Assighnment1"
DEFAULT_TRAIN = DEFAULT_SOURCE_ROOT / "MIND_data" / "MINDlarge_train.zip"
DEFAULT_DEV = DEFAULT_SOURCE_ROOT / "MIND_data" / "MINDlarge_dev.zip"
DEFAULT_TEST = DEFAULT_SOURCE_ROOT / "MIND_data" / "MINDlarge_test.zip"
DEFAULT_CACHE = Path("data/v2/mind_large")
DEFAULT_OUTPUT = Path("outputs/mind_v2_submission")

FEATURE_NAMES = [
    "category_affinity",
    "subcategory_affinity",
    "same_category_as_last",
    "same_subcategory_as_last",
    "semantic_mean",
    "semantic_max_recent",
    "semantic_last",
    "entity_jaccard",
    "entity_overlap_rate",
    "title_overlap",
    "candidate_popularity",
    "position_bias",
    "history_length_log",
    "history_category_diversity",
    "history_entity_count_log",
    "candidate_has_vector",
    "idf_token_overlap",
    "max_recent_idf_overlap",
]
TOKEN_RE = re.compile(r"[a-z0-9]+")
CHECKPOINT_VERSION = 1


def archive_member(archive: Path, suffix: str) -> str:
    """Return the one ZIP member ending in suffix, with a useful error."""
    with zipfile.ZipFile(archive) as bundle:
        matches = [name for name in bundle.namelist() if name.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"Expected one {suffix!r} in {archive}, found {matches!r}")
    return matches[0]


def parse_timestamp(value: str) -> datetime:
    # MIND stores e.g. "11/10/2019 11:30:54 AM".  This is faster and less
    # ambiguous than pandas' generic parser when called millions of times.
    return datetime.strptime(value, "%m/%d/%Y %I:%M:%S %p")


def date_key(value: str) -> str:
    return parse_timestamp(value).date().isoformat()


def tokens(value: str) -> frozenset[str]:
    return frozenset(TOKEN_RE.findall((value or "").lower()))


def entity_ids(value: str) -> frozenset[str]:
    try:
        objects = json.loads(value) if value else []
    except (json.JSONDecodeError, TypeError):
        return frozenset()
    return frozenset(
        item.get("WikidataId") for item in objects
        if isinstance(item, dict) and item.get("WikidataId")
    )


def stable_bucket(value: str, modulus: int) -> int:
    if modulus < 1:
        raise ValueError("sample modulus must be at least 1")
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % modulus


def parse_candidates(value: str, labelled: bool) -> tuple[list[str], list[int]]:
    articles: list[str] = []
    labels: list[int] = []
    for item in value.split():
        if labelled:
            article, separator, raw_label = item.rpartition("-")
            if not separator or raw_label not in {"0", "1"}:
                raise ValueError(f"Malformed labelled MIND candidate {item!r}")
            articles.append(article)
            labels.append(int(raw_label))
        else:
            articles.append(item)
    return articles, labels


def parse_behavior(line: bytes, labelled: bool) -> tuple[str, str, str, list[str], list[str], list[int]]:
    fields = line.decode("utf-8").rstrip("\n").split("\t")
    if len(fields) != 5:
        raise ValueError(f"Expected five behavior columns, got {len(fields)}")
    impression_id, user_id, timestamp, raw_history, raw_candidates = fields
    candidates, labels = parse_candidates(raw_candidates, labelled=labelled)
    return impression_id, user_id, timestamp, raw_history.split() if raw_history else [], candidates, labels


@dataclass
class Catalog:
    """Compact article metadata needed at training and serving time."""

    category: dict[str, str]
    subcategory: dict[str, str]
    entities: dict[str, frozenset[str]]
    title_tokens: dict[str, frozenset[str]]
    vectors: dict[str, np.ndarray]
    token_idf: dict[str, float]

    def get_vector(self, article: str) -> np.ndarray | None:
        return self.vectors.get(article)


def catalog_fingerprint(archives: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in archives:
        stat = path.stat()
        digest.update(str(path.resolve()).encode())
        digest.update(str(stat.st_size).encode())
        digest.update(str(stat.st_mtime_ns).encode())
    return digest.hexdigest()


def load_entity_vectors(train_archive: Path) -> dict[str, np.ndarray]:
    vectors: dict[str, np.ndarray] = {}
    entity_member = archive_member(train_archive, "/entity_embedding.vec")
    with zipfile.ZipFile(train_archive) as bundle, bundle.open(entity_member) as raw:
        for line in tqdm(raw, desc="Load MIND entity vectors", unit="vector"):
            fields = line.decode("utf-8").rstrip("\n").split("\t")
            if len(fields) < 2:
                continue
            # Official MIND vectors have a trailing tab, producing one empty
            # final field.  Filter it instead of treating the archive as bad.
            vector = np.asarray([value for value in fields[1:] if value], dtype=np.float32)
            norm = float(np.linalg.norm(vector))
            if norm > 1e-12:
                vectors[fields[0]] = vector / norm
    return vectors


def build_catalog(archives: Sequence[Path], cache_dir: Path, force: bool = False) -> Catalog:
    """Load a union news catalogue and cache it safely for repeated commands."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload_path = cache_dir / "catalog.pkl"
    metadata_path = cache_dir / "catalog.meta.json"
    fingerprint = catalog_fingerprint(archives)
    if not force and payload_path.exists() and metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text())
            if metadata.get("fingerprint") == fingerprint:
                with payload_path.open("rb") as handle:
                    cached = pickle.load(handle)
                # v2.1 added IDF-weighted lexical features.  Do not silently
                # reuse a catalogue with a different feature definition.
                if hasattr(cached, "token_idf"):
                    return cached
        except (OSError, ValueError, pickle.UnpicklingError):
            pass

    entity_vectors = load_entity_vectors(archives[0])
    category: dict[str, str] = {}
    subcategory: dict[str, str] = {}
    entities: dict[str, frozenset[str]] = {}
    title_tokens: dict[str, frozenset[str]] = {}
    vectors: dict[str, np.ndarray] = {}
    document_frequency: Counter = Counter()
    seen: set[str] = set()
    for archive in archives:
        news_member = archive_member(archive, "/news.tsv")
        with zipfile.ZipFile(archive) as bundle, bundle.open(news_member) as raw:
            for line in tqdm(raw, desc=f"Read {archive.stem} news", unit="article"):
                fields = line.decode("utf-8").rstrip("\n").split("\t")
                if len(fields) < 8 or fields[0] in seen:
                    continue
                article, cat, subcat, title, abstract = fields[:5]
                article_entities = entity_ids(fields[6]) | entity_ids(fields[7])
                category[article] = cat or ""
                subcategory[article] = subcat or ""
                entities[article] = article_entities
                title_tokens[article] = tokens(f"{title} {abstract}")
                document_frequency.update(title_tokens[article])
                available = [entity_vectors[item] for item in article_entities if item in entity_vectors]
                if available:
                    vector = np.mean(np.asarray(available, dtype=np.float32), axis=0)
                    norm = float(np.linalg.norm(vector))
                    if norm > 1e-12:
                        vectors[article] = (vector / norm).astype(np.float32)
                seen.add(article)
    document_count = max(len(category), 1)
    token_idf = {
        token: float(math.log((document_count + 1) / (frequency + 1)) + 1.0)
        for token, frequency in document_frequency.items()
    }
    catalog = Catalog(category, subcategory, entities, title_tokens, vectors, token_idf)
    temporary = payload_path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(catalog, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, payload_path)
    metadata_path.write_text(json.dumps({"fingerprint": fingerprint, "articles": len(category), "schema": 2}, indent=2))
    print(f"Cached MIND v2 catalogue: {len(category):,} articles; {len(vectors):,} entity-vector articles")
    return catalog


def daily_clicks(archive: Path) -> tuple[dict[str, Counter], Counter]:
    """Count labelled clicks by calendar day, never relying on ZIP row order."""
    by_day: dict[str, Counter] = {}
    behaviors_member = archive_member(archive, "/behaviors.tsv")
    with zipfile.ZipFile(archive) as bundle, bundle.open(behaviors_member) as raw:
        for line in tqdm(raw, desc=f"Count clicks {archive.stem}", unit="impression"):
            _id, _user, timestamp, _history, candidates, labels = parse_behavior(line, labelled=True)
            clicked = [article for article, label in zip(candidates, labels) if label]
            if clicked:
                day = date_key(timestamp)
                by_day.setdefault(day, Counter()).update(clicked)
    total: Counter = Counter()
    for counter in by_day.values():
        total.update(counter)
    return by_day, total


def day_prefixes(by_day: dict[str, Counter], initial: Counter | None = None) -> dict[str, Counter]:
    """Return click counts strictly before each day (a leakage-safe snapshot)."""
    running = Counter(initial or {})
    result: dict[str, Counter] = {}
    for day in sorted(by_day):
        result[day] = running.copy()
        running.update(by_day[day])
    return result


def recency_weights(history: Sequence[str], decay: float = 0.85, keep: int = 40) -> tuple[list[str], np.ndarray]:
    kept = list(history[-keep:])
    # Last item has age zero and weight 1.0.
    weights = np.asarray([decay ** (len(kept) - 1 - index) for index in range(len(kept))], dtype=np.float32)
    return kept, weights


@dataclass
class UserContext:
    category_weights: Counter
    subcategory_weights: Counter
    category_total: float
    subcategory_total: float
    history_entities: frozenset[str]
    history_tokens: frozenset[str]
    recent_token_sets: tuple[frozenset[str], ...]
    user_vector: np.ndarray | None
    recent_vectors: tuple[np.ndarray, ...]
    last_category: str
    last_subcategory: str
    history_length: int
    category_diversity: float


def make_context(history: Sequence[str], catalog: Catalog) -> UserContext:
    articles, weights = recency_weights(history)
    category_weights: Counter = Counter()
    subcategory_weights: Counter = Counter()
    entity_union: set[str] = set()
    token_union: set[str] = set()
    vector_sum: np.ndarray | None = None
    vector_weight = 0.0
    recent_vectors: list[np.ndarray] = []
    for article, weight in zip(articles, weights):
        category_weights[catalog.category.get(article, "")] += float(weight)
        subcategory_weights[catalog.subcategory.get(article, "")] += float(weight)
        entity_union.update(catalog.entities.get(article, ()))
        token_union.update(catalog.title_tokens.get(article, ()))
        vector = catalog.get_vector(article)
        if vector is not None:
            if vector_sum is None:
                vector_sum = np.zeros_like(vector)
            vector_sum += float(weight) * vector
            vector_weight += float(weight)
    for article in articles[-5:]:
        vector = catalog.get_vector(article)
        if vector is not None:
            recent_vectors.append(vector)
    recent_token_sets = tuple(catalog.title_tokens.get(article, frozenset()) for article in articles[-5:])
    user_vector = None
    if vector_sum is not None and vector_weight:
        norm = float(np.linalg.norm(vector_sum))
        if norm > 1e-12:
            user_vector = vector_sum / norm
    last = articles[-1] if articles else ""
    return UserContext(
        category_weights=category_weights,
        subcategory_weights=subcategory_weights,
        category_total=max(float(sum(category_weights.values())), 1e-12),
        subcategory_total=max(float(sum(subcategory_weights.values())), 1e-12),
        history_entities=frozenset(entity_union),
        history_tokens=frozenset(token_union),
        recent_token_sets=recent_token_sets,
        user_vector=user_vector,
        recent_vectors=tuple(recent_vectors),
        last_category=catalog.category.get(last, ""),
        last_subcategory=catalog.subcategory.get(last, ""),
        history_length=len(articles),
        category_diversity=float(len([key for key in category_weights if key])),
    )


def feature_matrix(
    candidates: Sequence[str],
    context: UserContext,
    catalog: Catalog,
    popularity: Counter,
    popularity_scale: float,
) -> np.ndarray:
    """Construct a small float32 candidate matrix; no pandas row objects."""
    matrix = np.zeros((len(candidates), len(FEATURE_NAMES)), dtype=np.float32)
    normalizer = max(math.log1p(max(popularity_scale, 1.0)), 1e-6)
    hist_entities = context.history_entities
    hist_tokens = context.history_tokens
    for position, article in enumerate(candidates):
        cat = catalog.category.get(article, "")
        subcat = catalog.subcategory.get(article, "")
        article_entities = catalog.entities.get(article, frozenset())
        article_tokens = catalog.title_tokens.get(article, frozenset())
        row = matrix[position]
        row[0] = context.category_weights[cat] / context.category_total
        row[1] = context.subcategory_weights[subcat] / context.subcategory_total
        row[2] = float(bool(cat) and cat == context.last_category)
        row[3] = float(bool(subcat) and subcat == context.last_subcategory)
        vector = catalog.get_vector(article)
        if vector is not None:
            row[15] = 1.0
            if context.user_vector is not None:
                row[4] = float(vector @ context.user_vector)
            if context.recent_vectors:
                similarities = [float(vector @ prior) for prior in context.recent_vectors]
                row[5] = max(similarities)
                row[6] = similarities[-1]
        if article_entities and hist_entities:
            overlap = len(article_entities & hist_entities)
            row[7] = overlap / max(len(article_entities | hist_entities), 1)
            row[8] = overlap / max(len(article_entities), 1)
        if article_tokens and hist_tokens:
            row[9] = len(article_tokens & hist_tokens) / max(len(article_tokens), 1)
            candidate_idf = sum(catalog.token_idf.get(token, 0.0) for token in article_tokens)
            if candidate_idf:
                overlap_idf = sum(catalog.token_idf.get(token, 0.0) for token in (article_tokens & hist_tokens))
                row[16] = overlap_idf / candidate_idf
                if context.recent_token_sets:
                    row[17] = max(
                        sum(catalog.token_idf.get(token, 0.0) for token in (article_tokens & previous)) / candidate_idf
                        for previous in context.recent_token_sets
                    )
        row[10] = math.log1p(popularity.get(article, 0)) / normalizer
        row[11] = 1.0 / math.log2(position + 2.0)
        row[12] = math.log1p(context.history_length) / math.log(41.0)
        row[13] = context.category_diversity / 20.0
        row[14] = math.log1p(len(hist_entities)) / math.log(101.0)
    return matrix


def selected_indices(
    impression_id: str,
    labels: Sequence[int],
    negatives_per_positive: int,
    seed: int,
) -> list[int]:
    positives = [index for index, label in enumerate(labels) if label]
    negatives = [index for index, label in enumerate(labels) if not label]
    if not positives or not negatives:
        return []
    limit = max(negatives_per_positive * len(positives), 1)
    # Stable hash sampling means a resumable/repeated build is identical.
    negatives.sort(key=lambda index: stable_bucket(f"{seed}:{impression_id}:{index}", 2**31 - 1))
    return sorted(positives + negatives[:limit])


def save_training_matrix(path: Path, x: np.ndarray, y: np.ndarray, groups: np.ndarray, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, x=x.astype(np.float32), y=y.astype(np.int8), groups=groups.astype(np.int32))
    os.replace(temporary, path)
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2))


def load_training_matrix(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    if not path.exists():
        raise FileNotFoundError(f"Training matrix not found: {path}. Run the train command first.")
    saved = np.load(path)
    metadata_path = path.with_suffix(".json")
    return saved["x"], saved["y"], saved["groups"], json.loads(metadata_path.read_text())


def build_training_matrix(
    archives: Sequence[Path],
    catalog: Catalog,
    cache_path: Path,
    sample_modulus: int,
    negatives_per_positive: int,
    seed: int,
    max_impressions: int,
    base_popularity: Counter | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a deterministic grouped sample without materialising all events."""
    rows: list[np.ndarray] = []
    labels_out: list[np.ndarray] = []
    groups: list[int] = []
    base = Counter(base_popularity or {})
    processed = selected = 0
    for archive in archives:
        per_day, _total = daily_clicks(archive)
        prefixes = day_prefixes(per_day, initial=base)
        # max is a scale only.  Candidate-specific counts are strictly from
        # earlier days; this avoids using a future click as a feature value.
        scale = max((max(values.values(), default=0) for values in prefixes.values()), default=1)
        behaviors_member = archive_member(archive, "/behaviors.tsv")
        with zipfile.ZipFile(archive) as bundle, bundle.open(behaviors_member) as raw:
            for line in tqdm(raw, desc=f"Sample {archive.stem}", unit="impression"):
                impression_id, _user, timestamp, history, candidates, labels = parse_behavior(line, labelled=True)
                processed += 1
                if stable_bucket(f"{seed}:{impression_id}", sample_modulus) != 0:
                    continue
                indices = selected_indices(impression_id, labels, negatives_per_positive, seed)
                if not indices:
                    continue
                context = make_context(history, catalog)
                selected_candidates = [candidates[index] for index in indices]
                matrix = feature_matrix(selected_candidates, context, catalog, prefixes.get(date_key(timestamp), base), scale)
                rows.append(matrix)
                labels_out.append(np.asarray([labels[index] for index in indices], dtype=np.int8))
                groups.append(len(indices))
                selected += 1
                if max_impressions and selected >= max_impressions:
                    break
        base.update(_total)
        if max_impressions and selected >= max_impressions:
            break
    if not rows:
        raise RuntimeError("No training impressions selected; lower --sample-modulus or raise --max-impressions.")
    x = np.vstack(rows).astype(np.float32)
    y = np.concatenate(labels_out).astype(np.int8)
    group_array = np.asarray(groups, dtype=np.int32)
    metadata = {
        "feature_names": FEATURE_NAMES,
        "sample_modulus": sample_modulus,
        "negatives_per_positive": negatives_per_positive,
        "seed": seed,
        "source_archives": [str(path.resolve()) for path in archives],
        "source_impressions_seen": processed,
        "selected_impressions": int(len(groups)),
        "candidate_rows": int(len(y)),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    save_training_matrix(cache_path, x, y, group_array, metadata)
    print(json.dumps(metadata, indent=2))
    return x, y, group_array


def train_ranker(x: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int):
    if LGBMRanker is None:
        print("WARNING: LightGBM is unavailable; using a portable classifier fallback. Install libomp + lightgbm for LambdaRank.", file=sys.stderr)
        model = HistGradientBoostingClassifier(
            max_iter=350, max_leaf_nodes=63, learning_rate=0.05,
            l2_regularization=1.0, random_state=seed,
        )
        model.fit(x, y)
        return model, "sklearn_hist_gradient_boosting"
    model = LGBMRanker(
        objective="lambdarank", metric="ndcg", eval_at=[5, 10],
        n_estimators=700, learning_rate=0.035, num_leaves=63,
        min_child_samples=150, subsample=0.85, colsample_bytree=0.85,
        reg_lambda=1.0, random_state=seed, n_jobs=-1, verbosity=-1,
    )
    model.fit(x, y, group=groups.tolist(), feature_name=FEATURE_NAMES)
    return model, "lightgbm_lambdarank"


def select_model_features(x: np.ndarray, model_feature_names: Sequence[str] | None) -> np.ndarray:
    """Project the current feature matrix for an older compatible v2 model."""
    if model_feature_names is None or list(model_feature_names) == FEATURE_NAMES:
        return x
    unknown = [name for name in model_feature_names if name not in FEATURE_NAMES]
    if unknown:
        raise ValueError(f"Model refers to unsupported feature(s): {unknown}")
    indices = [FEATURE_NAMES.index(name) for name in model_feature_names]
    return x[:, indices]


def model_scores(model, x: np.ndarray, model_feature_names: Sequence[str] | None = None) -> np.ndarray:
    x = select_model_features(x, model_feature_names)
    if isinstance(model, HistGradientBoostingClassifier):
        return model.predict_proba(x)[:, 1]
    # Avoid sklearn's ranker wrapper here.  It emits a warning once per
    # impression when a NumPy batch has no pandas feature names, and it also
    # re-parses ``eval_at``.  The fitted Booster is exactly the same model and
    # gives a quiet vectorised prediction path for both evaluation and serving.
    booster = getattr(model, "booster_", None)
    if booster is not None:
        return np.asarray(booster.predict(x), dtype=np.float64)
    return np.asarray(model.predict(x), dtype=np.float64)


def save_model(path: Path, model, backend: str, train_matrix: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "backend": backend,
        "feature_names": FEATURE_NAMES,
        "train_matrix": str(train_matrix),
        "version": 2,
    }
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def load_model(path: Path):
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    names = payload.get("feature_names")
    if not isinstance(names, list) or any(name not in FEATURE_NAMES for name in names):
        raise ValueError("Model feature schema is not compatible with this mind_large_v2.py version. Retrain it.")
    return payload["model"], payload


def ranks_from_scores(scores: np.ndarray) -> np.ndarray:
    # mergesort makes ties deterministic and respects source candidate order.
    order = np.argsort(-scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.int32)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.int32)
    return ranks


def per_impression_metrics(scores: np.ndarray, labels: Sequence[int]) -> tuple[float, float, float, float]:
    labels_array = np.asarray(labels, dtype=np.int8)
    positives = int(labels_array.sum())
    negatives = len(labels_array) - positives
    if positives == 0 or negatives == 0:
        return 0.0, 0.0, 0.0, 0.0
    order = np.argsort(-scores, kind="mergesort")
    ordered_labels = labels_array[order]
    # With stable (near-continuous) model scores, this computes the pairwise
    # AUC from the positive rank positions without O(P*N) comparisons.
    positive_positions = np.flatnonzero(ordered_labels) + 1
    wins = positives * negatives - (positive_positions.sum() - positives * (positives + 1) / 2)
    auc = float(wins / (positives * negatives))
    first = int(positive_positions[0])
    mrr = 1.0 / first
    def ndcg(k: int) -> float:
        dcg = sum(1.0 / math.log2(position + 1.0) for position in positive_positions if position <= k)
        ideal = sum(1.0 / math.log2(position + 1.0) for position in range(1, min(k, positives) + 1))
        return dcg / ideal if ideal else 0.0
    return auc, mrr, ndcg(5), ndcg(10)


def bootstrap_ci(samples: np.ndarray, seed: int, rounds: int = 1000) -> list[float]:
    if len(samples) == 0:
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    # Keep the matrix bounded: evaluating the full dev split needs no gigantic
    # bootstrap index matrix.
    means = np.empty(rounds, dtype=np.float64)
    n = len(samples)
    for index in range(rounds):
        means[index] = samples[rng.integers(0, n, n)].mean()
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def evaluate(
    archive: Path,
    catalog: Catalog,
    model,
    popularity: Counter,
    limit: int,
    bootstrap_rounds: int,
    output: Path | None,
    batch_impressions: int = 2_000,
    model_feature_names: Sequence[str] | None = None,
) -> dict:
    """Evaluate grouped MIND dev impressions in source order, bounded in memory."""
    scale = max(popularity.values(), default=1)
    totals = np.zeros(4, dtype=np.float64)
    baseline_totals = np.zeros(4, dtype=np.float64)
    values: list[tuple[float, float, float, float]] = []
    baseline_values: list[tuple[float, float, float, float]] = []
    count = 0
    pending_labels: list[list[int]] = []
    pending_baseline_scores: list[np.ndarray] = []
    pending_features: list[np.ndarray] = []

    def flush() -> None:
        """Score thousands of groups at once rather than predict per row."""
        nonlocal count, pending_labels, pending_baseline_scores, pending_features
        if not pending_features:
            return
        scores = model_scores(model, np.vstack(pending_features), model_feature_names)
        offset = 0
        for labels, baseline, matrix in zip(pending_labels, pending_baseline_scores, pending_features):
            width = len(matrix)
            current = per_impression_metrics(scores[offset:offset + width], labels)
            current_baseline = per_impression_metrics(baseline, labels)
            offset += width
            totals[:] += current
            baseline_totals[:] += current_baseline
            values.append(current)
            baseline_values.append(current_baseline)
            count += 1
        pending_labels, pending_baseline_scores, pending_features = [], [], []
    behaviors_member = archive_member(archive, "/behaviors.tsv")
    with zipfile.ZipFile(archive) as bundle, bundle.open(behaviors_member) as raw:
        for line in tqdm(raw, desc="Evaluate MINDlarge_dev", unit="impression"):
            _id, _user, _timestamp, history, candidates, labels = parse_behavior(line, labelled=True)
            if not any(labels):
                continue
            x = feature_matrix(candidates, make_context(history, catalog), catalog, popularity, scale)
            # Causal train-click popularity is a useful auditable baseline.
            baseline = np.asarray([popularity.get(article, 0) for article in candidates], dtype=np.float64)
            pending_labels.append(labels)
            pending_baseline_scores.append(baseline)
            pending_features.append(x)
            if len(pending_features) >= batch_impressions:
                flush()
            # Include unflushed rows in the limit condition.
            if limit and count + len(pending_features) >= limit:
                break
    flush()
    names = ["auc", "mrr", "ndcg@5", "ndcg@10"]
    metric_matrix = np.asarray(values, dtype=np.float64)
    baseline_matrix = np.asarray(baseline_values, dtype=np.float64)
    result = {
        "dataset": "MINDlarge",
        "split": "official_future_dev",
        "impressions": count,
        "model": {name: float(totals[index] / max(count, 1)) for index, name in enumerate(names)},
        "popularity_baseline": {name: float(baseline_totals[index] / max(count, 1)) for index, name in enumerate(names)},
        "bootstrap_95_ci": {
            name: bootstrap_ci(metric_matrix[:, index], seed=42 + index, rounds=bootstrap_rounds)
            for index, name in enumerate(names)
        },
        "notes": [
            "MINDlarge_dev is later than MINDlarge_train; no random interaction split is used.",
            "Popularity uses MINDlarge_train clicks only, so it contains no dev click labels.",
        ],
    }
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


def file_fingerprint(paths: Sequence[Path], model_path: Path, limit: int) -> str:
    digest = hashlib.sha256()
    digest.update(Path(__file__).read_bytes())
    digest.update(model_path.read_bytes())
    digest.update(str(limit).encode())
    for path in paths:
        stat = path.stat()
        digest.update(str(path.resolve()).encode())
        digest.update(str(stat.st_size).encode())
        digest.update(str(stat.st_mtime_ns).encode())
    return digest.hexdigest()


def validate_line(expected_id: str, candidates: Sequence[str], line: str) -> None:
    if not line.endswith("\n"):
        raise ValueError(f"Incomplete output line for impression {expected_id}")
    actual_id, rank_text = line.rstrip("\n").split(" ", 1)
    if not rank_text.startswith("[") or not rank_text.endswith("]"):
        raise ValueError(f"Malformed rank list for impression {expected_id}")
    ranks = [int(value) for value in rank_text[1:-1].split(",") if value]
    if actual_id != expected_id or len(ranks) != len(candidates) or sorted(ranks) != list(range(1, len(candidates) + 1)):
        raise ValueError(f"Invalid rank permutation for impression {expected_id}")


def resume_position(test_archive: Path, partial: Path, state: Path, fingerprint: str, limit: int) -> int:
    # Treat an empty output directory as a valid fresh start.  This makes the
    # documented ``--resume`` command safe to use from the very first run,
    # while still rejecting a damaged half-checkpoint (only one file present).
    if not partial.exists() and not state.exists():
        return 0
    if not partial.exists() or not state.exists():
        raise FileNotFoundError("A resumable run needs both prediction.txt.partial and prediction.resume.json")
    checkpoint = json.loads(state.read_text())
    if checkpoint.get("fingerprint") != fingerprint:
        raise ValueError("The checkpoint was generated by another model, archive, or code version.")
    completed = int(checkpoint["completed"])
    byte_offset = int(checkpoint["byte_offset"])
    with partial.open("r+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() < byte_offset:
            raise ValueError("Partial output is shorter than its checkpoint.")
        handle.truncate(byte_offset)
    # Verify the complete prefix so an old/malformed partial cannot silently
    # become a Codabench submission.
    checked = 0
    member = archive_member(test_archive, "/behaviors.tsv")
    with zipfile.ZipFile(test_archive) as bundle, bundle.open(member) as source, partial.open(encoding="utf-8") as prediction:
        for line in source:
            if limit and checked >= limit:
                break
            output_line = prediction.readline()
            if not output_line:
                break
            impression_id, _user, _time, _history, candidates, _labels = parse_behavior(line, labelled=False)
            validate_line(impression_id, candidates, output_line)
            checked += 1
        if prediction.readline():
            raise ValueError("Partial prediction contains rows beyond the verified source prefix.")
    if checked != completed:
        raise ValueError(f"Checkpoint says {completed} rows but only {checked} valid rows were found.")
    return completed


def write_state(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    os.replace(temporary, path)


def validate_prediction_file(test_archive: Path, prediction: Path, limit: int) -> int:
    checked = 0
    member = archive_member(test_archive, "/behaviors.tsv")
    with zipfile.ZipFile(test_archive) as bundle, bundle.open(member) as source, prediction.open(encoding="utf-8") as output:
        for line in tqdm(source, desc="Validate MIND v2 output", unit="impression"):
            if limit and checked >= limit:
                break
            output_line = output.readline()
            if not output_line:
                raise ValueError("Prediction has fewer rows than the requested test prefix.")
            impression_id, _user, _time, _history, candidates, _labels = parse_behavior(line, labelled=False)
            validate_line(impression_id, candidates, output_line)
            checked += 1
        if output.readline():
            raise ValueError("Prediction has more rows than the requested test prefix.")
    return checked


def submit(
    test_archive: Path,
    catalog: Catalog,
    model,
    popularity: Counter,
    output_dir: Path,
    model_path: Path,
    batch_impressions: int,
    limit: int,
    resume: bool,
    model_feature_names: Sequence[str] | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction = output_dir / "prediction.txt"
    partial = output_dir / "prediction.txt.partial"
    checkpoint = output_dir / "prediction.resume.json"
    submission = output_dir / "mind_v2_submission.zip"
    fingerprint = file_fingerprint([test_archive], model_path, limit)
    if not resume and (prediction.exists() or partial.exists() or checkpoint.exists() or submission.exists()):
        raise FileExistsError(
            f"A v2 artifact already exists in {output_dir}. Preserve it and choose a new --output-dir, "
            "or use --resume only for its matching partial run."
        )
    completed = resume_position(test_archive, partial, checkpoint, fingerprint, limit) if resume else 0
    scale = max(popularity.values(), default=1)
    buffer_meta: list[tuple[str, list[str]]] = []
    buffer_features: list[np.ndarray] = []
    mode = "ab" if completed else "wb"
    member = archive_member(test_archive, "/behaviors.tsv")
    with zipfile.ZipFile(test_archive) as bundle, bundle.open(member) as source, partial.open(mode) as output:
        for _ in range(completed):
            if not source.readline():
                raise ValueError("Checkpoint is beyond the end of the test archive.")

        def flush() -> None:
            nonlocal completed, buffer_meta, buffer_features
            if not buffer_meta:
                return
            scores = model_scores(model, np.vstack(buffer_features), model_feature_names)
            offset = 0
            lines: list[str] = []
            for impression_id, candidates in buffer_meta:
                width = len(candidates)
                ranks = ranks_from_scores(scores[offset:offset + width])
                offset += width
                lines.append(f"{impression_id} [{','.join(map(str, ranks.tolist()))}]\n")
            output.write("".join(lines).encode("utf-8"))
            output.flush()
            os.fsync(output.fileno())
            completed += len(buffer_meta)
            write_state(checkpoint, {
                "version": CHECKPOINT_VERSION,
                "fingerprint": fingerprint,
                "completed": completed,
                "byte_offset": output.tell(),
                "last_impression_id": buffer_meta[-1][0],
            })
            print(f"  checkpointed {completed:,} predictions")
            buffer_meta, buffer_features = [], []

        for line in tqdm(source, desc="Score MINDlarge_test v2", unit="impression"):
            if limit and completed + len(buffer_meta) >= limit:
                break
            impression_id, _user, _timestamp, history, candidates, _labels = parse_behavior(line, labelled=False)
            if not candidates:
                raise ValueError(f"Empty candidate list for impression {impression_id}")
            buffer_meta.append((impression_id, candidates))
            buffer_features.append(feature_matrix(candidates, make_context(history, catalog), catalog, popularity, scale))
            if len(buffer_meta) >= batch_impressions:
                flush()
        flush()
    if limit and completed != limit:
        raise ValueError(f"Requested {limit} output rows but wrote {completed}.")
    os.replace(partial, prediction)
    checked = validate_prediction_file(test_archive, prediction, limit)
    with zipfile.ZipFile(submission, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as bundle:
        bundle.write(prediction, arcname="prediction.txt")
    checkpoint.unlink(missing_ok=True)
    print(f"Validated {checked:,} rank lists. Ready: {submission}")
    if limit:
        print("This is a smoke ZIP only; run without --limit before submitting.")
    return submission


def add_common_archives(parser: argparse.ArgumentParser, include_test: bool = False) -> None:
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--dev", type=Path, default=DEFAULT_DEV)
    if include_test:
        parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--rebuild-catalog", action="store_true")


def ensure_archives(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing MIND archive(s):\n  " + "\n  ".join(missing))


def command_train(args) -> None:
    archives = [args.train] + ([args.dev] if args.include_dev_in_train else [])
    ensure_archives(archives)
    catalog_archives = [args.train, args.dev]
    catalog = build_catalog(catalog_archives, args.cache_dir, force=args.rebuild_catalog)
    split_tag = "train_plus_dev" if args.include_dev_in_train else "train"
    # The sampling configuration is part of the training artifact.  Keeping
    # it in the name prevents a later --sample-modulus change from silently
    # reusing a weaker cache.
    matrix_path = args.cache_dir / f"{split_tag}_sample_mod{args.sample_modulus}_neg{args.negatives_per_positive}.npz"
    if matrix_path.exists() and not args.rebuild_matrix:
        x, y, groups, metadata = load_training_matrix(matrix_path)
        print(f"Reusing {matrix_path}: {metadata.get('selected_impressions', len(groups)):,} groups / {len(y):,} rows")
    else:
        x, y, groups = build_training_matrix(
            archives, catalog, matrix_path, args.sample_modulus, args.negatives_per_positive,
            args.seed, args.max_impressions,
        )
    print(f"Training v2 ranker on {len(groups):,} impression groups / {len(y):,} candidates...")
    model, backend = train_ranker(x, y, groups, args.seed)
    model_path = args.model or (args.cache_dir / ("mind_v2_final.pkl" if args.include_dev_in_train else "mind_v2.pkl"))
    save_model(model_path, model, backend, matrix_path)
    print(f"Saved {backend} model: {model_path}")


def command_evaluate(args) -> None:
    ensure_archives([args.train, args.dev])
    catalog = build_catalog([args.train, args.dev], args.cache_dir, force=args.rebuild_catalog)
    model_path = args.model or (args.cache_dir / "mind_v2.pkl")
    model, payload = load_model(model_path)
    _daily, popularity = daily_clicks(args.train)
    print(f"Evaluating model backend={payload.get('backend')} with {len(popularity):,} train-clicked articles")
    evaluate(
        args.dev, catalog, model, popularity, args.limit, args.bootstrap_rounds,
        args.output, args.batch_impressions, payload.get("feature_names"),
    )


def command_submit(args) -> None:
    ensure_archives([args.train, args.dev, args.test])
    catalog = build_catalog([args.train, args.dev, args.test], args.cache_dir, force=args.rebuild_catalog)
    model_path = args.model or (args.cache_dir / "mind_v2_final.pkl")
    model, payload = load_model(model_path)
    _train_daily, train_popularity = daily_clicks(args.train)
    _dev_daily, dev_popularity = daily_clicks(args.dev)
    train_popularity.update(dev_popularity)  # all are chronologically before public test serving.
    print(f"Submitting model backend={payload.get('backend')}; popularity source=train+dev only.")
    submit(
        args.test, catalog, model, train_popularity, args.output_dir, model_path,
        args.batch_impressions, args.limit, args.resume, payload.get("feature_names"),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subcommands = parser.add_subparsers(dest="command", required=True)
    train = subcommands.add_parser("train", help="sample MINDlarge labels and train the v2 re-ranker")
    add_common_archives(train)
    train.add_argument("--model", type=Path, default=None)
    train.add_argument("--sample-modulus", type=int, default=25, help="keep 1/N deterministic impression groups (default: 1/25)")
    train.add_argument("--negatives-per-positive", type=int, default=20)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--max-impressions", type=int, default=0, help="cap selected groups; 0 means no cap")
    train.add_argument("--include-dev-in-train", action="store_true", help="only after validation/tuning; creates the final train+dev model")
    train.add_argument("--rebuild-matrix", action="store_true")
    train.set_defaults(func=command_train)

    evaluate_parser = subcommands.add_parser("evaluate", help="evaluate a v2 model on the future MINDlarge_dev split")
    add_common_archives(evaluate_parser)
    evaluate_parser.add_argument("--model", type=Path, default=None)
    evaluate_parser.add_argument("--limit", type=int, default=0, help="cap dev impressions; 0 evaluates all")
    evaluate_parser.add_argument("--batch-impressions", type=int, default=2_000, help="dev impression groups per vectorised prediction")
    evaluate_parser.add_argument("--bootstrap-rounds", type=int, default=1000)
    evaluate_parser.add_argument("--output", type=Path, default=Path("outputs/mind_v2_validation.json"))
    evaluate_parser.set_defaults(func=command_evaluate)

    submit_parser = subcommands.add_parser("submit", help="make a root-only MIND Codabench ZIP")
    add_common_archives(submit_parser, include_test=True)
    submit_parser.add_argument("--model", type=Path, default=None)
    submit_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    submit_parser.add_argument("--batch-impressions", type=int, default=4000)
    submit_parser.add_argument("--limit", type=int, default=0, help="smoke test only; never submit a limited ZIP")
    submit_parser.add_argument("--resume", action="store_true")
    submit_parser.set_defaults(func=command_submit)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
