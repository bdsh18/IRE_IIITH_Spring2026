"""Fast unit checks for the separate large-data score-improvement paths."""
from __future__ import annotations

from collections import Counter

import numpy as np

from ebnerd_large_v2 import (
    ArticleStore,
    HourlyPopularity,
    candidate_features,
    profile_from_history,
    ranks_from_scores as eb_ranks,
    valid_prediction_line,
)
from mind_large_v2 import (
    Catalog,
    FEATURE_NAMES,
    day_prefixes,
    feature_matrix,
    make_context,
    per_impression_metrics,
    ranks_from_scores,
    selected_indices,
)


def tiny_catalog() -> Catalog:
    return Catalog(
        category={"a": "news", "b": "news", "c": "sports"},
        subcategory={"a": "world", "b": "world", "c": "soccer"},
        entities={"a": frozenset({"q1"}), "b": frozenset({"q1", "q2"}), "c": frozenset({"q3"})},
        title_tokens={"a": frozenset({"election", "news"}), "b": frozenset({"election"}), "c": frozenset({"football"})},
        vectors={
            "a": np.asarray([1.0, 0.0], dtype=np.float32),
            "b": np.asarray([1.0, 0.0], dtype=np.float32),
            "c": np.asarray([0.0, 1.0], dtype=np.float32),
        },
        token_idf={"election": 3.0, "news": 2.0, "football": 3.0},
    )


def test_mind_v2_features_have_expected_shape_and_preference() -> None:
    catalog = tiny_catalog()
    matrix = feature_matrix(["b", "c"], make_context(["a"], catalog), catalog, Counter({"b": 3}), 3)
    assert matrix.shape == (2, len(FEATURE_NAMES))
    # Same category/entity/text candidate wins the history signals.
    assert matrix[0, 0] > matrix[1, 0]
    assert matrix[0, 7] > matrix[1, 7]
    assert matrix[0, 16] > matrix[1, 16]


def test_mind_v2_daily_popularity_is_strictly_prior_day() -> None:
    prefixes = day_prefixes({"2019-11-10": Counter({"a": 2}), "2019-11-11": Counter({"a": 3, "b": 1})})
    assert prefixes["2019-11-10"]["a"] == 0
    assert prefixes["2019-11-11"]["a"] == 2
    assert prefixes["2019-11-11"]["b"] == 0


def test_v2_rank_lists_are_source_ordered_permutations() -> None:
    expected = [3, 1, 2]
    assert ranks_from_scores(np.asarray([0.2, 0.8, 0.5])).tolist() == expected
    assert eb_ranks([0.2, 0.8, 0.5]) == expected
    assert valid_prediction_line("42", ("a", "b", "c"), "42 [3,1,2]\n")


def test_group_sampling_and_metrics_keep_whole_impressions() -> None:
    indices = selected_indices("impression", [1, 0, 0, 0], negatives_per_positive=2, seed=42)
    assert indices[0] == 0 and len(indices) == 3
    assert per_impression_metrics(np.asarray([0.9, 0.1]), [1, 0])[0] == 1.0


def test_ebnerd_semantic_max_is_finite_under_strict_numpy_errors() -> None:
    """Guard the Apple-Accelerate-safe matrix/vector feature path."""
    articles = ArticleStore(
        category={"h": "news", "c": "news"},
        subcategories={"h": (), "c": ()},
        topics={"h": (), "c": ()},
        entities={"h": (), "c": ()},
        published={"h": None, "c": None},
        title={"h": "history", "c": "candidate"},
        title_tokens={"h": 1.0, "c": 1.0},
        premium={"h": 0.0, "c": 0.0},
        sentiment={"h": 0.0, "c": 0.0},
        bm25=None,
    )
    positions = {"h": 0, "c": 1}
    vectors = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    profile = profile_from_history(("h",), articles, positions, vectors, decay=0.85)
    with np.errstate(all="raise"):
        features = candidate_features(
            "c", 0, profile, None, articles, HourlyPopularity({}, Counter(), 1),
            False, positions, vectors, 0.0, 0, False, False, None,
        )
    assert np.isfinite(features["semantic_max"])
    assert features["semantic_max"] == 0.0
