"""Regression tests for fast, submission-time feature generation.

These deliberately use tiny synthetic data so they can run without the large
MIND or EB-NeRD archives.  They guard against an optimization silently
changing a score or creating a malformed Codabench rank list.
"""
from __future__ import annotations

import json
from math import exp, log
import zipfile

import numpy as np
import pandas as pd
import pytest

from bm25_retrieval import InvertedBM25, tokenize
from features import candidate_features, category_affinity, semantic_score, topic_affinity
from make_ebnerd_personalized_submission import validate_prediction_line as validate_ebnerd_line
from make_mind_submission import resume_count as mind_resume_count
from make_mind_submission import validate_prediction_line as validate_mind_line


def legacy_candidate_scores(index: InvertedBM25, query: str, candidate_ids: list[str]) -> dict[str, float]:
    """The pre-optimization BM25 arithmetic, kept only as a test oracle."""
    wanted = {article: position for position, article in enumerate(index.ids) if article in set(candidate_ids)}
    scores = {article: 0.0 for article in candidate_ids}
    for term in set(tokenize(query)):
        posting = index.postings.get(term, [])
        if not posting:
            continue
        idf = log(1 + (index.n_docs - len(posting) + 0.5) / (len(posting) + 0.5))
        for document_index, tf in posting:
            article = index.ids[document_index]
            if article not in wanted:
                continue
            denominator = tf + index.k1 * (1 - index.b + index.b * index.lengths[document_index] / index.average_length)
            scores[article] += idf * tf * (index.k1 + 1) / denominator
    return scores


def test_candidate_driven_bm25_matches_the_original_formula():
    index = InvertedBM25(
        ["a", "b", "c", "d"],
        ["cat cat space", "space travel cat", "politics election", "unknown tokens"],
    )
    candidates = ["c", "b", "not-in-catalog", "a"]
    expected = legacy_candidate_scores(index, "cat space cat", candidates)
    actual = index.candidate_scores("cat space cat", candidates)

    assert actual.keys() == expected.keys()
    for article in expected:
        assert actual[article] == pytest.approx(expected[article])


def test_cached_candidate_features_match_the_existing_feature_definition():
    history = ["a", "b", "a"]
    weights = [0.2, 0.5, 1.0]
    categories = {"a": "sports", "b": "tech", "candidate": "sports"}
    topics = {"a": ("football",), "b": ("ai",), "candidate": ("football", "ai")}
    popularity = {"candidate": 7}
    positions = {"a": 0, "b": 1, "candidate": 2}
    vectors = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.8, 0.6]], dtype=np.float32)
    now = pd.Timestamp("2024-01-02 12:00:00")
    published = {"candidate": pd.Timestamp("2024-01-01 12:00:00")}

    actual = candidate_features(
        "candidate", 2, history, weights, categories, topics, popularity, 10,
        positions, vectors, published, now, 3, 4, 12.5,
    )
    expected = {
        "article_id": "candidate",
        "category_affinity": category_affinity(history, weights, categories, "sports"),
        "topic_affinity": topic_affinity(history, weights, topics, topics["candidate"]),
        "semantic_score": semantic_score(history, weights, positions, vectors, "candidate"),
        "popularity": log(8) / log(11),
        "freshness_hours_inv": exp(-24 / (24 * 4)),
        "session_prior_impressions": 3,
        "session_clicks_so_far": 4,
        "session_mean_dwell_time": 12.5,
        "position_bias": 1.0 / log(4, 2),
        "history_length": 3,
    }
    assert actual["article_id"] == expected["article_id"]
    for name, value in expected.items():
        if name != "article_id":
            assert actual[name] == pytest.approx(value)


@pytest.mark.parametrize("validator", [validate_mind_line, validate_ebnerd_line])
def test_submission_rank_lists_require_an_ordered_full_permutation(validator):
    validator("42", ["a", "b", "c"], "42 [3,1,2]\n")
    with pytest.raises(ValueError):
        validator("42", ["a", "b", "c"], "42 [1,1,2]\n")
    with pytest.raises(ValueError):
        validator("42", ["a", "b", "c"], "43 [1,2,3]\n")


def test_resume_keeps_only_the_durably_checkpointed_mind_prefix(tmp_path):
    """A crash after a write but before its checkpoint must not duplicate rows."""
    archive = tmp_path / "test.zip"
    source_rows = (
        "1\tu\t2024-01-01\t\tN1 N2\n"
        "2\tu\t2024-01-01\t\tN3 N4\n"
        "3\tu\t2024-01-01\t\tN5 N6\n"
    )
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("test/behaviors.tsv", source_rows)

    durable_prefix = "1 [1,2]\n2 [2,1]\n"
    partial = tmp_path / "prediction.txt.partial"
    partial.write_text(durable_prefix + "3 [1,2]\n", encoding="utf-8")
    checkpoint = tmp_path / "prediction.txt.resume.json"
    checkpoint.write_text(json.dumps({
        "fingerprint": "synthetic-run",
        "completed": 2,
        "byte_offset": len(durable_prefix.encode("utf-8")),
    }), encoding="utf-8")

    assert mind_resume_count(archive, partial, checkpoint, "synthetic-run", 0) == 2
    assert partial.read_text(encoding="utf-8") == durable_prefix
