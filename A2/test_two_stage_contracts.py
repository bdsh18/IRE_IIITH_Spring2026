from pathlib import Path

from bm25_retrieval import InvertedBM25, retrieve_from_history
from features import SUBMISSION_TIME_UNAVAILABLE, ranker_feature_names
from reranker import apply_stage1_gate, rank_from_scores


def test_submission_schema_excludes_every_unavailable_session_feature():
    serving_columns = ranker_feature_names(serving_only=True)

    assert set(SUBMISSION_TIME_UNAVAILABLE).isdisjoint(serving_columns)
    assert "popularity" in serving_columns
    assert "semantic_score" in serving_columns
    assert len(serving_columns) == len(ranker_feature_names()) - len(SUBMISSION_TIME_UNAVAILABLE)


def test_submission_scripts_use_the_shared_serving_schema():
    root = Path(__file__).parent
    for name in ("make_mind_submission.py", "make_ebnerd_personalized_submission.py"):
        source = (root / name).read_text(encoding="utf-8")
        assert "ranker_feature_names(serving_only=True)" in source


def test_stage_one_gate_demotes_a_high_scored_nonretrieved_article():
    article_ids = ["outside", "retrieved", "also_outside"]
    raw_scores = [100.0, 1.0, 99.0]

    gated = apply_stage1_gate(article_ids, raw_scores, {"retrieved"})

    assert rank_from_scores(gated) == [2, 1, 3]


def test_catalog_retrieval_excludes_articles_already_in_history():
    ids = ["history_article", "new_article"]
    index = InvertedBM25(ids, ["alpha alpha", "alpha beta"])
    titles = {"history_article": "alpha", "new_article": "alpha beta"}

    retrieved = retrieve_from_history(index, ["history_article"], titles, recent=1, k=2)

    assert "history_article" not in retrieved
    assert retrieved == ["new_article"]
