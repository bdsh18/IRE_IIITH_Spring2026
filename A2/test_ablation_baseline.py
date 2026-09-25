import zipfile

import pandas as pd

from ablation import starter_prediction_scores, temporal_popularity_baseline_ablation


def _validation_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "impression_id": [101, 101, 202, 202],
            "article_id": ["a", "b", "c", "d"],
            "label": [0, 1, 1, 0],
            "popularity": [0.9, 0.1, 0.1, 0.9],
        }
    )


def test_external_starter_rank_file_is_aligned_and_uses_score_direction(tmp_path):
    features = _validation_features()
    prediction = tmp_path / "prediction.txt"
    prediction.write_text("101 [2,1]\n202 [1,2]\n", encoding="utf-8")

    scores = starter_prediction_scores(prediction, features)

    assert scores["101"]["labels"] == [0, 1]
    # Rank 1 must become the largest score because rank_from_scores sorts
    # scores in descending order.
    assert scores["101"]["starter_score"] == [-2.0, -1.0]


def test_external_starter_submission_zip_and_internal_baseline_are_clearly_named(tmp_path):
    features = _validation_features()
    prediction_zip = tmp_path / "starter.zip"
    with zipfile.ZipFile(prediction_zip, "w") as archive:
        archive.writestr("prediction.txt", "101 [2,1]\n202 [1,2]\n")

    scores = starter_prediction_scores(prediction_zip, features)
    full_scores = {
        "101": {"labels": [0, 1], "model_score": [0.0, 1.0]},
        "202": {"labels": [1, 0], "model_score": [1.0, 0.0]},
    }
    report = temporal_popularity_baseline_ablation(features, full_scores)

    assert set(scores) == {"101", "202"}
    assert report["baseline"] == "causal_temporal_popularity"
    assert report["baseline_source"] == "internal_feature_store"
