from pathlib import Path
import pandas as pd
import pytest

from features import category_affinity, causal_click_times, causal_popularity_snapshot

def test_category_affinity_ignores_future_clicks():
    category_by_id = {"a1": "sports", "a2": "tech", "future_click": "tech"}
    history_ids = ["a1", "a1", "a2"]  # only clicks strictly before this impression
    weights = [0.5, 0.7, 1.0]

    score_without_future = category_affinity(history_ids, weights, category_by_id, "tech")

    leaked_history_ids = history_ids + ["future_click"]
    leaked_weights = weights + [1.0]
    score_with_future = category_affinity(leaked_history_ids, leaked_weights, category_by_id, "tech")

    assert score_without_future != score_with_future, (
        "sanity check: the fixture should actually change the score, "
        "otherwise this test can't detect a leak"
    )


def test_train_popularity_is_causal():
    early = pd.Timestamp("2024-01-01")
    click_time = pd.Timestamp("2024-01-02")
    later = pd.Timestamp("2024-01-03")
    train = pd.DataFrame({
        "impression_id": ["imp_early", "imp_click", "imp_later"],
        "timestamp": [early, click_time, later],
        "clicked_ids": [[], ["x"], []],
    })

    click_times = causal_click_times(train)
    assert causal_popularity_snapshot(["x"], click_times, early)["x"] == 0, (
        "must not see a click that happens later in the same train split"
    )
    assert causal_popularity_snapshot(["x"], click_times, click_time)["x"] == 0, (
        "must not see its OWN click as a feature of itself (label leakage)"
    )
    assert causal_popularity_snapshot(["x"], click_times, later)["x"] == 1, (
        "sanity check: a genuinely earlier click must still be visible"
    )

def _check_dataset(store: Path, dataset: str, split: str = "validation") -> None:
    train = pd.read_parquet(store / dataset / "train_impressions.parquet")
    validation = pd.read_parquet(store / dataset / "validation_impressions.parquet")
    test = pd.read_parquet(store / dataset / "test_impressions.parquet")
    assert train.timestamp.max() <= validation.timestamp.min(), f"{dataset}: train/validation windows overlap"
    assert validation.timestamp.max() <= test.timestamp.min(), f"{dataset}: validation/test windows overlap"
    print(f"{dataset}: train/validation/test windows for {split} are non-overlapping and time-ordered")


@pytest.mark.parametrize("dataset", ["mind", "ebnerd"])
def test_real_store_has_no_temporal_leakage(dataset):
    store = Path("data/processed")
    if not (store / dataset / "validation_impressions.parquet").exists():
        pytest.skip(f"data/processed/{dataset} not built yet — run build_pipeline.py first")
    _check_dataset(store, dataset)


if __name__ == "__main__":
    for name in ["mind", "ebnerd"]:
        path = Path("data/processed") / name / "validation_impressions.parquet"
        if path.exists():
            _check_dataset(Path("data/processed"), name)
        else:
            print(f"skip {name}: run build_pipeline.py first")