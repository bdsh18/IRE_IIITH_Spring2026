from pathlib import Path
import pandas as pd
import pytest

from features import build_features, causal_click_times, causal_popularity_snapshot, user_session_click_count, user_session_mean_dwell, user_session_order


def _impressions(rows):
    frame = pd.DataFrame(rows, columns=["impression_id", "user_id", "timestamp", "candidate_ids", "clicked_ids", "history_ids"])
    frame["timestamp"] = pd.to_datetime(frame.timestamp)
    frame["history_recency_weights"] = frame.history_ids.map(lambda h: [0.8 ** (len(h) - 1 - i) for i in range(len(h))])
    frame["session_id"] = ""
    frame["dwell_time"] = 0.0
    return frame


def _toy_store(root: Path, validation_clicks: list[str]) -> Path:
    """Tiny end-to-end store: two train impressions, one validation impression."""
    store = root / "store"
    folder = store / "toy"
    folder.mkdir(parents=True)
    pd.DataFrame({
        "article_id": ["a", "b", "c"],
        "title": ["alpha news", "beta news", "gamma news"],
        "abstract": ["", "", ""],
        "category": ["sports", "tech", "sports"],
    }).to_parquet(folder / "articles.parquet", index=False)
    _impressions([
        ("tr1", "u1", "2024-01-01 10:00", ["a", "b"], ["a"], ["c"]),
        ("tr2", "u2", "2024-01-01 12:00", ["a", "b"], ["b"], ["c"]),
    ]).to_parquet(folder / "train_impressions.parquet", index=False)
    validation = _impressions([("va1", "u3", "2024-01-02 10:00", ["a", "b", "c"], validation_clicks, ["a"])])
    validation.to_parquet(folder / "validation_impressions.parquet", index=False)
    validation.to_parquet(folder / "test_impressions.parquet", index=False)
    return store


def test_build_features_popularity_is_causal_inside_train(tmp_path):
    """End-to-end: the real feature builder never counts a click at or after t."""
    features = build_features(_toy_store(tmp_path, ["a"]), "toy", "train").set_index(["impression_id", "article_id"])
    assert features.loc[("tr1", "a"), "popularity"] == 0, "own click at t leaked into popularity"
    assert features.loc[("tr2", "b"), "popularity"] == 0, "own click at t leaked into popularity"
    assert features.loc[("tr1", "b"), "popularity"] == 0, "a later click leaked backwards in time"
    assert features.loc[("tr2", "a"), "popularity"] > 0, "sanity: an earlier click must be visible"


def test_validation_features_do_not_depend_on_validation_labels(tmp_path):
    """Flipping which candidate was clicked must not change any feature value.

    If any feature read the impression's own labels, or clicks from the
    validation period, the two feature tables would differ.
    """
    clicked_a = build_features(_toy_store(tmp_path / "one", ["a"]), "toy", "validation")
    clicked_c = build_features(_toy_store(tmp_path / "two", ["c"]), "toy", "validation")
    assert clicked_a.label.tolist() != clicked_c.label.tolist(), "sanity: the labels really differ"
    pd.testing.assert_frame_equal(clicked_a.drop(columns="label"), clicked_c.drop(columns="label"))
    assert clicked_c.set_index("article_id").loc["c", "popularity"] == 0


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

def test_session_features_only_use_earlier_events_and_do_not_fake_mind_sessions():
    frame = pd.DataFrame({
        "impression_id": ["first", "second", "mind"], "user_id": ["u", "u", "m"],
        "session_id": ["s", "s", ""], "timestamp": pd.to_datetime(["2024-01-01 10:00", "2024-01-01 10:05", "2024-01-01 10:10"]),
        "clicked_ids": [["x"], ["y"], ["z"]], "dwell_time": [12.0, 30.0, 99.0],
    })
    assert user_session_order(frame) == {"first": 0, "second": 1, "mind": 0}
    assert user_session_click_count(frame) == {"first": 0, "second": 1, "mind": 0}
    dwell = user_session_mean_dwell(frame)
    assert dwell["first"] == 0.0 and dwell["second"] == 12.0 and dwell["mind"] == 0.0

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