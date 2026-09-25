from __future__ import annotations
import argparse
import json
import zipfile
from pathlib import Path
import pandas as pd

MIND_NEWS = ["article_id", "category", "subcategory", "title", "abstract", "url", "title_entities", "abstract_entities"]
MIND_BEHAVIORS = ["impression_id", "user_id", "timestamp", "history", "raw_candidates"]

def extract_once(archive: Path, target: Path) -> None:
    marker = target / ".extracted_from"
    if marker.exists() and marker.read_text() == archive.name:
        return
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        safe_names = [n for n in bundle.namelist() if not n.startswith("__MACOSX/") and ".." not in Path(n).parts]
        for member in safe_names:
            bundle.extract(member, target)
    marker.write_text(archive.name)

def tokens(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)): return []
    if not isinstance(value, str) and hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)): return [str(x) for x in value]
    return str(value).split()

def mind_candidates(value: str) -> tuple[list[str], list[str]]:
    candidates, clicked = [], []
    for item in tokens(value):
        article, separator, label = item.rpartition("-")
        candidates.append(article if separator else item)
        if label == "1": clicked.append(article)
    return candidates, clicked

def recency_weights(history: list[str], decay: float = .8) -> list[float]:
    return [round(decay ** (len(history) - 1 - i), 6) for i in range(len(history))]

def load_mind(folder: Path, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    articles = pd.read_csv(folder / "news.tsv", sep="\t", names=MIND_NEWS, header=None, quoting=3)
    articles = articles.assign(dataset="mind", source_split=split, body="", entities=articles.title_entities.fillna("") + " " + articles.abstract_entities.fillna(""), embedding_available=False, embedding_path="")
    articles["article_id"] = articles.article_id.astype(str)
    behaviors = pd.read_csv(folder / "behaviors.tsv", sep="\t", names=MIND_BEHAVIORS, header=None, quoting=3)
    parsed = behaviors.raw_candidates.map(mind_candidates)
    impressions = pd.DataFrame({"dataset": "mind", "source_split": split, "impression_id": f"{split}-" + behaviors.impression_id.astype(str), "user_id": behaviors.user_id.astype(str), "timestamp": pd.to_datetime(behaviors.timestamp), "candidate_ids": parsed.map(lambda x: x[0]), "clicked_ids": parsed.map(lambda x: x[1]), "history_ids": behaviors.history.map(tokens), "session_id": "", "dwell_time": 0.0})
    impressions["history_recency_weights"] = impressions.history_ids.map(recency_weights)
    return articles[["dataset", "source_split", "article_id", "title", "abstract", "body", "category", "subcategory", "entities", "embedding_available", "embedding_path"]], impressions

def pick(frame: pd.DataFrame, *names: str, default: object = "") -> pd.Series:
    for name in names:
        if name in frame.columns: return frame[name]
    return pd.Series([default] * len(frame), index=frame.index)

def load_ebnerd(folder: Path, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_articles = pd.read_parquet(folder / "articles.parquet")
    articles = pd.DataFrame({"dataset": "ebnerd", "source_split": split, "article_id": pick(source_articles, "article_id").astype(str), "title": pick(source_articles, "title"), "abstract": pick(source_articles, "subtitle", "abstract"), "body": pick(source_articles, "body"), "category": pick(source_articles, "category_str", "category"), "entities": pick(source_articles, "entity_groups", "ner_clusters"), "topics": pick(source_articles, "topics", default=None), "published_time": pick(source_articles, "published_time", default=pd.NaT), "embedding_available": False, "embedding_path": ""})
    behaviors = pd.read_parquet(folder / split / "behaviors.parquet")
    history = pd.read_parquet(folder / split / "history.parquet")
    history_id_col = "article_id_fixed" if "article_id_fixed" in history else "article_id"
    history_map = dict(zip(history.user_id.astype(str), history[history_id_col].map(tokens)))
    impressions = pd.DataFrame({"dataset": "ebnerd", "source_split": split, "impression_id": pick(behaviors, "impression_id").astype(str), "user_id": pick(behaviors, "user_id").astype(str), "timestamp": pd.to_datetime(pick(behaviors, "impression_time")), "candidate_ids": pick(behaviors, "article_ids_inview").map(tokens), "clicked_ids": pick(behaviors, "article_ids_clicked").map(tokens), "session_id": pick(behaviors, "session_id").astype(str), "dwell_time": pd.to_numeric(pick(behaviors, "read_time", default=0.0), errors="coerce").fillna(0.0)})
    impressions["history_ids"] = impressions.user_id.map(history_map).map(lambda x: x if isinstance(x, list) else [])
    impressions["history_recency_weights"] = impressions.history_ids.map(recency_weights)
    return articles, impressions

def temporal_split(impressions: pd.DataFrame, validation_days: int, test_days: int) -> dict[str, pd.DataFrame]:
    ordered = impressions.sort_values("timestamp").reset_index(drop=True)
    end = ordered.timestamp.max(); test_start = end - pd.Timedelta(days=test_days); validation_start = test_start - pd.Timedelta(days=validation_days)
    return {"train": ordered[ordered.timestamp < validation_start], "validation": ordered[(ordered.timestamp >= validation_start) & (ordered.timestamp < test_start)], "test": ordered[ordered.timestamp >= test_start]}

def write_dataset(name: str, articles: pd.DataFrame, impressions: pd.DataFrame, destination: Path, validation_days: int, test_days: int) -> dict[str, int]:
    output = destination / name; output.mkdir(parents=True, exist_ok=True)
    articles.drop_duplicates("article_id").to_parquet(output / "articles.parquet", index=False)
    counts = {"articles": len(articles.drop_duplicates("article_id"))}
    for split, frame in temporal_split(impressions, validation_days, test_days).items():
        frame.to_parquet(output / f"{split}_impressions.parquet", index=False); counts[split] = len(frame)
    return counts

def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--source-root", type=Path, default=Path(".")); parser.add_argument("--output", type=Path, default=Path("data/processed")); parser.add_argument("--validation-days", type=int, default=1); parser.add_argument("--test-days", type=int, default=1); args = parser.parse_args()
    root = args.source_root.resolve(); raw = root / "data/raw"; raw.mkdir(parents=True, exist_ok=True)
    extract_once(root / "MIND_data/MINDsmall_train.zip", raw / "mind_train")
    extract_once(root / "MIND_data/MINDsmall_dev.zip", raw / "mind_dev")
    extract_once(root / "ebnerd_demo.zip", raw / "ebnerd_demo")
    mind_train = load_mind(raw / "mind_train/MINDsmall_train", "train"); mind_dev = load_mind(raw / "mind_dev/MINDsmall_dev", "dev")
    eb_train = load_ebnerd(raw / "ebnerd_demo", "train"); eb_val = load_ebnerd(raw / "ebnerd_demo", "validation")
    summary = {"mind": write_dataset("mind", pd.concat([mind_train[0], mind_dev[0]]), pd.concat([mind_train[1], mind_dev[1]]), args.output, args.validation_days, args.test_days), "ebnerd": write_dataset("ebnerd", pd.concat([eb_train[0], eb_val[0]]), pd.concat([eb_train[1], eb_val[1]]), args.output, args.validation_days, args.test_days)}
    (args.output / "manifest.json").write_text(json.dumps(summary, indent=2)); print(json.dumps(summary, indent=2))

if __name__ == "__main__": main()