#!/usr/bin/env python3
"""Q3 semantic retrieval: MIND entity vectors and EB-NeRD document vectors."""
from __future__ import annotations
import argparse, json, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

from features import weighted_user_vector

def normalize(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)

def entity_ids(raw: object) -> list[str]:
    """Read Wikidata IDs from MIND's title/abstract entity JSON."""
    try: values = json.loads(raw) if isinstance(raw, str) and raw else []
    except json.JSONDecodeError: return []
    return [str(item.get("WikidataId")) for item in values if item.get("WikidataId")]

def mind_embeddings(source: Path, article_ids: list[str]) -> tuple[list[str], np.ndarray]:
    news = pd.concat([pd.read_csv(source / "data/raw/mind_train/MINDsmall_train/news.tsv", sep="\t", header=None, names=["id","category","subcategory","title","abstract","url","title_entities","abstract_entities"], quoting=3), pd.read_csv(source / "data/raw/mind_dev/MINDsmall_dev/news.tsv", sep="\t", header=None, names=["id","category","subcategory","title","abstract","url","title_entities","abstract_entities"], quoting=3)]).drop_duplicates("id")
    embeddings: dict[str, np.ndarray] = {}
    with (source / "data/raw/mind_train/MINDsmall_train/entity_embedding.vec").open() as file:
        for line in file:
            fields = line.rstrip().split("\t")
            embeddings[fields[0]] = np.asarray(fields[1:], dtype=np.float32)
    vectors = []; ids = []
    wanted = set(article_ids)
    for row in news.itertuples(index=False):
        article_id = str(row.id)
        if article_id not in wanted: continue
        entities = entity_ids(row.title_entities) + entity_ids(row.abstract_entities)
        vectors_for_article = [embeddings[entity] for entity in entities if entity in embeddings]
        if vectors_for_article:
            ids.append(article_id); vectors.append(np.mean(vectors_for_article, axis=0))
    return ids, normalize(np.vstack(vectors).astype(np.float32))

def ebnerd_embeddings(source: Path, article_ids: list[str]) -> tuple[list[str], np.ndarray]:
    archive = source / "Ekstra_Bladet_word2vec.zip"; extracted = source / "data/raw/ebnerd_embeddings/document_vector.parquet"
    if not extracted.exists():
        extracted.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as bundle, bundle.open("Ekstra_Bladet_word2vec/document_vector.parquet") as reader, extracted.open("wb") as writer:
            writer.write(reader.read())
    frame = pd.read_parquet(extracted)
    id_col = next(col for col in ["article_id", "article_id_fixed"] if col in frame)
    vector_col = next(col for col in frame if col not in {id_col} and len(frame[col]) and hasattr(frame[col].iloc[0], "__len__"))
    wanted = set(article_ids); frame[id_col] = frame[id_col].astype(str); frame = frame[frame[id_col].isin(wanted)]
    return frame[id_col].tolist(), normalize(np.vstack(frame[vector_col].to_list()).astype(np.float32))

def retrieval_metrics(emb_ids: list[str], vectors: np.ndarray, impressions: pd.DataFrame) -> dict[str, object]:
    position = {article: i for i, article in enumerate(emb_ids)}; values = {50: [], 100: [], 200: []}; evaluated = 0
    for row in impressions.itertuples(index=False):
        history = [position[str(x)] for x in row.history_ids if str(x) in position]
        clicked = {str(x) for x in row.clicked_ids}
        if not history or not clicked: continue
        user = weighted_user_vector(row.history_ids, row.history_recency_weights, position, vectors)
        if user is None: continue
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            scores = vectors @ user
        scores = np.nan_to_num(scores, nan=-np.inf, posinf=1.0, neginf=-np.inf); scores[history] = -np.inf
        top = np.argsort(-scores)[:200]; retrieved = [emb_ids[i] for i in top]
        for k in values: values[k].append(len(clicked.intersection(retrieved[:k])) / len(clicked))
        evaluated += 1
    return {"evaluated_impressions": evaluated, **{f"recall@{k}": float(np.mean(values[k])) if values[k] else 0.0 for k in values}}

def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--source-root", type=Path, default=Path(".")); parser.add_argument("--store", type=Path, default=Path("data/processed")); parser.add_argument("--limit", type=int, default=0); parser.add_argument("--output", type=Path, default=Path("outputs/q3_semantic_metrics.json")); args = parser.parse_args()
    results = []
    for dataset, builder in [("mind", mind_embeddings), ("ebnerd", ebnerd_embeddings)]:
        articles = pd.read_parquet(args.store / dataset / "articles.parquet"); ids, vectors = builder(args.source_root, articles.article_id.astype(str).tolist())
        np.savez_compressed(args.store / dataset / "article_embeddings.npz", article_ids=np.asarray(ids), vectors=vectors)
        impressions = pd.read_parquet(args.store / dataset / "validation_impressions.parquet")
        if args.limit: impressions = impressions.head(args.limit)
        cold = impressions[impressions.history_ids.map(len) < 5]
        warm = impressions[impressions.history_ids.map(len) >= 5]
        result = {"dataset": dataset, "embedding_articles": len(ids), "embedding_dimensions": int(vectors.shape[1]), "index": "brute-force cosine ANN baseline", **retrieval_metrics(ids, vectors, impressions), "slices": {"cold_history_under_5": retrieval_metrics(ids, vectors, cold), "warm_history_5_or_more": retrieval_metrics(ids, vectors, warm)}}
        results.append(result); print(result)
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(results, indent=2))

if __name__ == "__main__": main()
