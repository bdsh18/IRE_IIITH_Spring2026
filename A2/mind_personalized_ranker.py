#!/usr/bin/env python3
"""Personalized MIND ranker: history category/subcategory affinity + popularity."""
from __future__ import annotations
import argparse, math, zipfile
from collections import Counter
from pathlib import Path
from tqdm.auto import tqdm

def member(archive: Path, suffix: str) -> str:
    with zipfile.ZipFile(archive) as z:
        return next(n for n in z.namelist() if n.endswith(suffix))

def metadata(archives: list[Path]) -> dict[str, tuple[str, str]]:
    result = {}
    for archive in archives:
        with zipfile.ZipFile(archive) as z, z.open(member(archive, "/news.tsv")) as f:
            for line in f:
                x = line.decode("utf-8").rstrip("\n").split("\t")
                if len(x) >= 3: result[x[0]] = (x[1], x[2])
    return result

def popularity(train: Path) -> Counter:
    out = Counter()
    with zipfile.ZipFile(train) as z, z.open(member(train, "/behaviors.tsv")) as f:
        for line in tqdm(f, desc="MIND evaluation", unit="impression"):
            x = line.decode("utf-8").rstrip("\n").split("\t")
            if len(x) == 5:
                for item in x[4].split():
                    article, sep, label = item.rpartition("-")
                    if sep and label == "1": out[article] += 1
    return out

def ranks(candidates, history, meta, pop, personalized=True):
    cats, subs = Counter(), Counter()
    for age, article in enumerate(reversed(history[-20:])):
        category, sub = meta.get(article, ("", "")); weight = .85 ** age
        cats[category] += weight; subs[sub] += weight
    max_pop = max(pop.values(), default=1)
    def score(article):
        p = .15 * math.log1p(pop[article]) / math.log1p(max_pop)
        if not personalized: return p
        category, sub = meta.get(article, ("", ""))
        return 1.0 * cats[category] + 2.0 * subs[sub] + p
    order = sorted(range(len(candidates)), key=lambda i: (-score(candidates[i]), i)); output = [0] * len(candidates)
    for rank, pos in enumerate(order, 1): output[pos] = rank
    return output

def metrics(rank_list, labels):
    order = sorted(range(len(rank_list)), key=lambda i: rank_list[i]); positives = [i for i, y in enumerate(labels) if y]
    p_scores = [rank_list[i] for i in positives]; n_scores = [rank_list[i] for i, y in enumerate(labels) if not y]
    auc = sum(1 if p < n else .5 if p == n else 0 for p in p_scores for n in n_scores) / max(len(p_scores) * len(n_scores), 1)
    mrr = next((1 / (i + 1) for i, pos in enumerate(order) if labels[pos]), 0.0)
    def ndcg(k):
        dcg = sum(1 / math.log2(i + 2) for i, pos in enumerate(order[:k]) if labels[pos]); ideal = sum(1 / math.log2(i + 2) for i in range(min(k, sum(labels))))
        return dcg / ideal if ideal else 0.0
    return auc, mrr, ndcg(5), ndcg(10)

def evaluate(dev, meta, pop, limit):
    totals = {"popularity": [0., 0., 0., 0.], "personalized": [0., 0., 0., 0.]}; n = 0
    with zipfile.ZipFile(dev) as z, z.open(member(dev, "/behaviors.tsv")) as f:
        for line in tqdm(f, desc="MIND submission", unit="impression"):
            x = line.decode("utf-8").rstrip("\n").split("\t")
            if len(x) != 5: continue
            history = x[3].split() if x[3] else []; candidates, labels = [], []
            for item in x[4].split():
                a, _, y = item.rpartition("-"); candidates.append(a); labels.append(int(y))
            for name, flag in [("popularity", False), ("personalized", True)]:
                values = metrics(ranks(candidates, history, meta, pop, flag), labels); totals[name] = [a + b for a, b in zip(totals[name], values)]
            n += 1
            if limit and n >= limit: break
    return {name: {key: round(value / n, 6) for key, value in zip(["auc", "mrr", "ndcg@5", "ndcg@10"], values)} for name, values in totals.items()} | {"impressions": n}

def submit(test, meta, pop, output, limit):
    count = 0
    with zipfile.ZipFile(test) as z, z.open(member(test, "/behaviors.tsv")) as f, output.open("w") as o:
        for line in f:
            x = line.decode("utf-8").rstrip("\n").split("\t"); history = x[3].split() if x[3] else []; candidates = x[4].split()
            o.write(f"{x[0]} [{','.join(map(str, ranks(candidates, history, meta, pop, True)))}]\n"); count += 1
            if count % 100000 == 0: print(f"  wrote {count:,}")
            if limit and count >= limit: break
    return count

def main():
    p = argparse.ArgumentParser(); p.add_argument("--train", type=Path, default=Path("MIND_data/MINDlarge_train.zip")); p.add_argument("--dev", type=Path, default=Path("MIND_data/MINDlarge_dev.zip")); p.add_argument("--test", type=Path, default=Path("MIND_data/MINDlarge_test.zip")); p.add_argument("--evaluate-only", action="store_true"); p.add_argument("--limit", type=int, default=0); p.add_argument("--output-dir", type=Path, default=Path("outputs/mind_personalized_submission")); a=p.parse_args()
    print("Loading article categories..."); meta=metadata([a.train,a.dev,a.test]); print(f"Articles: {len(meta):,}")
    print("Counting train clicks..."); pop=popularity(a.train); print(f"Clicked articles: {len(pop):,}")
    result=evaluate(a.dev, meta, pop, a.limit); print(result)
    if a.evaluate_only: return
    a.output_dir.mkdir(parents=True, exist_ok=True); prediction=a.output_dir/"prediction.txt"; total=submit(a.test,meta,pop,prediction,a.limit)
    with zipfile.ZipFile(a.output_dir/"mind_personalized_submission.zip","w",zipfile.ZIP_DEFLATED) as z:z.write(prediction,arcname="prediction.txt")
    print(f"Ready: {total:,} predictions")
if __name__ == "__main__": main()
