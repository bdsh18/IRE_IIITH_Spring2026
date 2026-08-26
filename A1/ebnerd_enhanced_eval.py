#!/usr/bin/env python3
"""Compare the enhanced EB-NeRD history/topic/freshness ranker on demo validation."""
from collections import Counter
from datetime import timezone
import math
import pandas as pd
from mind_personalized_ranker import metrics

ROOT = "data/raw/ebnerd_demo"
articles = pd.read_parquet(f"{ROOT}/articles.parquet", columns=["article_id", "category_str", "topics", "published_time"])
meta = {
    int(r.article_id): (r.category_str if isinstance(r.category_str, str) else "", tuple(r.topics.tolist()) if hasattr(r.topics, "tolist") else tuple(r.topics or ()), r.published_time)
    for r in articles.itertuples(index=False)
}
train = pd.read_parquet(f"{ROOT}/train/behaviors.parquet")
hist = pd.read_parquet(f"{ROOT}/validation/history.parquet", columns=["user_id", "article_id_fixed"])
validation = pd.read_parquet(f"{ROOT}/validation/behaviors.parquet")
pop = Counter(int(a) for row in train.article_ids_clicked for a in row)
maxpop = max(pop.values())
history = dict(zip(hist.user_id.astype(int), hist.article_id_fixed))

def ranks(candidates, past, now):
    cats, topics = Counter(), Counter()
    for age, article in enumerate(reversed(list(past)[-30:])):
        cat, article_topics, _ = meta.get(int(article), ("", (), None))
        w = .88 ** age
        cats[cat] += w
        for topic in article_topics: topics[topic] += w
    cm = max(cats.values(), default=1.0)
    tm = max(topics.values(), default=1.0)
    def score(article):
        cat, article_topics, published = meta.get(int(article), ("", (), None))
        category = cats[cat] / cm
        topic = sum(topics[t] for t in article_topics) / (tm * max(1, len(article_topics)))
        popularity = math.log1p(pop[int(article)]) / math.log1p(maxpop)
        freshness = 0.0
        if published is not None:
            hours = max(0.0, (now - published).total_seconds() / 3600)
            freshness = math.exp(-hours / (24 * 4))
        return .52 * category + .28 * topic + .12 * popularity + .08 * freshness
    order = sorted(range(len(candidates)), key=lambda i: (-score(candidates[i]), i))
    out = [0] * len(candidates)
    for rank, i in enumerate(order, 1): out[i] = rank
    return out

totals = [0.] * 4
for row in validation.itertuples(index=False):
    labels = [int(a in set(row.article_ids_clicked)) for a in row.article_ids_inview]
    result = metrics(ranks(row.article_ids_inview, history.get(int(row.user_id), []), row.impression_time), labels)
    totals = [x + y for x, y in zip(totals, result)]
print({key: round(value / len(validation), 6) for key, value in zip(["auc", "mrr", "ndcg@5", "ndcg@10"], totals)})
