#!/usr/bin/env python3
"""Validate EB-NeRD category-history ranking against popularity on demo data."""
from collections import Counter
import math
import pandas as pd
from mind_personalized_ranker import metrics

ROOT="data/raw/ebnerd_demo"
articles=pd.read_parquet(f"{ROOT}/articles.parquet")
category=dict(zip(articles.article_id.astype(int),articles.category_str.fillna("")))
train=pd.read_parquet(f"{ROOT}/train/behaviors.parquet"); hist=pd.read_parquet(f"{ROOT}/validation/history.parquet"); validation=pd.read_parquet(f"{ROOT}/validation/behaviors.parquet")
pop=Counter(int(a) for row in train.article_ids_clicked for a in row); maxpop=max(pop.values())
history=dict(zip(hist.user_id.astype(int),hist.article_id_fixed))
def rank(candidates, past, personal):
    preferences=Counter()
    for age,a in enumerate(reversed(list(past)[-20:])): preferences[category.get(int(a),"")]+=.85**age
    maximum=max(preferences.values(),default=1)
    def score(a): return .15*math.log1p(pop[int(a)])/math.log1p(maxpop)+(preferences[category.get(int(a),"")]/maximum if personal else 0)
    order=sorted(range(len(candidates)),key=lambda i:(-score(candidates[i]),i));out=[0]*len(candidates)
    for r,i in enumerate(order,1):out[i]=r
    return out
totals={False:[0.]*4,True:[0.]*4}
for row in validation.itertuples(index=False):
    past=history.get(int(row.user_id),[]); labels=[int(a in set(row.article_ids_clicked)) for a in row.article_ids_inview]
    for personal in [False,True]:
        result=metrics(rank(row.article_ids_inview,past,personal),labels); totals[personal]=[a+b for a,b in zip(totals[personal],result)]
for personal,name in [(False,"popularity"),(True,"history_category")]:print(name,{key:round(value/len(validation),6) for key,value in zip(["auc","mrr","ndcg@5","ndcg@10"],totals[personal])})
