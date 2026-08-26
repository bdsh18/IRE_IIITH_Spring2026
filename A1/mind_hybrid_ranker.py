#!/usr/bin/env python3
"""Hybrid MIND ranker: category affinity + entity cosine + popularity."""
from __future__ import annotations
import argparse, json, math, zipfile
from collections import Counter
from pathlib import Path
import numpy as np
from mind_personalized_ranker import member, popularity, metrics

def entity_ids(raw):
    try: values = json.loads(raw) if raw else []
    except Exception: return []
    return [x.get("WikidataId") for x in values if x.get("WikidataId")]

def entity_vectors(train):
    vectors = {}
    with zipfile.ZipFile(train) as z, z.open(member(train, "/entity_embedding.vec")) as f:
        for line in f:
            items=line.decode().rstrip().split("\t"); vectors[items[0]]=np.asarray(items[1:],dtype=np.float32)
    return vectors

def article_data(archives, entity):
    data={}
    for archive in archives:
        with zipfile.ZipFile(archive) as z, z.open(member(archive,"/news.tsv")) as f:
            for line in f:
                x=line.decode("utf-8").rstrip("\n").split("\t")
                if len(x)<8: continue
                ids=entity_ids(x[6])+entity_ids(x[7]); available=[entity[i] for i in ids if i in entity]
                v=np.mean(available,axis=0) if available else None
                if v is not None: v=v/max(float(np.linalg.norm(v)),1e-12)
                data[x[0]]=(x[1],x[2],v)
    return data

def rank(candidates, history, data, pop, hybrid=True):
    cats,subs=Counter(),Counter(); vectors=[]
    for age,article in enumerate(reversed(history[-20:])):
        cat,sub,v=data.get(article,("","",None)); w=.85**age; cats[cat]+=w; subs[sub]+=w
        if v is not None: vectors.append(v*w)
    user=np.sum(vectors,axis=0) if vectors else None
    if user is not None: user=user/max(float(np.linalg.norm(user)),1e-12)
    maxcat=max(cats.values(),default=1); maxsub=max(subs.values(),default=1); maxpop=max(pop.values(),default=1)
    def score(a):
        cat,sub,v=data.get(a,("","",None)); base=cats[cat]/maxcat + 2*subs[sub]/maxsub + .15*math.log1p(pop[a])/math.log1p(maxpop)
        semantic=float(v@user) if hybrid and v is not None and user is not None else 0.
        return base + .25*semantic
    order=sorted(range(len(candidates)),key=lambda i:(-score(candidates[i]),i)); result=[0]*len(candidates)
    for r,i in enumerate(order,1):result[i]=r
    return result

def evaluate(dev,data,pop,limit):
    totals={"category_popularity":[0.]*4,"hybrid":[0.]*4}; n=0
    with zipfile.ZipFile(dev) as z,z.open(member(dev,"/behaviors.tsv")) as f:
        for line in f:
            x=line.decode().rstrip("\n").split("\t"); hist=x[3].split() if x[3] else []; cand=[];labels=[]
            for item in x[4].split():a,_,y=item.rpartition("-");cand.append(a);labels.append(int(y))
            for name,flag in [("category_popularity",False),("hybrid",True)]:
                v=metrics(rank(cand,hist,data,pop,flag),labels);totals[name]=[a+b for a,b in zip(totals[name],v)]
            n+=1
            if limit and n>=limit:break
    return {key:{metric:round(x/n,6) for metric,x in zip(["auc","mrr","ndcg@5","ndcg@10"],values)} for key,values in totals.items()}|{"impressions":n}

def submit(test,data,pop,out,limit):
    n=0
    with zipfile.ZipFile(test) as z,z.open(member(test,"/behaviors.tsv")) as f,out.open("w") as w:
        for line in f:
            x=line.decode().rstrip("\n").split("\t");hist=x[3].split() if x[3] else [];w.write(f"{x[0]} [{','.join(map(str,rank(x[4].split(),hist,data,pop,True)))}]\n");n+=1
            if n%100000==0:print(f"  wrote {n:,}")
            if limit and n>=limit:break
    return n

def main():
    p=argparse.ArgumentParser();p.add_argument("--train",type=Path,default=Path("MIND_data/MINDlarge_train.zip"));p.add_argument("--dev",type=Path,default=Path("MIND_data/MINDlarge_dev.zip"));p.add_argument("--test",type=Path,default=Path("MIND_data/MINDlarge_test.zip"));p.add_argument("--limit",type=int,default=0);p.add_argument("--evaluate-only",action="store_true");p.add_argument("--output-dir",type=Path,default=Path("outputs/mind_hybrid_submission"));a=p.parse_args()
    print("Loading entity vectors and article metadata...");data=article_data([a.train,a.dev,a.test],entity_vectors(a.train));print(f"Articles: {len(data):,}")
    pop=popularity(a.train);result=evaluate(a.dev,data,pop,a.limit);print(result)
    if a.evaluate_only:return
    a.output_dir.mkdir(parents=True,exist_ok=True);pred=a.output_dir/"prediction.txt";n=submit(a.test,data,pop,pred,a.limit)
    with zipfile.ZipFile(a.output_dir/"mind_hybrid_submission.zip","w",zipfile.ZIP_DEFLATED) as z:z.write(pred,arcname="prediction.txt")
    print(f"Ready: {n:,}")
if __name__=="__main__":main()
