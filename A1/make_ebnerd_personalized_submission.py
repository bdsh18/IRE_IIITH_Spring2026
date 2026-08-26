#!/usr/bin/env python3
"""Large EB-NeRD history-category submission, streamed by parquet row group."""
from __future__ import annotations
import argparse, math, shutil, zipfile
from collections import Counter
from pathlib import Path
import pyarrow.parquet as pq

def extract(archive, member, target):
    if target.exists(): return target
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z, z.open(member) as s, target.open("wb") as d: shutil.copyfileobj(s,d,1024*1024)
    return target

def main():
    p=argparse.ArgumentParser();p.add_argument('--large',type=Path,default=Path('ebnerd_large.zip'));p.add_argument('--testzip',type=Path,default=Path('ebnerd_testset.zip'));p.add_argument('--work',type=Path,default=Path('data/raw/ebnerd_personalized'));p.add_argument('--out',type=Path,default=Path('outputs/ebnerd_personalized_submission'));a=p.parse_args()
    train=extract(a.large,'train/behaviors.parquet',a.work/'train.parquet'); articles=extract(a.large,'articles.parquet',a.work/'articles.parquet'); history=extract(a.testzip,'ebnerd_testset/test/history.parquet',a.work/'history.parquet'); test=extract(a.testzip,'ebnerd_testset/test/behaviors.parquet',a.work/'test.parquet')
    # Compact article-id -> category-code lookup.
    art=pq.read_table(articles,columns=['article_id','category_str','topics','published_time']); labels=sorted(set(x.as_py() or '' for x in art['category_str'])); code={label:i for i,label in enumerate(labels)}
    topic_code={}; category={}; topics={}; published={}
    for article, cat, article_topics, timestamp in zip(art['article_id'],art['category_str'],art['topics'],art['published_time']):
        aid=int(article.as_py()); category[aid]=code[cat.as_py() or '']; encoded=[]
        for topic in article_topics.as_py() or []:
            if topic not in topic_code: topic_code[topic]=len(topic_code)
            encoded.append(topic_code[topic])
        topics[aid]=tuple(encoded); published[aid]=timestamp.as_py()
    pop=Counter(); pf=pq.ParquetFile(train)
    for g in range(pf.num_row_groups):
        for row in pf.read_row_group(g,columns=['article_ids_clicked'])['article_ids_clicked'].to_pylist(): pop.update(row or [])
    maxpop=max(pop.values()); print(f'Popularity: {len(pop):,} articles; categories: {len(code)}')
    # Store last 30 category/topic pairs; this keeps per-user profiles small on the 13.5M-row test set.
    preferences={}; pf=pq.ParquetFile(history)
    for g in range(pf.num_row_groups):
        tab=pf.read_row_group(g,columns=['user_id','article_id_fixed'])
        for user,items in zip(tab['user_id'].to_pylist(),tab['article_id_fixed'].to_pylist()): preferences[int(user)]=tuple(int(x) for x in (items or [])[-30:])
        print(f'  history group {g+1}/{pf.num_row_groups}')
    print(f'Loaded compact histories for {len(preferences):,} users')
    a.out.mkdir(parents=True,exist_ok=True);pred=a.out/'predictions.txt'; n=0; pf=pq.ParquetFile(test)
    with pred.open('w') as o:
        for g in range(pf.num_row_groups):
            tab=pf.read_row_group(g,columns=['impression_id','impression_time','user_id','article_ids_inview'])
            for imp,now,user,candidates in zip(tab['impression_id'].to_pylist(),tab['impression_time'].to_pylist(),tab['user_id'].to_pylist(),tab['article_ids_inview'].to_pylist()):
                category_pref=Counter(); topic_pref=Counter(); hist=preferences.get(int(user),())
                for age,clicked in enumerate(reversed(hist)):
                    w=.88**age; category_pref[category.get(clicked,0)]+=w
                    for topic in topics.get(clicked,()): topic_pref[topic]+=w
                category_max=max(category_pref.values(),default=1); topic_max=max(topic_pref.values(),default=1)
                def score(article):
                    article=int(article); category_score=category_pref[category.get(article,0)]/category_max
                    article_topics=topics.get(article,()); topic_score=sum(topic_pref[t] for t in article_topics)/(topic_max*max(1,len(article_topics)))
                    popularity=.12*math.log1p(pop[article])/math.log1p(maxpop); freshness=0.
                    if published.get(article) is not None: freshness=.08*math.exp(-max(0.,(now-published[article]).total_seconds()/3600)/(24*4))
                    return .52*category_score+.28*topic_score+popularity+freshness
                order=sorted(range(len(candidates)),key=lambda i:(-score(candidates[i]),i));ranks=[0]*len(candidates)
                for r,i in enumerate(order,1):ranks[i]=r
                o.write(f'{imp} [{",".join(map(str,ranks))}]\n');n+=1
            print(f'  wrote {n:,} rows ({g+1}/{pf.num_row_groups})')
    # Validate every output line with the source row order and rank permutation.
    checked=0
    with pred.open() as o:
        for g in range(pf.num_row_groups):
            tab=pf.read_row_group(g,columns=['impression_id','article_ids_inview'])
            for imp,candidates in zip(tab['impression_id'].to_pylist(),tab['article_ids_inview'].to_pylist()):
                ident,text=o.readline().rstrip('\n').split(' ',1); ranks=[int(x) for x in text[1:-1].split(',')]
                if ident!=str(imp) or sorted(ranks)!=list(range(1,len(candidates)+1)): raise ValueError(f'Invalid row {imp}')
                checked+=1
        if o.readline(): raise ValueError('Extra predictions')
    bundle=a.out/'ebnerd_personalized_submission.zip'
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as z:z.write(pred,arcname='predictions.txt')
    print(f'Ready: {bundle}; validated {checked:,} rows')
if __name__=='__main__':main()
