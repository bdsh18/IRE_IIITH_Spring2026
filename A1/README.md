# Assignment 1 - Q1 Data Pipeline

Build the small development feature stores with one command:

```bash
make data
```

It extracts only the already-downloaded `MINDsmall_train.zip`, `MINDsmall_dev.zip`, and `ebnerd_demo.zip`, then writes:

```text
data/processed/
  manifest.json
  mind/articles.parquet
  mind/train_impressions.parquet
  mind/validation_impressions.parquet
  mind/test_impressions.parquet
  ebnerd/articles.parquet
  ebnerd/train_impressions.parquet
  ebnerd/validation_impressions.parquet
  ebnerd/test_impressions.parquet
```

The time split is chronological: oldest data is train, then validation, and newest data is test. No random interaction split is used.

## Q2 - BM25 lexical candidate generation

```bash
make q2-bm25
```

This command builds one inverted BM25 index per dataset over `title + abstract`. For each labeled validation impression, it joins the titles of the user's five most recent clicked articles into a query, retrieves from the complete article catalog, and writes average Recall@50, Recall@100, and Recall@200 to `outputs/q2_bm25_metrics.json`.

## Q3 - Semantic candidate generation

```bash
make q3-semantic
```

MIND article vectors are mean-pooled from the supplied 100-dimensional entity embeddings. EB-NeRD article vectors come from the supplied Word2Vec `document_vector.parquet`. A user vector is the mean of their clicked-article vectors; cosine similarity retrieves the nearest 50/100/200 articles. The command writes compressed article vectors into each feature store and reports overall plus cold/warm-user validation recall in `outputs/q3_semantic_metrics.json`. Brute-force cosine search is intentionally used for the small/demo data; replace it with FAISS for the large bundles.

## Q4 - Offline evaluation harness

```bash
make q4-eval
```

The harness compares BM25 and semantic candidate ranking on every labeled validation impression. It reports AUC, MRR, nDCG@5, nDCG@10, intra-list diversity@10, novelty@10, coverage@10, cold/warm user slices, and 95% bootstrap confidence intervals. Results are written to `outputs/q4_evaluation.json`.

## MIND large-test submission file

```bash
make mind-submission
```

This streams `MINDlarge_train.zip` and `MINDlarge_test.zip` directly from `MIND_data/`, builds a click-popularity baseline from train clicks, writes exactly one candidate-rank line per test impression, validates every line, and creates `outputs/mind_submission/mind_submission.zip`. It does not extract the multi-gigabyte archives into memory. The ZIP contains `prediction.txt`.

## EB-NeRD large-test submission file

```bash
make ebnerd-submission
```

This extracts only the required train/test behavior parquet files, processes them row group by row group, validates every rank list against the original test order, and creates `outputs/ebnerd_submission/ebnerd_submission.zip`. The ZIP contains only `predictions.txt`, exactly as required by EB-NeRD Codabench.
