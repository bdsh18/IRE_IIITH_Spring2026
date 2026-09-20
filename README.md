# IRE IIITH Spring 2026

This `sbidisha_A2` branch contains the complete Assignment 2 implementation, reproducible experiment instructions, small tracked metrics, and the final design note. Large datasets, extracted stores, virtual environments, model caches, and Codabench prediction archives are intentionally ignored by Git.

The report with local validation results and large-test Codabench evidence is available at [`A2/outputs/Design_Note_A2.pdf`](A2/outputs/Design_Note_A2.pdf). The completed MIND large-test submission recorded AUC **0.6290**, MRR **0.3071**, nDCG@5 **0.3295**, and nDCG@10 **0.3860**. The EB-NeRD large-test submission is recorded as submitted; its numeric score was pending when the report was generated.

## Assignment 2 — Learning from Click-Logs

This `sbidisha_A2` branch contains Assignment 2 only. It builds the retrieval components needed for MIND and EB-NeRD, then trains an in-view behavioural re-ranker with a separately measured catalog-retrieval stage. Assignment 1 source is intentionally not part of this branch.

### One-command reproduction

Run the commands below from `A2/`. Set `SOURCE_ROOT` to the external directory holding the downloaded archives; it is deliberately outside this Git branch:

```bash
cd A2
python3 -m venv .venv
source .venv/bin/activate
export SOURCE_ROOT=/absolute/path/to/news-datasets
./run_a2_pipeline.sh
```

For a quick verification without generating large submissions:

```bash
SOURCE_ROOT="$SOURCE_ROOT" ./run_a2_pipeline.sh --smoke-limit 500
```

### A2 Q1 — Click-history, session, and article features

`features.py` produces candidate-level features from strictly earlier behaviour: recency-weighted category/topic and embedding affinity, causal popularity, freshness, candidate position, session prior impressions/clicks, and session mean dwell time. MIND has no session ID or dwell time; those session-specific features are set to zero rather than treating a user's whole history as one session.

### A2 Q2 — Two-stage retrieve-then-rank

Stage 1 uses BM25 lexical retrieval over `title + abstract` to retrieve top-200 articles from the complete catalog and reports Recall@50/100/200 plus shown-candidate coverage. Stage 2 uses LightGBM LambdaRank with lexical, semantic, behavioural, session, and article features. The default pipeline records both (a) normal **in-view** re-ranking, which matches Codabench's requirement to assign a rank to every supplied candidate, and (b) a clearly labelled strict diagnostic that gives only catalog-top-200 candidates a stage-2 score. It never presents the in-view metric as an open-catalog result.

### A2 Q3 — Baseline, improvement, and ablation

The built-in baseline is an **internal causal temporal-popularity baseline**, not an official repository baseline. The principled improvement is category-affinity from the recency-weighted user history; `ablation.py` trains a model without that feature and compares it with the full model. Every comparison uses paired bootstrap 95% CIs; only a CI excluding zero may be claimed as a gain. To compare an actual upstream starter implementation, supply its predictions on this exact validation split with `--starter-predictions PATH`; the parser rejects mismatched IDs, row counts, and non-permutation ranks.

### A2 Q4 — Serving and scale

`serving_scale.py` measures BM25 index size, FAISS HNSW ANN serialized size, feature-store size, and end-to-end p50/p95/p99 latency for retrieval plus re-ranking. It uses the same serving-available columns as the submission scripts, estimates QPS/cost at p99 <100 ms, and records the 10x scaling risks. Run it after the re-ranker:

```bash
python3 serving_scale.py --dataset mind --target-sla-ms 100
python3 serving_scale.py --dataset ebnerd --target-sla-ms 100
```

### A2 Q5 — Extended evaluation and submission

`offline_evaluation.py` evaluates BM25, semantic, and the behavioural re-ranker. It reports AUC, MRR, nDCG@5, nDCG@10, diversity, novelty, coverage, cold/warm and head/tail slices, and bootstrap 95% CIs. The production command uses only columns available in the test-set submission generators, and adds a separate strict catalog-top-200-gated diagnostic alongside normal in-view results.

```bash
python3 offline_evaluation.py --serving-only --strict-two-stage --output outputs/q5_evaluation.json
```

After verifying these outputs, generate and submit the two final Codabench ZIP files as described in [Final Codabench submissions](#final-codabench-submissions).  The final submission commands deliberately do **not** rerun the full feature/evaluation pipeline.

### Data preparation

The raw data directory referenced by `SOURCE_ROOT` must contain:

```text
MIND_data/MINDsmall_train.zip
MIND_data/MINDsmall_dev.zip
MIND_data/MINDlarge_train.zip
MIND_data/MINDlarge_dev.zip
MIND_data/MINDlarge_test.zip
ebnerd_demo.zip
ebnerd_large.zip
ebnerd_testset.zip
Ekstra_Bladet_word2vec.zip
```

Build the small development feature stores with one command:

```bash
python3 build_pipeline.py --source-root "$SOURCE_ROOT" --output data/processed
```

It extracts the MIND-small and EB-NeRD demo archives and writes:

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

The interaction split is chronological: oldest data is train, then validation, and newest data is test. No random interaction split is used.

### Q2 - BM25 lexical candidate generation

```bash
python3 bm25_retrieval.py --store data/processed
```

This command builds one inverted BM25 index per dataset over `title + abstract`. For each labeled validation impression, it joins the titles of the user's five most recent clicked articles into a query, retrieves from the complete article catalog, and writes average Recall@50, Recall@100, and Recall@200 to `outputs/q2_bm25_metrics.json`.

### Q3 - Semantic candidate generation

```bash
python3 semantic_retrieval.py --source-root "$SOURCE_ROOT" --store data/processed
```

MIND article vectors are mean-pooled from the supplied 100-dimensional entity embeddings. EB-NeRD article vectors come from the supplied Word2Vec `document_vector.parquet`. A user vector is the mean of their clicked-article vectors; cosine similarity retrieves the nearest 50/100/200 articles. The command writes compressed article vectors into each feature store and reports overall plus cold/warm-user validation recall in `outputs/q3_semantic_metrics.json`. Brute-force cosine search is intentionally used for the small/demo data; replace it with FAISS for the large bundles.

### Q4 - Offline evaluation harness

```bash
python3 offline_evaluation.py --store data/processed --serving-only --strict-two-stage --output outputs/q5_evaluation.json
```

The harness compares BM25 and semantic candidate ranking on every labeled validation impression. It reports AUC, MRR, nDCG@5, nDCG@10, intra-list diversity@10, novelty@10, coverage@10, cold/warm user slices, and 95% bootstrap confidence intervals. Results are written to `outputs/q5_evaluation.json`.

### Final Codabench submissions

Run these from the `A2/` directory after the small processed feature store has been built.  They train on the cached small-scale A2 features, but rank the **large** competition test files.  Do not pass `--limit` for a real submission.

The optimized generators score only the articles shown in each impression, precompute user-side feature context once per impression (and reuse fixed EB-NeRD user histories through a bounded cache), checkpoint every 5,000 impressions, and validate the complete output before creating a ZIP.  An incomplete run remains as `*.partial`; it is never zipped.

```bash
source .venv/bin/activate
export SOURCE_ROOT=/absolute/path/to/news-datasets
```

To generate both ZIPs sequentially using the explicit external raw-data root, use:

```bash
./run_a2_pipeline.sh --submissions-only
```

The explicit commands below are useful when you want to run or resume one dataset independently.

#### MIND large-test submission

```bash
python3 make_mind_submission.py \
  --train "$SOURCE_ROOT/MIND_data/MINDlarge_train.zip" \
  --dev "$SOURCE_ROOT/MIND_data/MINDlarge_dev.zip" \
  --test "$SOURCE_ROOT/MIND_data/MINDlarge_test.zip" \
  --store data/processed \
  --output-dir outputs/mind_submission \
  --batch-size 5000
```

The completed upload file is `outputs/mind_submission/mind_submission.zip`; it contains only the root-level file `prediction.txt`.

#### EB-NeRD large-test submission

```bash
python3 make_ebnerd_personalized_submission.py \
  --large "$SOURCE_ROOT/ebnerd_large.zip" \
  --testzip "$SOURCE_ROOT/ebnerd_testset.zip" \
  --source-root "$SOURCE_ROOT" \
  --store data/processed \
  --work data/raw/ebnerd_personalized \
  --out outputs/ebnerd_submission \
  --batch-size 5000
```

The completed upload file is `outputs/ebnerd_submission/ebnerd_personalized_submission.zip`; it contains only the root-level file `predictions.txt`.

#### If a run is interrupted

Repeat the exact same command and append `--resume`.  The generator verifies the archive, model/cache configuration, code fingerprint, and completed prefix before continuing.  If any of these checks differ, start a fresh run instead of trusting the old partial file.

For either completed ZIP, inspect it before upload:

```bash
unzip -Z1 outputs/mind_submission/mind_submission.zip
unzip -Z1 outputs/ebnerd_submission/ebnerd_personalized_submission.zip
```

### Optional large-data v2 experiments

The baseline generators above are preserved for reproducibility.  The separate
v2 scripts train on the labelled **large** bundles rather than applying a
MIND-small/demo model to the competition test set.  They are intended for a
score-improvement experiment and never overwrite the baseline ZIPs.

For MIND, v2 uses deterministic whole-impression sampling, LambdaRank,
subcategory affinity, entity overlap, mean/max entity-vector similarity,
content-token overlap (including IDF-weighted overlap), and time-safe train-click popularity.  Its validation
is the official later `MINDlarge_dev` archive.  Run the commands in order and
make a final `train + dev` model only if the dev result improves:

```bash
python3 mind_large_v2.py train \
  --train "$SOURCE_ROOT/MIND_data/MINDlarge_train.zip" \
  --dev "$SOURCE_ROOT/MIND_data/MINDlarge_dev.zip" \
  --sample-modulus 10 --negatives-per-positive 20
python3 mind_large_v2.py evaluate \
  --train "$SOURCE_ROOT/MIND_data/MINDlarge_train.zip" \
  --dev "$SOURCE_ROOT/MIND_data/MINDlarge_dev.zip" \
  --model data/v2/mind_large/mind_v2.pkl
# Inspect outputs/mind_v2_validation.json before continuing.
python3 mind_large_v2.py train \
  --train "$SOURCE_ROOT/MIND_data/MINDlarge_train.zip" \
  --dev "$SOURCE_ROOT/MIND_data/MINDlarge_dev.zip" \
  --include-dev-in-train --sample-modulus 10 --negatives-per-positive 20 \
  --model data/v2/mind_large/mind_v2_final_mod10.pkl
python3 mind_large_v2.py submit \
  --train "$SOURCE_ROOT/MIND_data/MINDlarge_train.zip" \
  --dev "$SOURCE_ROOT/MIND_data/MINDlarge_dev.zip" \
  --test "$SOURCE_ROOT/MIND_data/MINDlarge_test.zip" \
  --model data/v2/mind_large/mind_v2_final_mod10.pkl \
  --output-dir outputs/mind_v2_submission_mod10 --batch-impressions 4000 --resume
```

The final MIND upload is
`outputs/mind_v2_submission_mod10/mind_v2_submission.zip`; it contains only
`prediction.txt`.  Use `--limit 1000` only for a smoke ZIP, and append
`--resume` to continue an interrupted matching v2 test run.  A fresh empty
output directory is also safe with `--resume`.

The selected MIND configuration achieved this temporal `MINDlarge_dev`
measurement before final training on train+dev: AUC **0.6205**, MRR **0.3471**,
nDCG@5 **0.3281**, and nDCG@10 **0.3869** (224,597 deterministically sampled
impressions).  This is an offline selection result, not a promise of a
Codabench score.

For EB-NeRD, v2 trains a sampled large-bundle LambdaRank model and holds out
the supplied later validation partition.  It uses article freshness, category/
subcategory/topic/entity history affinity, provided document-vector
similarity, bounded BM25, causal hourly popularity, and only serving-available
context fields.  It deliberately excludes `read_time`, scrolling, next-event,
and click-label fields from test features.

```bash
python3 ebnerd_large_v2.py --mode all \
  --large "$SOURCE_ROOT/ebnerd_large.zip" \
  --embeddings "$SOURCE_ROOT/Ekstra_Bladet_word2vec.zip"
# Inspect data/v2/ebnerd_large/validation_metrics.json before continuing.
python3 ebnerd_large_v2.py --mode submit \
  --large "$SOURCE_ROOT/ebnerd_large.zip" \
  --testzip "$SOURCE_ROOT/ebnerd_testset.zip" \
  --embeddings "$SOURCE_ROOT/Ekstra_Bladet_word2vec.zip" \
  --prediction-batch-size 6000
```

The final EB-NeRD upload is
`outputs/ebnerd_v2_submission/accelerate_safe/final/ebnerd_v2_submission.zip`;
it contains only `predictions.txt`.  The `accelerate_safe` run uses a stable
semantic maximum calculation on macOS/Apple Accelerate, avoiding spurious
NumPy matrix-vector warnings without changing the model schema. Both v2 output
folders and all v2 caches are ignored by Git because they are large local
artifacts.

The sampled temporal EB-NeRD validation run measured AUC **0.6811**, MRR
**0.4461**, nDCG@5 **0.5015**, and nDCG@10 **0.5546**, compared with BM25-only
AUC **0.4940** on the same 629,087 validation impressions.  As with MIND,
hidden Codabench results can differ from this local validation result.
