# Assignment 2 - Learning from Click-Logs

Team Name: Nexus
Team Member 1: Sriharini Margapuri (2026701030)
Team Member 2: Bidisha Shaw (2022900023)

This project implements the Assignment 2 news recommendation pipeline for both
MIND and EB-NeRD. It extends the Assignment 1 candidate generators with
click-history and session features, a LightGBM re-ranker, ablation analysis,
serving/scale measurements, and extended offline evaluation.

## Setup

From this directory, install the Python dependencies:

```bash
pip install -r requirements.txt
```

The default development run expects these small/demo archives:

```text
MIND_data/MINDsmall_train.zip
MIND_data/MINDsmall_dev.zip
ebnerd_demo.zip
```

Large archives are needed for Codabench submissions, but should not be committed
to Git. The submission defaults are:

```text
MIND_data/MINDlarge_train.zip
MIND_data/MINDlarge_test.zip
ebnerd_large.zip
ebnerd_testset.zip
```

## Run the full Assignment 2 pipeline

Run all development stages in dependency order:

```bash
./run_a2_pipeline.sh --skip-install
```

For a quick smoke run, limit the number of impressions used by the re-ranker and
ablation stages:

```bash
./run_a2_pipeline.sh --skip-install --smoke-limit 500
```

The script performs the following stages for both datasets:

1. Builds a chronological train/validation/test feature store.
2. Runs the behavioural-window leakage test.
3. Generates BM25 and semantic candidates from the Assignment 1 pipeline.
4. Builds Q1 click-history, recency, popularity, freshness, and session features.
5. Trains and evaluates the Q2 two-stage re-ranker.
6. Runs the Q3 baseline/improvement ablation with paired bootstrap intervals.
7. Measures Q4 index memory, p99 latency, and target-SLA scaling estimates.
8. Runs Q5 metrics, cold/warm and head/tail slices, and bootstrap confidence intervals.

The processed feature store is written to `data/processed/`. To use another
location, set `STORE`:

```bash
STORE=/path/to/processed ./run_a2_pipeline.sh --skip-install
```

## Outputs

Results are written under `outputs/`:

```text
q2_reranker_metrics_mind.json
q2_reranker_metrics_ebnerd.json
q3_ablation_mind.json
q3_ablation_ebnerd.json
q4_serving_scale_mind.json
q4_serving_scale_ebnerd.json
q5_evaluation.json
q9_serving_availability_mind.json
q9_serving_availability_ebnerd.json
```

The evaluation reports AUC, MRR, nDCG@5, nDCG@10, diversity, novelty,
coverage, dataset/user slices, and confidence intervals where applicable.

## Codabench submissions

After validating the development pipeline and downloading the large archives,
build both submission ZIPs with:

```bash
./run_a2_pipeline.sh --skip-install --submissions
```

Override archive locations when necessary:

```bash
MIND_TRAIN=/path/MINDlarge_train.zip \
MIND_TEST=/path/MINDlarge_test.zip \
EBNERD_LARGE=/path/ebnerd_large.zip \
EBNERD_TEST=/path/ebnerd_testset.zip \
./run_a2_pipeline.sh --skip-install --submissions
```

The generated ZIP files are placed in `outputs/` for upload to the MIND and
RecSys 2024 Codabench competitions. Register and submit to both leaderboards:

- MIND: https://www.codabench.org/competitions/13967/
- RecSys 2024 Challenge: https://www.codabench.org/competitions/2469/

## Leakage test

Run the behavioural-window boundary test independently with:

```bash
python -m pytest test_no_leakage.py -v
```

This verifies that future clicks are not used when constructing features for a
training or serving impression.

## Assignment 2 deliverables

- Reproducible code and one-command pipeline.
- Baseline and improved re-ranker results with an ablation and paired bootstrap CI.
- Serving and 10x scale analysis.
- Extended evaluation with required metrics and slices.
- MIND and EB-NeRD Codabench prediction files and leaderboard screenshots.
- Six-page design note/report and AI usage log.

Large datasets, generated outputs, model checkpoints, and Python cache files are
excluded by `.gitignore`.
