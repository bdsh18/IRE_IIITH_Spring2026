#!/usr/bin/env bash
# Runs the full Assignment 2 pipeline end to end, in dependency order.
#
# Usage:
#   ./run_a2_pipeline.sh                  # full run, both datasets, no submissions
#   ./run_a2_pipeline.sh --submissions     # also builds the two Codabench ZIPs
#   ./run_a2_pipeline.sh --skip-install    # skip the pip install step
#   ./run_a2_pipeline.sh --smoke-limit 500 # cap reranker/ablation on N impressions
#
# Env vars (override if your paths differ from the assignment defaults):
#   SOURCE_ROOT   raw-data root passed to build_pipeline.py   (default: .)
#   STORE         processed feature store                     (default: data/processed)
#   MIND_TRAIN    MINDlarge_train.zip, for the submission step (default: MIND_data/MINDlarge_train.zip)
#   MIND_TEST     MINDlarge_test.zip,  for the submission step (default: MIND_data/MINDlarge_test.zip)
#   EBNERD_LARGE  ebnerd_large.zip,    for the submission step (default: ebnerd_large.zip)
#   EBNERD_TEST   ebnerd_testset.zip,  for the submission step (default: ebnerd_testset.zip)

set -euo pipefail

SOURCE_ROOT="${SOURCE_ROOT:-.}"
STORE="${STORE:-data/processed}"
MIND_TRAIN="${MIND_TRAIN:-MIND_data/MINDlarge_train.zip}"
MIND_TEST="${MIND_TEST:-MIND_data/MINDlarge_test.zip}"
EBNERD_LARGE="${EBNERD_LARGE:-ebnerd_large.zip}"
EBNERD_TEST="${EBNERD_TEST:-ebnerd_testset.zip}"

RUN_SUBMISSIONS=false
SKIP_INSTALL=false
SMOKE_LIMIT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --submissions)  RUN_SUBMISSIONS=true; shift ;;
    --skip-install) SKIP_INSTALL=true; shift ;;
    --smoke-limit)  SMOKE_LIMIT="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

step() { echo; echo "=== $1 ==="; }

if [[ "$SKIP_INSTALL" == false ]]; then
  step "0. Installing dependencies"
  pip install lightgbm pyarrow --break-system-packages -q
fi

step "1. Rebuilding the feature store (build_pipeline.py)"
python build_pipeline.py --source-root "$SOURCE_ROOT" --output "$STORE"

step "2. Leakage / behavioural-window boundary test (Q9)"
python -m pytest test_no_leakage.py -v

step "3. Candidate generation — BM25 recall check (A1 / Q2 stage 1)"
python bm25_retrieval.py --store "$STORE"

step "4. Candidate generation — semantic embeddings (A1 / Q2 stage 1)"
python semantic_retrieval.py --source-root "$SOURCE_ROOT" --store "$STORE"

for DATASET in mind ebnerd; do
  step "5. Q1 features — $DATASET"
  python features.py --dataset "$DATASET" --split validation --store "$STORE"

  step "6. Q2 re-ranker — $DATASET"
  if [[ "$SMOKE_LIMIT" -gt 0 ]]; then
    python reranker.py --dataset "$DATASET" --store "$STORE" --limit "$SMOKE_LIMIT"
  else
    python reranker.py --dataset "$DATASET" --store "$STORE"
  fi

  step "7. Q3 ablation (paired bootstrap) — $DATASET"
  python ablation.py --dataset "$DATASET" --store "$STORE" --limit "$SMOKE_LIMIT"

  step "8. Q4 serving & scale — $DATASET"
  python serving_scale.py --dataset "$DATASET" --store "$STORE" --target-sla-ms 100

  step "9. Q9 serving-availability ablation — $DATASET"
  python serving_availability_ablation.py --dataset "$DATASET" --store "$STORE" --limit "$SMOKE_LIMIT"
done

step "10. Q5 extended evaluation — both datasets, all metrics + slices"
python offline_evaluation.py --store "$STORE" --limit 0

if [[ "$RUN_SUBMISSIONS" == true ]]; then
  step "11. MIND Codabench submission (scored with the trained re-ranker)"
  python make_mind_submission.py --train "$MIND_TRAIN" --test "$MIND_TEST" --store "$STORE"

  step "12. EB-NeRD Codabench submission (scored with the trained re-ranker)"
  python make_ebnerd_personalized_submission.py --large "$EBNERD_LARGE" --testzip "$EBNERD_TEST" --store "$STORE" --source-root "$SOURCE_ROOT"
else
  step "11-12. Skipping Codabench submissions (pass --submissions to build them)"
fi

step "Done"
echo "Results:"
echo "  outputs/q2_reranker_metrics.json"
echo "  outputs/q3_ablation.json"
echo "  outputs/q4_serving_scale.json"
echo "  outputs/q5_evaluation.json"
echo "  outputs/q9_serving_availability.json"