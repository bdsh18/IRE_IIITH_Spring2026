#!/usr/bin/env bash
# Runs the full Assignment 2 pipeline end to end, in dependency order.
#
# Usage:
#   ./run_a2_pipeline.sh                  # full run, both datasets, no submissions
#   ./run_a2_pipeline.sh --submissions     # also builds the two Codabench ZIPs
#   ./run_a2_pipeline.sh --submissions-only # build only the two final ZIPs
#   ./run_a2_pipeline.sh --skip-install    # skip the pip install step
#   ./run_a2_pipeline.sh --smoke-limit 500 # cap reranker/ablation on N impressions
#
# Env vars (override if your paths differ from the assignment defaults):
#   SOURCE_ROOT   raw-data root passed to build_pipeline.py   (default: .)
#   STORE         processed feature store                     (default: data/processed)
#   MIND_TRAIN    MINDlarge_train.zip, for the submission step (default: MIND_data/MINDlarge_train.zip)
#   MIND_DEV      MINDlarge_dev.zip,   for the submission step (default: MIND_data/MINDlarge_dev.zip)
#   MIND_TEST     MINDlarge_test.zip,  for the submission step (default: MIND_data/MINDlarge_test.zip)
#   EBNERD_LARGE  ebnerd_large.zip,    for the submission step (default: ebnerd_large.zip)
#   EBNERD_TEST   ebnerd_testset.zip,  for the submission step (default: ebnerd_testset.zip)
#   SUBMISSION_BATCH_SIZE durable submission checkpoint size (default: 5000)
#   PYTHON_BIN    Python interpreter / virtual-environment executable (default: python3)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Prefer data colocated with A2, but automatically reuse the downloaded A1
# archives when A2 lives in the same IRA workspace.
DEFAULT_SOURCE_ROOT="$SCRIPT_DIR"
if [[ ! -f "$DEFAULT_SOURCE_ROOT/MIND_data/MINDsmall_train.zip" && -f "$SCRIPT_DIR/../../Assighnment1/MIND_data/MINDsmall_train.zip" ]]; then
  DEFAULT_SOURCE_ROOT="$SCRIPT_DIR/../../Assighnment1"
fi
SOURCE_ROOT="${SOURCE_ROOT:-$DEFAULT_SOURCE_ROOT}"
STORE="${STORE:-data/processed}"
MIND_TRAIN="${MIND_TRAIN:-$SOURCE_ROOT/MIND_data/MINDlarge_train.zip}"
MIND_DEV="${MIND_DEV:-$SOURCE_ROOT/MIND_data/MINDlarge_dev.zip}"
MIND_TEST="${MIND_TEST:-$SOURCE_ROOT/MIND_data/MINDlarge_test.zip}"
EBNERD_LARGE="${EBNERD_LARGE:-$SOURCE_ROOT/ebnerd_large.zip}"
EBNERD_TEST="${EBNERD_TEST:-$SOURCE_ROOT/ebnerd_testset.zip}"
SUBMISSION_BATCH_SIZE="${SUBMISSION_BATCH_SIZE:-5000}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

RUN_SUBMISSIONS=false
SUBMISSIONS_ONLY=false
SKIP_INSTALL=false
SMOKE_LIMIT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --submissions)  RUN_SUBMISSIONS=true; shift ;;
    --submissions-only) RUN_SUBMISSIONS=true; SUBMISSIONS_ONLY=true; shift ;;
    --skip-install) SKIP_INSTALL=true; shift ;;
    --smoke-limit)  SMOKE_LIMIT="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

step() { echo; echo "=== $1 ==="; }

if [[ "$SUBMISSIONS_ONLY" == false && "$SKIP_INSTALL" == false ]]; then
  step "0. Installing dependencies"
  "$PYTHON_BIN" -m pip install -r requirements.txt -q
fi

if [[ "$SUBMISSIONS_ONLY" == false ]]; then
  step "1. Rebuilding the feature store (build_pipeline.py)"
  "$PYTHON_BIN" build_pipeline.py --source-root "$SOURCE_ROOT" --output "$STORE"

  step "2. Leakage / behavioural-window boundary test (Q9)"
  "$PYTHON_BIN" -m pytest test_no_leakage.py -v

  step "3. Candidate generation — BM25 recall check (A1 / Q2 stage 1)"
  "$PYTHON_BIN" bm25_retrieval.py --store "$STORE"

  step "4. Candidate generation — semantic embeddings (A1 / Q2 stage 1)"
  "$PYTHON_BIN" semantic_retrieval.py --source-root "$SOURCE_ROOT" --store "$STORE"

  for DATASET in mind ebnerd; do
    step "5. Q1 features — $DATASET"
    "$PYTHON_BIN" features.py --dataset "$DATASET" --split validation --store "$STORE"

    step "6. Q2 re-ranker — $DATASET"
    if [[ "$SMOKE_LIMIT" -gt 0 ]]; then
      "$PYTHON_BIN" reranker.py --dataset "$DATASET" --store "$STORE" --limit "$SMOKE_LIMIT" \
        --serving-only --strict-two-stage \
        --output "outputs/q2_reranker_metrics_${DATASET}.json"
    else
      "$PYTHON_BIN" reranker.py --dataset "$DATASET" --store "$STORE" \
        --serving-only --strict-two-stage \
        --output "outputs/q2_reranker_metrics_${DATASET}.json"
    fi

    step "7. Q3 ablation (paired bootstrap) — $DATASET"
    "$PYTHON_BIN" ablation.py --dataset "$DATASET" --store "$STORE" --limit "$SMOKE_LIMIT" --serving-only \
      --output "outputs/q3_ablation_${DATASET}.json"

    step "8. Q4 serving & scale — $DATASET"
    "$PYTHON_BIN" serving_scale.py --dataset "$DATASET" --store "$STORE" --target-sla-ms 100 \
      --output "outputs/q4_serving_scale_${DATASET}.json"

    step "9. Q9 serving-availability ablation — $DATASET"
    "$PYTHON_BIN" serving_availability_ablation.py --dataset "$DATASET" --store "$STORE" --limit "$SMOKE_LIMIT" \
      --output "outputs/q9_serving_availability_${DATASET}.json"
  done

  step "10. Q5 extended evaluation — both datasets, all metrics + slices"
  "$PYTHON_BIN" offline_evaluation.py --store "$STORE" --limit 0 --serving-only --strict-two-stage
else
  step "0-10. Reusing the existing processed store and trained-feature caches"
fi

if [[ "$RUN_SUBMISSIONS" == true ]]; then
  step "11. MIND Codabench submission (scored with the trained re-ranker)"
  "$PYTHON_BIN" make_mind_submission.py --train "$MIND_TRAIN" --dev "$MIND_DEV" --test "$MIND_TEST" \
    --store "$STORE" --output-dir outputs/mind_submission --batch-size "$SUBMISSION_BATCH_SIZE"

  step "12. EB-NeRD Codabench submission (scored with the trained re-ranker)"
  "$PYTHON_BIN" make_ebnerd_personalized_submission.py --large "$EBNERD_LARGE" --testzip "$EBNERD_TEST" \
    --store "$STORE" --source-root "$SOURCE_ROOT" --work data/raw/ebnerd_personalized \
    --out outputs/ebnerd_submission --batch-size "$SUBMISSION_BATCH_SIZE"
else
  step "11-12. Skipping Codabench submissions (pass --submissions to build them)"
fi

step "Done"
echo "Results:"
echo "  outputs/q2_reranker_metrics_mind.json"
echo "  outputs/q2_reranker_metrics_ebnerd.json"
echo "  outputs/q3_ablation_mind.json"
echo "  outputs/q3_ablation_ebnerd.json"
echo "  outputs/q4_serving_scale_mind.json"
echo "  outputs/q4_serving_scale_ebnerd.json"
echo "  outputs/q5_evaluation.json"
echo "  outputs/q9_serving_availability_mind.json"
echo "  outputs/q9_serving_availability_ebnerd.json"
