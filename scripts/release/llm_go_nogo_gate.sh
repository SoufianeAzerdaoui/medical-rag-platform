#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

mkdir -p reports

# 1) Suite 15 target gate (safety/hallucination)
python3 scripts/evaluation/comprehensive_rag_tester.py \
  --suites suite_15_unexpected_user_phrasings \
  --output reports/unexpected_user_phrasings_prod_gate.json

python3 scripts/evaluation/analyze_suite15_targets.py \
  --report reports/unexpected_user_phrasings_prod_gate.json \
  --output-json reports/suite15_targets_prod_gate.json \
  --output-md reports/suite15_targets_prod_gate.md \
  --enforce

# 2) LLM go/no-go gate
# Requires benchmark artifacts:
# - reports/llm_writer_benchmark_results.json       (LLM ON)
# - reports/llm_writer_benchmark_results_no_llm.json (LLM OFF baseline)
#
# Thresholds can be overridden via env vars:
# - LLM_MAX_TIMEOUT_RATE
# - LLM_MAX_FALLBACK_AFTER_RATE
# - LLM_MIN_ACCEPT_RATE
# - LLM_MAX_P95_RESPONSE_MS
# - LLM_MAX_P95_WRITER_MS
# - LLM_MIN_PROFESSIONAL_ACCEPT_RATE
# - LLM_MIN_SCORE_DELTA
# - LLM_MAX_TIMEOUT_STABILITY_DELTA
# - LLM_MIN_EXPECTED_COUNT

CHAT_SUMMARY_LOG_PATH="${CHAT_SUMMARY_LOG_PATH:-}"
CHAT_SUMMARY_LOG_ARGS=()
if [[ -n "$CHAT_SUMMARY_LOG_PATH" ]]; then
  CHAT_SUMMARY_LOG_ARGS+=(--chat-summary-log "$CHAT_SUMMARY_LOG_PATH")
fi

LLM_MAX_TIMEOUT_RATE="${LLM_MAX_TIMEOUT_RATE:-0.10}"
LLM_MAX_FALLBACK_AFTER_RATE="${LLM_MAX_FALLBACK_AFTER_RATE:-0.25}"
LLM_MIN_ACCEPT_RATE="${LLM_MIN_ACCEPT_RATE:-0.60}"
LLM_MAX_P95_RESPONSE_MS="${LLM_MAX_P95_RESPONSE_MS:-3000}"
LLM_MAX_P95_WRITER_MS="${LLM_MAX_P95_WRITER_MS:-2500}"
LLM_MIN_PROFESSIONAL_ACCEPT_RATE="${LLM_MIN_PROFESSIONAL_ACCEPT_RATE:-0.95}"
LLM_MIN_SCORE_DELTA="${LLM_MIN_SCORE_DELTA:-0.0}"
LLM_MAX_TIMEOUT_STABILITY_DELTA="${LLM_MAX_TIMEOUT_STABILITY_DELTA:-0.15}"
LLM_MIN_EXPECTED_COUNT="${LLM_MIN_EXPECTED_COUNT:-1}"

python3 scripts/evaluation/analyze_llm_go_nogo.py \
  --suite15-targets reports/suite15_targets_prod_gate.json \
  --llm-on-benchmark reports/llm_writer_benchmark_results.json \
  --llm-off-benchmark reports/llm_writer_benchmark_results_no_llm.json \
  --output-json reports/llm_go_nogo_report.json \
  --output-md reports/llm_go_nogo_report.md \
  --max-llm-timeout-rate "$LLM_MAX_TIMEOUT_RATE" \
  --max-fallback-after-llm-rate "$LLM_MAX_FALLBACK_AFTER_RATE" \
  --min-llm-accept-rate "$LLM_MIN_ACCEPT_RATE" \
  --max-p95-response-ms "$LLM_MAX_P95_RESPONSE_MS" \
  --max-p95-llm-writer-ms "$LLM_MAX_P95_WRITER_MS" \
  --min-professional-llm-accept-rate "$LLM_MIN_PROFESSIONAL_ACCEPT_RATE" \
  --min-llm-score-delta-vs-baseline "$LLM_MIN_SCORE_DELTA" \
  --max-llm-timeout-stability-delta "$LLM_MAX_TIMEOUT_STABILITY_DELTA" \
  --min-llm-expected-count "$LLM_MIN_EXPECTED_COUNT" \
  "${CHAT_SUMMARY_LOG_ARGS[@]}" \
  --enforce

python3 scripts/evaluation/analyze_llm_preprod_readiness.py \
  --go-nogo-report reports/llm_go_nogo_report.json \
  --llm-on-benchmark reports/llm_writer_benchmark_results.json \
  --llm-off-benchmark reports/llm_writer_benchmark_results_no_llm.json \
  --output-json reports/llm_preprod_readiness_report.json \
  --output-md reports/llm_preprod_readiness_report.md \
  --enforce

echo "[OK] LLM go/no-go gate passed."
