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

CHAT_SUMMARY_LOG_PATH="${CHAT_SUMMARY_LOG_PATH:-}"
CHAT_SUMMARY_LOG_ARGS=()
if [[ -n "$CHAT_SUMMARY_LOG_PATH" ]]; then
  CHAT_SUMMARY_LOG_ARGS+=(--chat-summary-log "$CHAT_SUMMARY_LOG_PATH")
fi

python3 scripts/evaluation/analyze_llm_go_nogo.py \
  --suite15-targets reports/suite15_targets_prod_gate.json \
  --llm-on-benchmark reports/llm_writer_benchmark_results.json \
  --llm-off-benchmark reports/llm_writer_benchmark_results_no_llm.json \
  --output-json reports/llm_go_nogo_report.json \
  --output-md reports/llm_go_nogo_report.md \
  "${CHAT_SUMMARY_LOG_ARGS[@]}" \
  --enforce

echo "[OK] LLM go/no-go gate passed."

