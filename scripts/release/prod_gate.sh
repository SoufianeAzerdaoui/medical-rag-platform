#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

mkdir -p reports

python3 -m unittest \
  tests.test_generation_doc_scope \
  tests.test_query_understanding \
  tests.test_reference_range_lookup_generation \
  tests.test_analyte_resolver \
  tests.test_general_conversation_config \
  tests.test_query_planner \
  -v

python3 scripts/evaluation/advanced_medical_smoke_runner.py

python3 scripts/evaluation/comprehensive_rag_tester.py \
  --output reports/full_test_prod_gate.json

python3 scripts/evaluation/comprehensive_rag_tester.py \
  --suites suite_15_unexpected_user_phrasings \
  --output reports/unexpected_user_phrasings_prod_gate.json

python3 scripts/evaluation/analyze_suite15_targets.py \
  --report reports/unexpected_user_phrasings_prod_gate.json \
  --output-json reports/suite15_targets_prod_gate.json \
  --output-md reports/suite15_targets_prod_gate.md \
  --enforce
