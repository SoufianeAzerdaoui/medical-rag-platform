#!/usr/bin/env bash
set -euo pipefail

python3 -m unittest tests.test_reference_range_parser
python3 -m unittest tests.test_reference_range_selector
python3 -m unittest tests.test_reference_range_lookup_generation
python3 -m unittest tests.test_query_understanding
python3 -m unittest tests.test_evidence_filtering
python3 -m unittest tests.test_professional_answer_composer
npx tsc --noEmit
