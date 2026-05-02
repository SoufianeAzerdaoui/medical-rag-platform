# 2) Lancer l’extraction pour tous les PDFs
for pdf in docs/*.pdf; do
  python3 scripts/extraction_data/run_extraction.py "$pdf" --output-root data/extraction
done

# 3) Validation QA des outputs extraction
python3 tests/validate_extraction_outputs.py \
  --root data/extraction \
  --report-json data/extraction/qa_validation_report.json

# 4) Validation end-to-end docs (score PASS/WARNING/FAIL par PDF)
python3 tests/validate_docs_end_to_end.py

# 5) Validation ciblée du rapport de référence
python3 tests/validate_real_report_extraction.py
