#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
TOKEN="${TOKEN:-}"
CONV_ID="${CONV_ID:-}"
OUT_DIR="${OUT_DIR:-reports/summary_quality_check}"
DATE_TAG="${DATE_TAG:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-$OUT_DIR/$DATE_TAG}"
JSONL="$RUN_DIR/results.jsonl"
REPORT_JSON="$RUN_DIR/report.json"
REPORT_MD="$RUN_DIR/report.md"
LAST_RESPONSE_JSON="${LAST_RESPONSE_JSON:-/tmp/last_chat_response.json}"

SUMMARY_Q1="Fais une synthèse biologique courte du report 12. Limite-toi à 3 à 5 lignes, mentionne uniquement les anomalies majeures, les résultats dans la référence et une conclusion prudente, sans diagnostic."
SUMMARY_Q2="Résume le report 12 en 4 lignes maximum, avec une formulation médicale prudente, sans diagnostic et sans recommandation thérapeutique."
SUMMARY_Q3="Dans le report 12, quelle valeur est retrouvée pour la clairance de la créatinine ? Donne la valeur, l’unité, la référence si disponible, et la source."
SUMMARY_Q4="Dans le report 12, donne les anomalies de LDH, CKMB, bilirubine directe et ammonium."

summary_quality_require_tools() {
  local missing=()
  command -v curl >/dev/null 2>&1 || missing+=("curl")
  command -v jq >/dev/null 2>&1 || missing+=("jq")
  if [[ "${#missing[@]}" -gt 0 ]]; then
    echo "ERROR: missing required tools: ${missing[*]}" >&2
    return 2
  fi
}

summary_quality_require_env() {
  if [[ -z "${BASE_URL:-}" || -z "${TOKEN:-}" ]]; then
    echo "ERROR: BASE_URL and TOKEN are required" >&2
    return 2
  fi
}

summary_quality_healthcheck() {
  curl -sS --fail-with-body "$BASE_URL/health" >/dev/null
}

create_summary_quality_conversation() {
  summary_quality_require_tools
  summary_quality_require_env

  local title="${1:-test ux resume llm}"
  local payload created id
  payload="$(jq -nc --arg title "$title" '{title:$title}')"
  created="$(
    curl -sS --fail-with-body -X POST "$BASE_URL/conversations" \
      -H "Authorization: Bearer $TOKEN" \
      -H "Content-Type: application/json" \
      -d "$payload"
  )"
  id="$(printf '%s' "$created" | jq -r '.id // .conversation_id // empty')"
  if [[ -z "$id" ]]; then
    echo "ERROR: failed to create conversation" >&2
    printf '%s\n' "$created" >&2
    return 2
  fi
  export CONV_ID="$id"
  echo "$CONV_ID"
}

summary_quality_ensure_conversation() {
  if [[ -n "${CONV_ID:-}" ]]; then
    return 0
  fi
  create_summary_quality_conversation "test ux resume llm" >/dev/null
}

summary_quality_chat_raw() {
  local query="$1"
  local payload
  payload="$(jq -nc --arg conversation_id "$CONV_ID" --arg message "$query" '{conversation_id:$conversation_id,message:$message}')"
  curl -sS --fail-with-body -X POST "$BASE_URL/chat" \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -d "$payload"
}

summary_quality_projection_jq() {
  jq '{
    answer,
    sources,

    generation_mode,
    generation_writer,
    final_answer_source,
    fallback_reason,
    validation_status,
    quality_final_status,
    synthesis_quality_reason,
    response_time,

    provider,
    model,
    selected_route,
    llm_writer_attempted,
    llm_writer_accepted,

    llm_candidate_validation_status,
    llm_candidate_validation_errors,
    llm_candidate_validation_warnings,
    llm_candidate_rejected_reason,

    llm_repair_attempted,
    llm_repair_status,
    llm_repair_validation_errors,
    llm_repair_truncation_detected,

    llm_quality_gate,
    final_answer_quality_gate,

    requested_doc_id,
    resolved_doc_id,
    resolved_filename,
    resolved_page_count,
    indexed_page_count,
    document_identity_mismatch,

    displayed_evidences_count,
    evidence_pack_count,
    lab_result_count,
    value_numeric_count,
    structured_values_count,
    sources_count,
    above_reference_count,
    below_reference_count,
    within_reference_count,
    needs_clinical_context_count,
    major_anomalies_count,

    debug_selected_route: (.debug.selected_route // .debug.raw_debug.selected_route // null),
    debug_final_answer_source: (.debug.final_answer_source // .debug.raw_debug.final_answer_source // null),
    debug_fallback_reason: (.debug.fallback_reason // .debug.raw_debug.fallback_reason // null),
    debug_quality_final_status: (.debug.quality_final_status // .debug.raw_debug.quality_final_status // null),
    debug_llm_writer_ms: (.debug.stage_timings_ms.llm_writer_ms // .debug.raw_debug.stage_timings_ms.llm_writer_ms // null),
    debug_prompt_chars: (.debug.prompt_chars // .debug.raw_debug.prompt_chars // null),
    debug_prompt_hard_limit_chars: (.debug.prompt_hard_limit_chars // .debug.raw_debug.prompt_hard_limit_chars // null)
  }'
}

run_q() {
  local Q="$*"

  summary_quality_require_tools
  summary_quality_require_env
  if [[ -z "${CONV_ID:-}" ]]; then
    echo "Missing CONV_ID. Run: export CONV_ID=\"\$(create_summary_quality_conversation)\"" >&2
    return 1
  fi
  if [[ -z "$Q" ]]; then
    echo "Usage: run_q \"question\"" >&2
    return 1
  fi

  echo
  echo "============================================================"
  echo "Q: $Q"
  echo "============================================================"

  summary_quality_chat_raw "$Q" \
    | tee "$LAST_RESPONSE_JSON" \
    | summary_quality_projection_jq
}

summary_quality_sources_pages_gt_one() {
  jq '[.sources[]? | (.page // .page_number // empty) | tonumber? | select(. > 1)] | length'
}

summary_quality_contains_bad_legacy_terms() {
  jq -r '
    (.answer // "") as $answer
    | ["Bilirubine", "LDH", "CKMB", "APO", "AMMONIUM"]
    | map(select($answer | test(.; "i")))
    | join(",")
  '
}

summary_quality_case_verdict() {
  local case_id="$1"
  local resp_file="$2"

  jq -nc --arg case_id "$case_id" --slurpfile r "$resp_file" '
    def response: $r[0];
    def answer: (response.answer // "");
    def lower_answer: (answer | ascii_downcase);
    def sources: (response.sources // []);
    def source_pages_gt_one:
      [sources[]? | (.page // .page_number // empty) | tonumber? | select(. > 1)];
    def bad_legacy_terms:
      ["bilirubine", "ldh", "ckmb", "apo", "ammonium"]
      | map(. as $term | select(lower_answer | contains($term)));
    def old_report_value_leak:
      (answer | test("bilirubine[^\\n.;]{0,80}\\b6\\s*mg"; "i"))
      or (answer | test("\\bldh[^\\n.;]{0,80}\\b250\\s*ui"; "i"))
      or (answer | test("ck\\s*-?\\s*mb|ckmb"; "i") and (answer | test("\\b40\\s*ui"; "i")))
      or (answer | test("\\bapo[^\\n.;]{0,80}\\b2[,.]3\\s*g"; "i"))
      or (answer | test("ammonium[^\\n.;]{0,80}\\b20\\s*[µu]g"; "i"));
    def has_clairance:
      ((lower_answer | contains("clairance")) or (lower_answer | contains("créatinine")) or (lower_answer | contains("creatinine")));
    def has_value_20:
      (answer | test("(^|[^0-9])20([^0-9]|$)"));
    def final_source:
      (response.final_answer_source // response.debug.final_answer_source // response.debug.raw_debug.final_answer_source // "");
    def fallback:
      (response.fallback_reason // response.debug.fallback_reason // response.debug.raw_debug.fallback_reason // null);
    def quality:
      (response.quality_final_status // response.quality_report.final_status // response.debug.quality_final_status // response.debug.raw_debug.quality_final_status // "");
    def validation:
      (response.validation_status // response.validation.validation_status // response.debug.validation.validation_status // response.debug.raw_debug.validation.validation_status // "");
    def llm_accepted:
      ((response.llm_writer_accepted // response.debug.llm_writer_accepted // response.debug.raw_debug.llm_writer_accepted // false) == true);
    def source_count:
      ((response.sources_count // (sources | length)) | tonumber? // 0);
    def identity_ok:
      (response.resolved_doc_id == "report_12")
      and ((response.resolved_page_count // 0) == 1)
      and ((response.indexed_page_count // 0) == 1)
      and ((response.document_identity_mismatch // false) == false);
    def no_old_report_leak:
      ((bad_legacy_terms | length) == 0) and ((source_pages_gt_one | length) == 0);
    def no_old_report_value_leak:
      (old_report_value_leak | not) and ((source_pages_gt_one | length) == 0);
    def no_source_line_label:
      ([sources[]? | (.label // "") | select(test("ligne\\s+1"; "i"))] | length) == 0;
    def no_diagnosis_or_treatment:
      ((lower_answer | test("\\bdiagnostic\\s+(de|d\\u2019|d\\x27|affirm|probable|certain)|recommandation thérapeutique|traitement"; "i")) | not);
    def not_found_for_requested_absent_analytes:
      (lower_answer | test("pas retrouv|non retrouv|aucun.*(ldh|ckmb|bilirubine|ammonium)|ne sont pas retrouv|absence"; "i"));

    {
      case_id: $case_id,
      answer: answer,
      final_answer_source: final_source,
      fallback_reason: fallback,
      quality_final_status: quality,
      validation_status: validation,
      llm_writer_accepted: llm_accepted,
      source_count: source_count,
      resolved_doc_id: response.resolved_doc_id,
      resolved_page_count: response.resolved_page_count,
      indexed_page_count: response.indexed_page_count,
      document_identity_mismatch: response.document_identity_mismatch,
      bad_legacy_terms: bad_legacy_terms,
      source_pages_gt_one: source_pages_gt_one,
      checks: {
        identity_ok: identity_ok,
        source_count_positive: (source_count > 0),
        no_old_report_leak: no_old_report_leak,
        no_old_report_value_leak: no_old_report_value_leak,
        no_source_line_label: no_source_line_label,
        has_clairance: has_clairance,
        has_value_20: has_value_20,
        llm_writer_used: (final_source == "llm_writer"),
        llm_writer_accepted: llm_accepted,
        no_fallback: (fallback == null or fallback == ""),
        quality_pass: (quality == "pass"),
        validation_acceptable: (validation == "pass" or validation == "warning"),
        no_diagnosis_or_treatment: no_diagnosis_or_treatment,
        not_found_for_requested_absent_analytes: not_found_for_requested_absent_analytes
      }
    }
    | .required_checks = (
      if $case_id == "doc_identity_report12" then
        {
          identity_ok: .checks.identity_ok,
          source_count_positive: .checks.source_count_positive,
          no_old_report_leak: .checks.no_old_report_leak,
          no_source_line_label: .checks.no_source_line_label,
          has_clairance: .checks.has_clairance,
          llm_writer_used: .checks.llm_writer_used,
          no_fallback: .checks.no_fallback,
          quality_pass: .checks.quality_pass
        }
      elif $case_id == "short_writing_quality" then
        {
          has_clairance: .checks.has_clairance,
          source_count_positive: .checks.source_count_positive,
          llm_writer_accepted: .checks.llm_writer_accepted,
          no_fallback: .checks.no_fallback,
          quality_pass: .checks.quality_pass,
          no_diagnosis_or_treatment: .checks.no_diagnosis_or_treatment
        }
      elif $case_id == "strict_fact_value" then
        {
          has_clairance: .checks.has_clairance,
          has_value_20: .checks.has_value_20,
          source_count_positive: .checks.source_count_positive,
          no_old_report_leak: .checks.no_old_report_leak,
          no_source_line_label: .checks.no_source_line_label
        }
      else
        {
          source_count_positive: .checks.source_count_positive,
          no_old_report_value_leak: .checks.no_old_report_value_leak,
          not_found_for_requested_absent_analytes: .checks.not_found_for_requested_absent_analytes,
          validation_acceptable: .checks.validation_acceptable
        }
      end
    )
    | .pass = ([.required_checks[]] | all(. == true))
    | .failed_checks = (.required_checks | to_entries | map(select(.value != true)) | map(.key))
  '
}

summary_quality_run_case() {
  local case_id="$1"
  local question="$2"
  local index="$3"
  local resp_file="$RUN_DIR/${index}_${case_id}.json"

  echo
  echo "============================================================"
  echo "CASE $index: $case_id"
  echo "Q: $question"
  echo "============================================================"

  summary_quality_chat_raw "$question" | tee "$resp_file" >/dev/null
  cp "$resp_file" "$LAST_RESPONSE_JSON"

  local verdict
  verdict="$(summary_quality_case_verdict "$case_id" "$resp_file")"
  printf '%s\n' "$verdict" >> "$JSONL"

  local pass final_source quality failed
  pass="$(printf '%s' "$verdict" | jq -r '.pass')"
  final_source="$(printf '%s' "$verdict" | jq -r '.final_answer_source // ""')"
  quality="$(printf '%s' "$verdict" | jq -r '.quality_final_status // ""')"
  failed="$(printf '%s' "$verdict" | jq -r '.failed_checks | join(",")')"

  if [[ "$pass" == "true" ]]; then
    echo "PASS $case_id final_answer_source=$final_source quality=$quality"
  else
    echo "FAIL $case_id final_answer_source=$final_source quality=$quality failed_checks=$failed"
  fi

  printf '%s\n' "$verdict" | jq '{
    pass,
    failed_checks,
    final_answer_source,
    fallback_reason,
    quality_final_status,
    validation_status,
    resolved_doc_id,
    resolved_page_count,
    indexed_page_count,
    bad_legacy_terms,
    source_pages_gt_one,
    checks
  }'
}

summary_quality_build_report() {
  jq -s '
    {
      generated_at: (now | todate),
      base_url: env.BASE_URL,
      conversation_id: env.CONV_ID,
      total: length,
      passed: (map(select(.pass == true)) | length),
      failed: (map(select(.pass != true)) | length),
      results: .
    }
  ' "$JSONL" > "$REPORT_JSON"

  {
    echo "# Summary Quality Check"
    echo
    echo "- Generated at: $(date -Is)"
    echo "- Base URL: $BASE_URL"
    echo "- Conversation: $CONV_ID"
    echo "- Last response JSON: $LAST_RESPONSE_JSON"
    echo
    echo "## Summary"
    echo
    jq -r '"- Total: \(.total)\n- Passed: \(.passed)\n- Failed: \(.failed)"' "$REPORT_JSON"
    echo
    echo "## Results"
    echo
    jq -r '.results[] | "- [" + (if .pass then "PASS" else "FAIL" end) + "] " + .case_id + " | source=" + (.final_answer_source // "") + " | quality=" + (.quality_final_status // "") + " | failed=" + ((.failed_checks // []) | join(","))' "$REPORT_JSON"
    echo
    echo "## Front Checklist"
    echo
    echo "- Ouvrir une nouvelle conversation front."
    echo "- Rejouer les 4 questions dans le même ordre."
    echo "- Vérifier: mêmes faits, source page 1, pas de 'ligne 1' fictive, pas de fallback si final_answer_source=llm_writer."
  } > "$REPORT_MD"

  echo
  echo "Report JSON: $REPORT_JSON"
  echo "Report MD:   $REPORT_MD"
}

run_summary_quality_suite() {
  summary_quality_require_tools
  summary_quality_require_env
  summary_quality_healthcheck
  summary_quality_ensure_conversation

  mkdir -p "$RUN_DIR"
  : > "$JSONL"

  echo "BASE_URL=$BASE_URL"
  echo "CONV_ID=$CONV_ID"
  echo "RUN_DIR=$RUN_DIR"

  summary_quality_run_case "doc_identity_report12" "$SUMMARY_Q1" "01"
  summary_quality_run_case "short_writing_quality" "$SUMMARY_Q2" "02"
  summary_quality_run_case "strict_fact_value" "$SUMMARY_Q3" "03"
  summary_quality_run_case "anti_hallucination_absent_analytes" "$SUMMARY_Q4" "04"

  summary_quality_build_report

  local failed
  failed="$(jq -r '.failed' "$REPORT_JSON")"
  if [[ "$failed" != "0" ]]; then
    return 1
  fi
}

summary_quality_usage() {
  cat <<'EOF'
Usage:
  export BASE_URL="http://127.0.0.1:8000"
  export TOKEN="..."

  # Full backend suite:
  bash scripts/ops/summary_quality_check.sh

  # Interactive function:
  source scripts/ops/summary_quality_check.sh
  export CONV_ID="$(create_summary_quality_conversation)"
  run_q "Fais une synthèse biologique courte du report 12..."

Outputs:
  reports/summary_quality_check/<timestamp>/
  /tmp/last_chat_response.json
EOF
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  case "${1:-suite}" in
    suite)
      run_summary_quality_suite
      ;;
    create-conversation)
      create_summary_quality_conversation "${2:-test ux resume llm}"
      ;;
    help|-h|--help)
      summary_quality_usage
      ;;
    *)
      echo "ERROR: unknown command: $1" >&2
      summary_quality_usage >&2
      exit 2
      ;;
  esac
fi
