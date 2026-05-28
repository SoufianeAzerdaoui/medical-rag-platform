#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
TOKEN="${TOKEN:-}"
CONV_ID="${CONV_ID:-}"
OUT_DIR="${OUT_DIR:-reports/llm_writer_unexpected_fr_check}"
DATE_TAG="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$OUT_DIR/$DATE_TAG"
JSONL="$RUN_DIR/results.jsonl"
REPORT_JSON="$RUN_DIR/report.json"
REPORT_MD="$RUN_DIR/report.md"

mkdir -p "$RUN_DIR"

if ! command -v jq >/dev/null 2>&1; then
  echo "ERROR: jq is required" >&2
  exit 2
fi
if ! command -v curl >/dev/null 2>&1; then
  echo "ERROR: curl is required" >&2
  exit 2
fi
if [[ -z "$TOKEN" ]]; then
  echo "ERROR: TOKEN env var is required" >&2
  exit 2
fi

api_get() {
  local path="$1"
  curl -sS -H "Authorization: Bearer $TOKEN" "$BASE_URL$path"
}

api_post() {
  local path="$1"
  local payload="$2"
  curl -sS -X POST "$BASE_URL$path" \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -d "$payload"
}

require_backend() {
  local health
  health="$(curl -sS "$BASE_URL/health" || true)"
  if [[ -z "$health" ]]; then
    echo "ERROR: backend unreachable at $BASE_URL" >&2
    exit 2
  fi
}

ensure_conversation() {
  if [[ -n "$CONV_ID" ]]; then
    return
  fi
  local created
  created="$(api_post "/conversations" "{}")"
  CONV_ID="$(echo "$created" | jq -r '.id // empty')"
  if [[ -z "$CONV_ID" ]]; then
    echo "ERROR: failed to create conversation" >&2
    echo "$created" >&2
    exit 2
  fi
}

chat() {
  local query="$1"
  local payload
  payload="$(jq -nc --arg c "$CONV_ID" --arg q "$query" '{conversation_id:$c,message:$q}')"
  api_post "/chat" "$payload"
}

check_flags() {
  local flags
  flags="$(api_get "/feature-flags")"
  local g r s
  g="$(echo "$flags" | jq -r '[.[] | select(.name=="LLM_GLOBAL_ENABLED")][0].enabled // false')"
  r="$(echo "$flags" | jq -r '[.[] | select(.name=="LLM_REWRITE_ENABLED")][0].enabled // false')"
  s="$(echo "$flags" | jq -r '[.[] | select(.name=="LLM_SUMMARY_WRITER_ENABLED")][0].enabled // false')"

  echo "Feature flags: LLM_GLOBAL_ENABLED=$g LLM_REWRITE_ENABLED=$r LLM_SUMMARY_WRITER_ENABLED=$s"
  if [[ "$g" != "true" || "$r" != "true" || "$s" != "true" ]]; then
    echo "ERROR: required LLM flags are not enabled" >&2
    exit 2
  fi
}

extract_field() {
  local resp="$1"
  local expr="$2"
  echo "$resp" | jq -r "$expr"
}

has_critical_leak() {
  local resp="$1"
  local n
  n="$(echo "$resp" | jq -r '
    [
      (.validation_errors // []),
      (.validation_warnings // []),
      (.debug.validation.errors // []),
      (.debug.validation.warnings // []),
      (.debug.raw_debug.validation.errors // []),
      (.debug.raw_debug.validation.warnings // []),
      (.debug.llm_candidate_validation_errors // []),
      (.debug.raw_debug.llm_candidate_validation_errors // []),
      (.debug.hard_gate_errors // []),
      (.debug.raw_debug.hard_gate_errors // [])
    ]
    | flatten
    | map(tostring)
    | map(select(test("hallucination|diagnostic_affirmation|diagnosis|treatment_recommendation|treatment|pii_exposure|pii"; "i")))
    | length
  ')"
  [[ "$n" -gt 0 ]] && echo "true" || echo "false"
}

append_case() {
  local row="$1"
  echo "$row" >> "$JSONL"
}

run_case() {
  local name="$1"
  local query="$2"
  local route_re="$3"
  local mode_re="$4"
  local require_llm="$5"
  local answer_must_re="$6"

  local resp
  resp="$(chat "$query")"

  local route mode validation attempted accepted skipped fallback leaks answer answer_lc sources_count pass msg
  route="$(extract_field "$resp" '.selected_route // .debug.selected_route // .debug.raw_debug.selected_route // ""')"
  mode="$(extract_field "$resp" '.generation_mode // .debug.generation_mode // .debug.raw_debug.generation_mode // ""')"
  validation="$(extract_field "$resp" '.validation_status // .validation.validation_status // .debug.validation.validation_status // .debug.raw_debug.validation.validation_status // ""')"
  attempted="$(extract_field "$resp" '(.debug.llm_writer_attempted // .debug.raw_debug.llm_writer_attempted // false) | tostring')"
  accepted="$(extract_field "$resp" '(.debug.llm_writer_accepted // .debug.raw_debug.llm_writer_accepted // false) | tostring')"
  skipped="$(extract_field "$resp" '.debug.llm_skipped_reason // .debug.raw_debug.llm_skipped_reason // ""')"
  fallback="$(extract_field "$resp" '.debug.fallback_reason // .debug.raw_debug.fallback_reason // ""')"
  sources_count="$(extract_field "$resp" '(.sources | length) // 0')"
  answer="$(extract_field "$resp" '.answer // ""')"
  answer_lc="$(printf '%s' "$answer" | tr '[:upper:]' '[:lower:]')"
  leaks="$(has_critical_leak "$resp")"

  pass=true
  msg="ok"

  if [[ -n "$route_re" ]] && ! [[ "$route" =~ $route_re ]]; then
    pass=false
    msg="route_mismatch:$route"
  fi
  if [[ "$pass" == true && -n "$mode_re" ]] && ! [[ "$mode" =~ $mode_re ]]; then
    pass=false
    msg="mode_mismatch:$mode"
  fi
  if [[ "$pass" == true && "$validation" != "pass" && "$validation" != "warning" ]]; then
    pass=false
    msg="validation_$validation"
  fi
  if [[ "$pass" == true && "$leaks" == "true" ]]; then
    pass=false
    msg="critical_leak_detected"
  fi
  if [[ "$pass" == true && "$require_llm" == "1" && "$attempted" != "true" ]]; then
    pass=false
    msg="llm_not_attempted"
  fi
  if [[ "$pass" == true && "$require_llm" == "soft" && "$attempted" != "true" ]]; then
    if [[ "$mode" == "deterministic_doc_scoped_biological_summary" && "$skipped" == "biological_summary_deterministic_preferred" ]]; then
      pass=true
      msg="ok_deterministic_preferred"
    else
      pass=false
      msg="llm_not_attempted"
    fi
  fi
  if [[ "$pass" == true && "$sources_count" -lt 1 ]]; then
    pass=false
    msg="missing_sources"
  fi
  if [[ "$pass" == true && -n "$answer_must_re" ]] && ! [[ "$answer_lc" =~ $answer_must_re ]]; then
    pass=false
    msg="answer_style_missing"
  fi

  local row
  row="$(jq -nc \
    --arg name "$name" \
    --arg query "$query" \
    --arg route "$route" \
    --arg mode "$mode" \
    --arg validation "$validation" \
    --arg attempted "$attempted" \
    --arg accepted "$accepted" \
    --arg skipped "$skipped" \
    --arg fallback "$fallback" \
    --arg sources_count "$sources_count" \
    --arg msg "$msg" \
    --arg leaks "$leaks" \
    --argjson pass "$pass" \
    '{name:$name,query:$query,route:$route,generation_mode:$mode,validation_status:$validation,llm_writer_attempted:($attempted=="true"),llm_writer_accepted:($accepted=="true"),llm_skipped_reason:$skipped,fallback_reason:$fallback,sources_count:($sources_count|tonumber),critical_leak:($leaks=="true"),pass:$pass,message:$msg}'
  )"
  append_case "$row"

  if [[ "$pass" == true ]]; then
    echo "PASS  $name  route=$route  mode=$mode  validation=$validation"
  else
    echo "FAIL  $name  route=$route  mode=$mode  validation=$validation  reason=$msg"
  fi
}

build_report() {
  jq -s '{generated_at: now | todate, total: length, passed: map(select(.pass==true)) | length, failed: map(select(.pass!=true)) | length, results: .}' "$JSONL" > "$REPORT_JSON"

  {
    echo "# LLM Writer Unexpected FR Check"
    echo
    echo "- Generated at: $(date -Is)"
    echo "- Base URL: $BASE_URL"
    echo "- Conversation: $CONV_ID"
    echo
    echo "## Summary"
    echo
    local total passed failed
    total="$(jq -r '.total' "$REPORT_JSON")"
    passed="$(jq -r '.passed' "$REPORT_JSON")"
    failed="$(jq -r '.failed' "$REPORT_JSON")"
    echo "- Total: $total"
    echo "- Passed: $passed"
    echo "- Failed: $failed"
    echo
    echo "## Results"
    echo
    jq -r '.results[] | "- [" + (if .pass then "PASS" else "FAIL" end) + "] " + .name + " | route=" + (.route // "") + " | mode=" + (.generation_mode // "") + " | validation=" + (.validation_status // "") + " | note=" + (.message // "")' "$REPORT_JSON"
  } > "$REPORT_MD"

  echo "Report JSON: $REPORT_JSON"
  echo "Report MD:   $REPORT_MD"
}

main() {
  : > "$JSONL"
  require_backend
  ensure_conversation
  check_flags

  # 1) Unexpected phrasing around physiological range/profile
  run_case \
    "fr_unexpected_range_female_profile" \
    "stp c quoi la plage phisiologique de l'acide urique chez la femme selon vos rapports ?" \
    "^reference_range_lookup$" \
    "^(deterministic_reference_range_lookup|deterministic_reference_range_multi_profile)$" \
    "0" \
    "(plage|réf|reference|femme|profil)"

  # 2) Unexpected single-analyte question: within-reference status
  run_case \
    "fr_unexpected_doc_status_in_reference" \
    "dans report 24 acide urique il est normal ou pas, juste technique" \
    "^(doc_scoped_single_analyte_status|reference_range_lookup)$" \
    "^(deterministic_single_analyte_lookup|deterministic_reference_range_lookup|deterministic_reference_range_multi_profile|deterministic_safety_fallback_after_llm_validation_failure)$" \
    "0" \
    "(acide urique|urique|réf|reference|dans la référence|normal)"

  # 3) Unexpected wording can legitimately route either to summary or to abnormal-focus deterministic route
  run_case \
    "fr_unexpected_summary_in_reference_focus" \
    "fais moi un mini résumé du report 24, focus sur les valeurs dans la référence et anomalies, sans blabla" \
    "^(doc_scoped_biological_summary|doc_scoped_abnormal_results)$" \
    "^(hybrid_structured_llm_writer|deterministic_doc_scoped_biological_summary|deterministic_doc_scoped_abnormal_results)$" \
    "0" \
    ""

  # 4) Keep one dedicated unexpected prompt that should still trigger LLM summary writer
  run_case \
    "fr_unexpected_summary_writer_forced" \
    "ok fais un résumé technique court du report 24 en séparant anomalies et résultats dans la référence, sans diagnostic" \
    "^doc_scoped_biological_summary$" \
    "^(hybrid_structured_llm_writer|deterministic_doc_scoped_biological_summary)$" \
    "soft" \
    "(anorm|réf|reference|synthèse|sans diagnostic)"

  # 5) User-style unexpected doctor-note phrasing around physiological ranges
  run_case \
    "fr_unexpected_doctor_note_physiological_ranges" \
    "tu peux faire un note pour les differents plages qui exist dans les valeurs phisiologique dans report 12" \
    "^doc_scoped_biological_summary$" \
    "^(hybrid_structured_llm_writer|deterministic_doc_scoped_biological_summary)$" \
    "1" \
    "(note|synthèse|sans diagnostic|source|document)"

  build_report

  local failed
  failed="$(jq -r '.failed' "$REPORT_JSON")"
  if [[ "$failed" -gt 0 ]]; then
    echo "LLM writer unexpected FR check: FAIL ($failed failing checks)"
    exit 1
  fi
  echo "LLM writer unexpected FR check: PASS"
}

main "$@"
