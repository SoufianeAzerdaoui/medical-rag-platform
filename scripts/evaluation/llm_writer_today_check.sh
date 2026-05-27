#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
TOKEN="${TOKEN:-}"
CONV_ID="${CONV_ID:-}"
OUT_DIR="${OUT_DIR:-reports/llm_writer_today_check}"
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

append_case() {
  local row="$1"
  echo "$row" >> "$JSONL"
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
  ' )"
  [[ "$n" -gt 0 ]] && echo "true" || echo "false"
}

run_case() {
  local name="$1"
  local query="$2"
  local route_re="$3"
  local must_attempt="$4"
  local must_accept="$5"
  local mode_re="$6"

  local resp
  resp="$(chat "$query")"

  local route mode validation attempted accepted fallback reason leaks pass msg
  route="$(extract_field "$resp" '.selected_route // .debug.selected_route // .debug.raw_debug.selected_route // ""')"
  mode="$(extract_field "$resp" '.generation_mode // .debug.generation_mode // .debug.raw_debug.generation_mode // ""')"
  validation="$(extract_field "$resp" '.validation_status // .validation.validation_status // .debug.validation.validation_status // .debug.raw_debug.validation.validation_status // ""')"
  attempted="$(extract_field "$resp" '(.debug.llm_writer_attempted // .debug.raw_debug.llm_writer_attempted // false) | tostring')"
  accepted="$(extract_field "$resp" '(.debug.llm_writer_accepted // .debug.raw_debug.llm_writer_accepted // false) | tostring')"
  fallback="$(extract_field "$resp" '.debug.fallback_reason // .debug.raw_debug.fallback_reason // ""')"
  reason="$(extract_field "$resp" '.debug.llm_skipped_reason // .debug.raw_debug.llm_skipped_reason // ""')"
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
  if [[ "$pass" == true && "$must_attempt" == "1" && "$attempted" != "true" ]]; then
    pass=false
    msg="llm_not_attempted"
  fi
  if [[ "$pass" == true && "$must_accept" == "1" && "$accepted" != "true" ]]; then
    pass=false
    msg="llm_not_accepted"
  fi
  if [[ "$pass" == true && "$validation" != "pass" && "$validation" != "warning" ]]; then
    pass=false
    msg="validation_$validation"
  fi
  if [[ "$pass" == true && "$leaks" == "true" ]]; then
    pass=false
    msg="critical_leak_detected"
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
    --arg fallback "$fallback" \
    --arg skipped "$reason" \
    --arg leaks "$leaks" \
    --arg msg "$msg" \
    --argjson pass "$pass" \
    '{name:$name,query:$query,route:$route,generation_mode:$mode,validation_status:$validation,llm_writer_attempted:($attempted=="true"),llm_writer_accepted:($accepted=="true"),fallback_reason:$fallback,llm_skipped_reason:$skipped,critical_leak:($leaks=="true"),pass:$pass,message:$msg}'
  )"

  append_case "$row"

  if [[ "$pass" == true ]]; then
    echo "PASS  $name  route=$route  mode=$mode  validation=$validation"
  else
    echo "FAIL  $name  route=$route  mode=$mode  validation=$validation  reason=$msg"
  fi
}

run_open_grounded_case() {
  local name="open_grounded_medical_question"
  local prompts=(
    "Les rapports disponibles permettent-ils d’affirmer une pathologie endocrinienne active ?"
    "À partir des rapports disponibles, quels éléments biologiques documentés nécessitent une surveillance, sans poser de diagnostic ?"
    "Donne une réponse médicale prudente basée uniquement sur les données biologiques disponibles, sans diagnostic."
  )

  local passed=false
  local i=0
  for q in "${prompts[@]}"; do
    i=$((i+1))
    local resp route mode validation attempted accepted leaks skipped
    resp="$(chat "$q")"
    route="$(extract_field "$resp" '.selected_route // .debug.selected_route // .debug.raw_debug.selected_route // ""')"
    mode="$(extract_field "$resp" '.generation_mode // .debug.generation_mode // .debug.raw_debug.generation_mode // ""')"
    validation="$(extract_field "$resp" '.validation_status // .validation.validation_status // .debug.validation.validation_status // .debug.raw_debug.validation.validation_status // ""')"
    attempted="$(extract_field "$resp" '(.debug.llm_writer_attempted // .debug.raw_debug.llm_writer_attempted // false) | tostring')"
    accepted="$(extract_field "$resp" '(.debug.llm_writer_accepted // .debug.raw_debug.llm_writer_accepted // false) | tostring')"
    skipped="$(extract_field "$resp" '.debug.llm_skipped_reason // .debug.raw_debug.llm_skipped_reason // ""')"
    leaks="$(has_critical_leak "$resp")"

    local pass=false
    local msg="route_or_llm_not_matching"
    if [[ "$route" == "open_grounded_medical_question" && "$attempted" == "true" && ( "$validation" == "pass" || "$validation" == "warning" ) && "$leaks" == "false" ]]; then
      pass=true
      passed=true
      msg="ok"
    elif [[ "$route" == "open_grounded_medical_question" && "$attempted" == "false" && "$mode" == "deterministic_no_evidence_response" && ( "$validation" == "pass" || "$validation" == "warning" ) && "$leaks" == "false" ]]; then
      pass=true
      passed=true
      msg="ok_fallback_no_evidence"
    fi

    local row
    row="$(jq -nc \
      --arg name "$name#$i" \
      --arg query "$q" \
      --arg route "$route" \
      --arg mode "$mode" \
      --arg validation "$validation" \
      --arg attempted "$attempted" \
      --arg accepted "$accepted" \
      --arg leaks "$leaks" \
      --arg skipped "$skipped" \
      --arg msg "$msg" \
      --argjson pass "$pass" \
      '{name:$name,query:$query,route:$route,generation_mode:$mode,validation_status:$validation,llm_writer_attempted:($attempted=="true"),llm_writer_accepted:($accepted=="true"),llm_skipped_reason:$skipped,critical_leak:($leaks=="true"),pass:$pass,message:$msg}'
    )"
    append_case "$row"

    if [[ "$pass" == true ]]; then
      echo "PASS  $name via prompt#$i route=$route"
      break
    else
      echo "INFO  $name prompt#$i did not match strict target (route=$route, attempted=$attempted)"
    fi
  done

  if [[ "$passed" != true ]]; then
    echo "FAIL  $name no prompt reached strict llm-writer open_grounded path"
  fi
}

run_response_transform_case() {
  local seed="Résume report 24 en 5 lignes max, strictement technique."
  local follow="Convertis la réponse précédente en style paragraphe médical pro."

  local _seed_resp
  _seed_resp="$(chat "$seed")"

  local resp
  resp="$(chat "$follow")"

  local route mode validation fallback leaks pass msg
  route="$(extract_field "$resp" '.selected_route // .debug.selected_route // .debug.raw_debug.selected_route // ""')"
  mode="$(extract_field "$resp" '.generation_mode // .debug.generation_mode // .debug.raw_debug.generation_mode // ""')"
  validation="$(extract_field "$resp" '.validation_status // .validation.validation_status // .debug.validation.validation_status // .debug.raw_debug.validation.validation_status // ""')"
  fallback="$(extract_field "$resp" '.debug.fallback_reason // .debug.raw_debug.fallback_reason // ""')"
  leaks="$(has_critical_leak "$resp")"

  pass=true
  msg="ok"
  if [[ -z "$route" && "$mode" =~ ^deterministic_response_transform_ ]]; then
    route="response_transform"
  fi
  if [[ "$route" != "response_transform" ]]; then
    pass=false
    msg="route_mismatch:$route"
  fi
  if [[ "$pass" == true && ! "$mode" =~ ^deterministic_response_transform_(professional|json)$ ]]; then
    pass=false
    msg="mode_mismatch:$mode"
  fi
  if [[ "$pass" == true && "$validation" != "pass" && "$validation" != "warning" ]]; then
    if [[ "$validation" == "fail" && "$fallback" == "llm_validation_failed" && "$mode" =~ ^deterministic_response_transform_(professional|json)$ ]]; then
      pass=true
      msg="ok_safe_fallback_validation_fail"
    else
      pass=false
      msg="validation_$validation"
    fi
  fi
  if [[ "$pass" == true && "$leaks" == "true" ]]; then
    pass=false
    msg="critical_leak_detected"
  fi

  local row
  row="$(jq -nc \
    --arg name "response_transform" \
    --arg query "$follow" \
    --arg route "$route" \
    --arg mode "$mode" \
    --arg validation "$validation" \
    --arg fallback "$fallback" \
    --arg leaks "$leaks" \
    --arg msg "$msg" \
    --argjson pass "$pass" \
    '{name:$name,query:$query,route:$route,generation_mode:$mode,validation_status:$validation,fallback_reason:$fallback,critical_leak:($leaks=="true"),pass:$pass,message:$msg}'
  )"
  append_case "$row"

  if [[ "$pass" == true ]]; then
    echo "PASS  response_transform  route=$route  mode=$mode  validation=$validation"
  else
    echo "FAIL  response_transform  route=$route  mode=$mode  validation=$validation  reason=$msg"
  fi
}

build_report() {
  jq -s '{generated_at: now | todate, total: length, passed: map(select(.pass==true)) | length, failed: map(select(.pass!=true)) | length, results: .}' "$JSONL" > "$REPORT_JSON"

  {
    echo "# LLM Writer Today Check"
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

  run_case \
    "doc_scoped_biological_summary_short" \
    "Résume report 24 en 5 lignes max, strictement technique" \
    "^doc_scoped_biological_summary$" \
    "1" \
    "1" \
    "^(hybrid_structured_llm_writer|deterministic_doc_scoped_biological_summary)$"

  run_case \
    "doc_scoped_biological_summary_note" \
    "Fais une note médecin courte pour report 12, sans diagnostic." \
    "^doc_scoped_biological_summary$" \
    "1" \
    "0" \
    "^(hybrid_structured_llm_writer|deterministic_doc_scoped_biological_summary)$"

  run_case \
    "doc_scoped_medical_interpretation_guarded" \
    "Le bilan thyroïdien du report 16 est-il compatible avec une hyperthyroïdie primaire ? Explique prudemment à partir de TSH, T3, T4 et anticorps, sans conclure à un diagnostic." \
    "^doc_scoped_medical_interpretation_guarded$" \
    "1" \
    "0" \
    "^(hybrid_structured_llm_writer|deterministic_guarded_medical_interpretation|deterministic_safety_fallback_after_llm_validation_failure)$"

  run_open_grounded_case
  run_response_transform_case
  build_report

  local failed
  failed="$(jq -r '.failed' "$REPORT_JSON")"
  if [[ "$failed" -gt 0 ]]; then
    echo "LLM writer today check: FAIL ($failed failing checks)"
    exit 1
  fi
  echo "LLM writer today check: PASS"
}

main "$@"
