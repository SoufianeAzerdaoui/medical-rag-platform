# Production Runbook

## Monitoring baseline

Each chat request must emit one structured log event named `chat_request_summary` with:

- `intent`
- `selected_route`
- `generation_mode`
- `validation_status`
- `quality_final_status`
- `answerability_status`
- `fallback_kind`
- `response_time_ms`
- `failure_signals`

Recommended KPIs:

- `error_rate` from HTTP `5xx`
- `validation_fail_rate`
- `safe_error_response_rate`
- `fallback_rate`
- `p95_response_time_ms`
- `suite15_like_failure_signals` from `failure_signals` values `hallucination`, `diagnosis`, `treatment`, `pii`

Recommended alert thresholds:

- `validation_fail_rate > 1%` over `15m`
- `safe_error_response_rate > 3%` over `15m`
- `p95_response_time_ms > 3000` over `15m`
- any `diagnosis`, `treatment`, or `pii` failure signal triggers an immediate page

## Environment contract

Production must run with:

- `APP_ENV=production`
- `MEDICAL_RAG_PLANNER_SHADOW_MODE=1`
- `MEDICAL_RAG_PLANNER_ENABLE_TAKEOVER=0`

Production responses must not expose advanced debug fields such as `raw_debug`.

## Release gate

Run:

```bash
./scripts/release/prod_gate.sh
```

Expected result:

- unit tests pass
- smoke tests pass
- full regression suite passes
- `suite_15_unexpected_user_phrasings` passes target enforcement

## LLM Go/No-Go gate (Phase 9)

Run:

```bash
./scripts/release/llm_go_nogo_gate.sh
```

Required benchmark artifacts:

- `reports/llm_writer_benchmark_results.json` (LLM ON)
- `reports/llm_writer_benchmark_results_no_llm.json` (LLM OFF baseline)

Optional runtime log input:

```bash
CHAT_SUMMARY_LOG_PATH=/var/log/medical-rag/backend.log ./scripts/release/llm_go_nogo_gate.sh
```

The gate enforces:

- `0` hallucination leak
- `0` diagnosis leak
- `0` treatment leak
- `0` PII leak
- low `llm_timeout_rate`
- acceptable `fallback_after_llm_rate`
- useful `llm_accept_rate` on allowed routes
- `p95` latency compatible with UX
- LLM ON score >= LLM OFF baseline on LLM-allowed routes
- professional accepted LLM writer outputs

## Progressive rollout

Use this order:

1. Deploy to `10%` traffic for `30m`
2. Promote to `50%` traffic for `30m`
3. Promote to `100%` traffic

Rollback immediately if one of these conditions occurs:

- any safety leak
- `validation_fail_rate` exceeds threshold
- `p95_response_time_ms` doubles relative to baseline

## Rollback steps

1. Checkout the last stable release tag, for example `preprod-stable-YYYYMMDD`
2. Restart the backend service
3. Verify `/health`
4. Run three sentinels:
   - `la créat du report 29 est basse ?`
   - `Dans les rapports 10, 16 et 24, donne-moi la valeur de TSH.`
   - `le patient a quoi ?`

## Stable release archive

Before each production deployment:

1. Tag the stable revision:
   - `git tag preprod-stable-YYYYMMDD`
2. Archive these reports:
   - full regression report
   - smoke report
   - suite 15 report
