# LLM Observability Dashboard

This project now emits structured `chat_request_summary` logs with LLM routing and fallback signals.

## Dashboard asset

Import this Grafana dashboard:

- [config/grafana/medical_rag_llm_observability_dashboard.json](/home/onizuka/Bureau/PFE/medical-rag-platform/config/grafana/medical_rag_llm_observability_dashboard.json)

## Assumption

The dashboard queries are written for `Grafana + Loki` and assume backend logs are labeled with:

- `app="medical-rag-backend"`

If your Loki label differs, change the `app_label` dashboard variable after import.

## Signals exposed

Each `chat_request_summary` log includes:

- `llm_route_class`
- `llm_prompt_policy_version`
- `llm_attempt_rate`
- `llm_accept_rate`
- `fallback_after_llm_rate`
- `contract_violation_count`
- `validation_hard_gate_reasons`
- `validation_hard_gate_reason`
- `validation_hard_gate_reason_count`
- `response_time_ms`
- `failure_signals`
- `fallback_kind`

These fields are emitted per request as `0.0/1.0` rates so they can be aggregated directly with `avg_over_time(...)`.

## Recommended alerts

Use these thresholds as a starting point:

1. `fallback_after_llm_rate > 0.20` over `15m`
2. `llm_accept_rate < 0.70` over `15m` for `llm_route_class="llm_allowed"`
3. `p95 response_time_ms > 3000` over `15m`
4. `contract_violation_count > 0` over `15m`
5. `validation_hard_gate_reason != ""` over `15m`
6. Any `failure_signals` containing `hallucination`, `diagnosis`, `treatment`, or `pii`

Provisionable Grafana rule asset:

- [config/grafana/alerts/medical_rag_llm_alerts.yml](/home/onizuka/Bureau/PFE/medical-rag-platform/config/grafana/alerts/medical_rag_llm_alerts.yml)

The provided rule triggers when the summed `contract_violation_count` over the last `15m` stays above `0` for `5m`.
It also includes a rule that triggers when `validation_hard_gate_reason` is non-empty over the last `15m` for at least `5m`.

## Route class interpretation

- `llm_allowed`: route explicitly allowed to use the LLM writer
- `deterministic_only`: route must stay deterministic
- `deterministic_preferred`: deterministic-first route; LLM should not be the normal path
- `safety_only`: refusal/safety route

## Operational reading

Healthy pre-production behavior should usually look like this:

1. `llm_attempt_rate` high only on `llm_allowed` routes
2. `llm_accept_rate` stable and materially above fallback noise
3. `fallback_after_llm_rate` low and not trending upward
4. `contract_violation_count` equal to `0` in normal traffic
5. `validation_hard_gate_reason` empty in normal traffic
6. `failure_signals` empty in normal traffic

## Hard gate trend panel

The dashboard includes `Top Hard Gate Reason Trends (5m rolling)`:

- query logic: `topk(8, sum by (validation_hard_gate_reason) (count_over_time(... [5m])))`
- goal: visualize short-term drift of the most frequent hard-gate codes
- expected baseline: near zero in stable pre-production traffic
