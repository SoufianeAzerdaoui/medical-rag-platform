# LLM Go/No-Go Report

- Generated at: `2026-05-25T23:59:59.169257+00:00`
- Overall: **NO-GO**

## Gates
- zero_hallucination_final_accepted: **OK**
- zero_diagnosis_leak: **OK**
- zero_treatment_leak: **OK**
- zero_pii_leak: **OK**
- llm_timeout_rate_low: **KO**
- fallback_after_llm_rate_acceptable: **KO**
- llm_accept_rate_useful_on_allowed_routes: **KO**
- p95_latency_compatible_ux: **KO**
- p95_llm_writer_latency_compatible_ux: **KO**
- llm_writer_calls_professional: **OK**
- system_better_with_llm_on_allowed_routes: **OK**

## Core Metrics
- llm_accept_rate: `0.5`
- llm_timeout_rate: `0.5`
- fallback_after_llm_rate: `0.5`
- professional_llm_accept_rate: `1.0`
- p95_response_time_ms: `90364.0`
- p95_llm_writer_ms: `90090.937`

## LLM vs Baseline
- llm_on_avg_score: `7.0`
- llm_off_avg_score: `7.0`
- score_delta: `0.0`
