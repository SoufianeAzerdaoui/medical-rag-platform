# LLM Go/No-Go Report

- Generated at: `2026-05-26T00:29:06.528552+00:00`
- Overall: **NO-GO**

## Gates
- zero_hallucination_final_accepted: **OK**
- zero_diagnosis_leak: **OK**
- zero_treatment_leak: **OK**
- zero_pii_leak: **OK**
- llm_timeout_rate_low: **OK**
- fallback_after_llm_rate_acceptable: **OK**
- llm_accept_rate_useful_on_allowed_routes: **KO**
- p95_latency_compatible_ux: **KO**
- p95_llm_writer_latency_compatible_ux: **OK**
- llm_writer_calls_professional: **KO**
- system_better_with_llm_on_allowed_routes: **OK**

## Core Metrics
- llm_accept_rate: `0.0`
- llm_timeout_rate: `0.0`
- fallback_after_llm_rate: `0.0`
- professional_llm_accept_rate: `0.0`
- p95_response_time_ms: `15523.0`
- p95_llm_writer_ms: `0.0`

## LLM vs Baseline
- llm_on_avg_score: `0.0`
- llm_off_avg_score: `0.0`
- score_delta: `0.0`
