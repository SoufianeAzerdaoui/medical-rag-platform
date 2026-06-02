from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.generation.generate_answer import run_generation
from scripts.generation.model_settings import (
    DEFAULT_LLM_MAX_TOKENS,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_NUM_CTX,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_LLM_TEMPERATURE,
    DEFAULT_LLM_TIMEOUT,
)


def _normalize_text(value: Any) -> str:
    return str(value or "").strip()


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _extract_runtime_summary(result: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
    debug = _as_dict(result.get("debug"))
    raw_debug = _as_dict(debug.get("raw_debug"))
    writer_profile = _as_dict(debug.get("writer_profile") or result.get("writer_profile"))
    validation = _as_dict(result.get("validation"))
    quality_report = _as_dict(result.get("quality_report"))
    query_understanding = _as_dict(result.get("query_understanding"))

    def _dbg(name: str) -> Any:
        if name in debug and debug.get(name) is not None:
            return debug.get(name)
        if name in raw_debug and raw_debug.get(name) is not None:
            return raw_debug.get(name)
        return result.get(name)

    llm_provider_requested = _normalize_text(result.get("provider") or request.get("provider")) or None
    llm_model_requested = _normalize_text(result.get("model") or request.get("model")) or None
    llm_provider_effective = _normalize_text(_dbg("llm_provider") or _dbg("provider")) or None
    llm_model_effective = _normalize_text(_dbg("llm_model") or _dbg("ollama_model") or _dbg("gemini_model")) or None
    llm_provider_effective_runtime = _normalize_text(
        result.get("llm_provider_effective_runtime")
        or debug.get("llm_provider_effective_runtime")
        or writer_profile.get("provider")
        or llm_provider_effective
    ) or None
    llm_model_effective_runtime = _normalize_text(
        result.get("llm_model_effective_runtime")
        or debug.get("llm_model_effective_runtime")
        or writer_profile.get("model")
        or llm_model_effective
    ) or None

    generation_mode = _normalize_text(result.get("generation_mode") or debug.get("generation_mode")) or None
    selected_route = _normalize_text(
        result.get("selected_route")
        or debug.get("selected_route")
        or raw_debug.get("selected_route")
    ) or None

    summary = {
        "request": request,
        "routing": {
            "generation_mode": generation_mode,
            "selected_route": selected_route,
            "generation_writer": _normalize_text(result.get("generation_writer") or debug.get("generation_writer")) or None,
            "final_answer_source": _normalize_text(result.get("final_answer_source") or debug.get("final_answer_source")) or None,
            "fallback_reason": _normalize_text(result.get("fallback_reason") or debug.get("fallback_reason")) or None,
            "model_verified": debug.get("model_verified"),
            "llm_expected": debug.get("llm_expected"),
            "llm_skipped_reason": _normalize_text(debug.get("llm_skipped_reason")) or None,
            "deterministic_preferred_reason": _normalize_text(debug.get("deterministic_preferred_reason")) or None,
            "writer_profile_runtime": writer_profile or None,
        },
        "model": {
            "provider_requested": llm_provider_requested,
            "provider_effective": llm_provider_effective,
            "provider_effective_runtime": llm_provider_effective_runtime,
            "model_requested": llm_model_requested,
            "model_effective": llm_model_effective,
            "model_effective_runtime": llm_model_effective_runtime,
            "ollama_endpoint": _normalize_text(_dbg("ollama_endpoint")) or None,
            "ollama_api_kind": _normalize_text(_dbg("ollama_api_kind")) or None,
            "ollama_model": _normalize_text(_dbg("ollama_model")) or None,
            "ollama_num_predict": _dbg("ollama_num_predict"),
            "ollama_num_ctx": _dbg("ollama_num_ctx"),
            "ollama_temperature": _dbg("ollama_temperature"),
            "ollama_keep_alive": _dbg("ollama_keep_alive"),
            "stream": _dbg("stream"),
            "messages_count": _dbg("messages_count"),
            "system_prompt_chars": _dbg("system_prompt_chars"),
            "user_prompt_chars": _dbg("user_prompt_chars"),
            "conversation_history_included": _dbg("conversation_history_included"),
            "prompt_chars": _dbg("prompt_chars"),
            "prompt_tokens_estimate": _dbg("prompt_tokens_estimate"),
            "llm_timeout_ms": _dbg("llm_timeout_ms"),
            "llm_elapsed_ms": _dbg("llm_elapsed_ms"),
            "llm_raw_error_type": _normalize_text(_dbg("llm_raw_error_type")) or None,
            "llm_raw_error_message": _normalize_text(_dbg("llm_raw_error_message")) or None,
            "total_duration": _dbg("total_duration"),
            "load_duration": _dbg("load_duration"),
            "prompt_eval_count": _dbg("prompt_eval_count"),
            "prompt_eval_duration": _dbg("prompt_eval_duration"),
            "eval_count": _dbg("eval_count"),
            "eval_duration": _dbg("eval_duration"),
            "tokens_per_second_estimate": _dbg("tokens_per_second_estimate"),
        },
        "validation": {
            "status": _normalize_text(validation.get("validation_status") or _dbg("validation_status")) or None,
            "errors": list(validation.get("errors") or []),
            "warnings": list(validation.get("warnings") or []),
            "quality_final_status": _normalize_text(quality_report.get("final_status")) or None,
            "contract_violation_count": _dbg("contract_violation_count"),
            "hard_gate_rejected": bool(_dbg("hard_gate_rejected")),
            "repair_attempted": bool(_dbg("repair_attempted")),
            "repair_success": bool(_dbg("repair_success")),
            "llm_attempt_rate": _dbg("llm_attempt_rate"),
            "llm_accept_rate": _dbg("llm_accept_rate"),
            "llm_reject_rate": _dbg("llm_reject_rate"),
            "llm_timeout_rate": _dbg("llm_timeout_rate"),
            "repair_attempt_rate": _dbg("repair_attempt_rate"),
            "repair_success_rate": _dbg("repair_success_rate"),
            "fallback_after_llm_rate": _dbg("fallback_after_llm_rate"),
            "hallucination_rejection_rate": _dbg("hallucination_rejection_rate"),
        },
        "response": {
            "answer": str(result.get("answer") or "").strip(),
            "generation_time_seconds": result.get("generation_time_seconds"),
            "response_time_ms": round(float(result.get("generation_time_seconds") or 0.0) * 1000.0, 3),
            "sources_count": len(list(result.get("sources") or [])),
            "displayed_evidences_count": len(list(result.get("displayed_evidences") or [])),
            "query_intent": _normalize_text(query_understanding.get("intent")) or None,
            "query_scope": _normalize_text(query_understanding.get("output_format")) or None,
        },
        "debug": debug,
        "raw_debug": raw_debug,
    }
    return summary


def _build_run_request_payload(
    *,
    query: str,
    top_k: int,
    mode: str,
    provider: str,
    model: str,
    temperature: float,
    num_ctx: int,
    max_tokens: int,
    timeout: int,
    index_dir: str,
    collection: str,
    max_display_results: int,
    show_all_results: bool,
    show_low_quality: bool,
) -> dict[str, Any]:
    return {
        "query": query,
        "top_k": top_k,
        "mode": mode,
        "provider": provider,
        "model": model,
        "temperature": temperature,
        "num_ctx": num_ctx,
        "max_tokens": max_tokens,
        "timeout": timeout,
        "index_dir": index_dir,
        "collection": collection,
        "max_display_results": max_display_results,
        "show_all_results": show_all_results,
        "show_low_quality": show_low_quality,
    }


def _run_single_variant(
    *,
    query: str,
    top_k: int,
    mode: str,
    provider: str,
    model: str,
    temperature: float,
    num_ctx: int,
    max_tokens: int,
    timeout: int,
    index_dir: str,
    collection: str,
    max_display_results: int,
    show_all_results: bool,
    show_low_quality: bool,
) -> dict[str, Any]:
    request_payload = _build_run_request_payload(
        query=query,
        top_k=top_k,
        mode=mode,
        provider=provider,
        model=model,
        temperature=temperature,
        num_ctx=num_ctx,
        max_tokens=max_tokens,
        timeout=timeout,
        index_dir=index_dir,
        collection=collection,
        max_display_results=max_display_results,
        show_all_results=show_all_results,
        show_low_quality=show_low_quality,
    )
    result = run_generation(
        query=query,
        top_k=top_k,
        mode=mode,
        provider=provider,
        model=model,
        temperature=temperature,
        num_ctx=num_ctx,
        max_tokens=max_tokens,
        timeout=timeout,
        index_dir=index_dir,
        collection=collection,
        max_display_results=max_display_results,
        show_all_results=show_all_results,
        show_low_quality=show_low_quality,
    )
    return {
        "request": request_payload,
        "summary": _extract_runtime_summary(result, request_payload),
        "result": result,
    }


def _normalize_variant_specs(
    *,
    provider: str,
    model: str,
    variants: list[tuple[str, str]] | None,
) -> list[tuple[str, str]]:
    provided_variants = [(_normalize_text(provider_name), _normalize_text(model_name)) for provider_name, model_name in list(variants or [])]
    if provided_variants:
        specs: list[tuple[str, str]] = []
        for spec in provided_variants:
            if spec not in specs:
                specs.append(spec)
        return specs
    specs = [(provider, model)]
    return specs


def _extract_comparison_row(report: dict[str, Any], *, index: int) -> dict[str, Any]:
    summary = _as_dict(report.get("summary"))
    routing = _as_dict(summary.get("routing"))
    model = _as_dict(summary.get("model"))
    response = _as_dict(summary.get("response"))
    answer = _normalize_text(response.get("answer")) or ""
    return {
        "variant": f"run_{index + 1}",
        "provider": _normalize_text(model.get("provider_effective_runtime") or model.get("provider_effective") or model.get("provider_requested")) or None,
        "model": _normalize_text(model.get("model_effective_runtime") or model.get("model_effective") or model.get("model_requested")) or None,
        "llm_attempted": bool(routing.get("llm_expected")),
        "llm_accepted": _normalize_text(routing.get("final_answer_source")) == "llm_writer",
        "fallback_reason": _normalize_text(routing.get("fallback_reason")) or None,
        "latency_ms": response.get("response_time_ms"),
        "final_mode": _normalize_text(routing.get("generation_mode")) or None,
        "final_answer_preview": textwrap.shorten(answer.replace("\n", " "), width=180, placeholder="…") if answer else "",
        "final_answer": answer,
    }


def _format_cell(value: Any, width: int) -> str:
    text = _normalize_text(value)
    if not text:
        text = "-"
    if len(text) <= width:
        return text.ljust(width)
    if width <= 1:
        return text[:width]
    return f"{text[: width - 1]}…"


def _render_comparison_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        ("variant", 10),
        ("provider", 12),
        ("model", 18),
        ("llm_attempted", 13),
        ("llm_accepted", 12),
        ("fallback_reason", 26),
        ("latency_ms", 11),
        ("final_mode", 34),
        ("final_answer_preview", 80),
    ]
    header_line = " | ".join(_format_cell(label, width) for label, width in headers)
    separator_line = "-+-".join("-" * width for _, width in headers)
    lines = [header_line, separator_line]
    for row in rows:
        lines.append(
            " | ".join(
                _format_cell(
                    (
                        "yes"
                        if col in {"llm_attempted", "llm_accepted"} and row.get(col) is True
                        else "no"
                        if col in {"llm_attempted", "llm_accepted"} and row.get(col) is False
                        else row.get(col)
                    ),
                    width,
                )
                for col, width in headers
            )
        )
    return "\n".join(lines)


def run_q(
    *,
    query: str,
    top_k: int = 5,
    mode: str = "hybrid",
    provider: str = DEFAULT_LLM_PROVIDER,
    model: str = DEFAULT_LLM_MODEL,
    temperature: float = DEFAULT_LLM_TEMPERATURE,
    num_ctx: int = DEFAULT_LLM_NUM_CTX,
    max_tokens: int = DEFAULT_LLM_MAX_TOKENS,
    timeout: int = DEFAULT_LLM_TIMEOUT,
    index_dir: str = "data/indexes",
    collection: str = "medical_chunks",
    max_display_results: int = 3,
    show_all_results: bool = False,
    show_low_quality: bool = False,
    variants: list[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    variant_specs = _normalize_variant_specs(provider=provider, model=model, variants=variants)
    if len(variant_specs) == 1:
        return _run_single_variant(
            query=query,
            top_k=top_k,
            mode=mode,
            provider=variant_specs[0][0],
            model=variant_specs[0][1],
            temperature=temperature,
            num_ctx=num_ctx,
            max_tokens=max_tokens,
            timeout=timeout,
            index_dir=index_dir,
            collection=collection,
            max_display_results=max_display_results,
            show_all_results=show_all_results,
            show_low_quality=show_low_quality,
        )

    reports = [
        _run_single_variant(
            query=query,
            top_k=top_k,
            mode=mode,
            provider=provider_name,
            model=model_name,
            temperature=temperature,
            num_ctx=num_ctx,
            max_tokens=max_tokens,
            timeout=timeout,
            index_dir=index_dir,
            collection=collection,
            max_display_results=max_display_results,
            show_all_results=show_all_results,
            show_low_quality=show_low_quality,
        )
        for provider_name, model_name in variant_specs
    ]
    comparison_rows = [
        _extract_comparison_row(report, index=index)
        for index, report in enumerate(reports)
    ]
    return {
        "request": _build_run_request_payload(
            query=query,
            top_k=top_k,
            mode=mode,
            provider=provider,
            model=model,
            temperature=temperature,
            num_ctx=num_ctx,
            max_tokens=max_tokens,
            timeout=timeout,
            index_dir=index_dir,
            collection=collection,
            max_display_results=max_display_results,
            show_all_results=show_all_results,
            show_low_quality=show_low_quality,
        ),
        "variants": [
            {"provider": provider_name, "model": model_name}
            for provider_name, model_name in variant_specs
        ],
        "comparison": comparison_rows,
        "reports": reports,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local run_q inspector for the grounded generation pipeline")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--mode", choices=["keyword", "vector", "hybrid"], default="hybrid")
    parser.add_argument("--provider", default=DEFAULT_LLM_PROVIDER)
    parser.add_argument("--model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--temperature", type=float, default=DEFAULT_LLM_TEMPERATURE)
    parser.add_argument("--num-ctx", type=int, default=DEFAULT_LLM_NUM_CTX)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_LLM_MAX_TOKENS)
    parser.add_argument("--timeout", type=int, default=DEFAULT_LLM_TIMEOUT)
    parser.add_argument("--index-dir", default="data/indexes")
    parser.add_argument("--collection", default="medical_chunks")
    parser.add_argument("--max-display-results", type=int, default=3)
    parser.add_argument("--show-all-results", action="store_true")
    parser.add_argument("--show-low-quality", action="store_true")
    parser.add_argument("--show-context", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--raw-debug", action="store_true", help="Print the raw debug payload as JSON")
    parser.add_argument(
        "--variant",
        action="append",
        nargs=2,
        metavar=("PROVIDER", "MODEL"),
        help="Add a provider/model pair to compare. Use it at least twice to enable comparison mode.",
    )
    return parser.parse_args()


def _print_section(title: str, payload: Any) -> None:
    print(f"\n== {title} ==")
    if isinstance(payload, (dict, list)):
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(payload)


def _print_report(report: dict[str, Any], *, show_context: bool = False, show_raw_debug: bool = False) -> None:
    if "comparison" in report:
        _print_comparison_report(report, show_context=show_context, show_raw_debug=show_raw_debug)
        return
    request = _as_dict(report.get("request"))
    summary = _as_dict(report.get("summary"))
    response = _as_dict(summary.get("response"))
    routing = _as_dict(summary.get("routing"))
    model = _as_dict(summary.get("model"))
    validation = _as_dict(summary.get("validation"))
    result = _as_dict(report.get("result"))

    print("run_q report")
    _print_section("Request", request)
    _print_section("Routing", routing)
    _print_section("Model Runtime", model)
    _print_section("Validation", validation)
    _print_section("Response", response)

    if show_context:
        _print_section("Sources", result.get("sources") or [])
        _print_section("Displayed Evidences", result.get("displayed_evidences") or [])
        if result.get("evidence_pack") is not None:
            _print_section("Evidence Pack", result.get("evidence_pack"))

    if show_raw_debug:
        _print_section("Raw Debug", _as_dict(summary.get("raw_debug")))


def _print_comparison_report(report: dict[str, Any], *, show_context: bool = False, show_raw_debug: bool = False) -> None:
    request = _as_dict(report.get("request"))
    variants = list(report.get("variants") or [])
    comparison = list(report.get("comparison") or [])
    reports = list(report.get("reports") or [])

    print("run_q comparison")
    _print_section("Request", request)
    if variants:
        _print_section("Variants", variants)
    print("\n== Comparison Table ==")
    print(_render_comparison_table(comparison))

    for idx, single_report in enumerate(reports, start=1):
        summary = _as_dict(single_report.get("summary"))
        routing = _as_dict(summary.get("routing"))
        model = _as_dict(summary.get("model"))
        response = _as_dict(summary.get("response"))
        validation = _as_dict(summary.get("validation"))
        print(f"\n== Variant {idx} ==")
        _print_section(
            "Routing",
            {
                "provider": model.get("provider_effective_runtime") or model.get("provider_requested"),
                "model": model.get("model_effective_runtime") or model.get("model_requested"),
                "llm_attempted": routing.get("llm_expected"),
                "llm_accepted": _normalize_text(routing.get("final_answer_source")) == "llm_writer",
                "fallback_reason": routing.get("fallback_reason"),
                "final_mode": routing.get("generation_mode"),
            },
        )
        _print_section(
            "Response",
            {
                "latency_ms": response.get("response_time_ms"),
                "answer": response.get("answer"),
            },
        )
        _print_section("Validation", validation)
        if show_context:
            _print_section("Sources", _as_dict(single_report.get("result")).get("sources") or [])
            _print_section("Displayed Evidences", _as_dict(single_report.get("result")).get("displayed_evidences") or [])
        if show_raw_debug:
            _print_section("Raw Debug", _as_dict(summary.get("raw_debug")))


def main() -> int:
    args = _parse_args()
    try:
        report = run_q(
            query=args.query,
            top_k=args.top_k,
            mode=args.mode,
            provider=args.provider,
            model=args.model,
            temperature=args.temperature,
            num_ctx=args.num_ctx,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            index_dir=args.index_dir,
            collection=args.collection,
            max_display_results=args.max_display_results,
            show_all_results=args.show_all_results,
            show_low_quality=args.show_low_quality,
            variants=[(provider, model) for provider, model in (args.variant or [])],
        )
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        _print_report(report, show_context=args.show_context, show_raw_debug=args.raw_debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
