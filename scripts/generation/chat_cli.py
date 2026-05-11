from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Reduce noisy third-party progress/log output in normal CLI mode.
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from generate_answer import run_generation
from llm_client import LLMClient
from retrieval.search import SearchEngine
from query_understanding import parse_query_understanding


@dataclass
class ChatState:
    provider: str
    model: str
    mode: str
    top_k: int
    temperature: float
    num_ctx: int
    max_tokens: int
    timeout: int
    index_dir: str
    collection: str
    max_display_results: int
    show_all_results: bool
    show_low_quality: bool
    show_context: bool = False
    json_output: bool = False
    debug: bool = False


@dataclass
class ConversationState:
    last_user_question: str = ""
    last_query_understanding: dict[str, Any] | None = None
    last_evidence_pack: dict[str, Any] | None = None
    last_answer: str = ""
    last_output_format: str = ""
    last_rendered_rows: list[dict[str, Any]] | None = None
    last_sources: list[dict[str, Any]] | None = None
    last_visualization: dict[str, Any] | None = None
    last_intent: str = ""
    last_requested_columns: list[str] | None = None


def _startup_banner(state: ChatState) -> None:
    print("Medical RAG Assistant - Terminal Mode")
    print(f"Provider: {state.provider}")
    print(f"Model: {state.model}")
    print(f"Mode retrieval: {state.mode}")
    print("Tapez votre question médicale.")
    print("Commandes disponibles:")
    print("  /exit ou /quit : quitter")
    print("  /help : afficher l’aide")
    print("  /context on : afficher les chunks/evidence utilisés")
    print("  /context off : masquer les chunks/evidence")
    print("  /json on : afficher la sortie JSON")
    print("  /json off : afficher la sortie texte normale")
    print("  /settings : afficher les paramètres courants")
    print("  /clear : effacer l’écran")


def _print_help() -> None:
    print("Aide")
    print("- Posez une question médicale en français.")
    print("- /exit ou /quit : quitter la session.")
    print("- /help : afficher cette aide.")
    print("- /context on|off : afficher/masquer l’evidence pack.")
    print("- /json on|off : afficher/masquer la sortie JSON complète.")
    print("- /settings : afficher les paramètres actifs.")
    print("- /clear : effacer l’écran.")


def _print_settings(state: ChatState) -> None:
    print("Paramètres courants")
    print(f"- provider: {state.provider}")
    print(f"- model: {state.model}")
    print(f"- mode: {state.mode}")
    print(f"- top_k: {state.top_k}")
    print(f"- temperature: {state.temperature}")
    print(f"- num_ctx: {state.num_ctx}")
    print(f"- max_tokens: {state.max_tokens}")
    print(f"- timeout: {state.timeout}")
    print(f"- index_dir: {state.index_dir}")
    print(f"- collection: {state.collection}")
    print(f"- max_display_results: {state.max_display_results}")
    print(f"- show_all_results: {state.show_all_results}")
    print(f"- show_low_quality: {state.show_low_quality}")
    print(f"- context: {'on' if state.show_context else 'off'}")
    print(f"- json: {'on' if state.json_output else 'off'}")


def _print_evidence(evidence_pack: list[dict[str, Any]]) -> None:
    print("\nEvidences utilisées :")
    if not evidence_pack:
        print("- (aucune evidence)")
        return
    for idx, ev in enumerate(evidence_pack, start=1):
        print(f"- rank: {ev.get('rank') or idx}")
        print(f"  chunk_id: {ev.get('chunk_id')}")
        print(f"  doc_id: {ev.get('doc_id')}")
        print(f"  chunk_type: {ev.get('chunk_type')}")
        print(f"  analyte: {ev.get('analyte_display') or ev.get('analyte') or ev.get('parameter')}")
        print(f"  value_raw: {ev.get('value_raw')}")
        print(f"  unit: {ev.get('unit')}")
        print(f"  reference_range: {ev.get('reference_range')}")
        print(f"  previous_result: {ev.get('previous_result')}")
        print(f"  source_kind: {ev.get('source_kind')}")
        print(f"  page_number: {ev.get('page_number')}")
        print(f"  row_index: {ev.get('row_index')}")
        print(f"  source: {ev.get('source')}")
        print(f"  final_score: {ev.get('final_score')}")
        print(f"  clinical_rerank_score: {ev.get('clinical_rerank_score')}")
        print(f"  evidence_display_quality: {ev.get('evidence_display_quality')}")


def _print_human_result(query: str, result: dict[str, Any], show_context: bool, debug: bool) -> None:
    validation = result.get("validation") or {}
    raw_warnings = validation.get("warnings") or []
    generation_mode = str(result.get("generation_mode") or "")
    llm_error = str(result.get("llm_error") or "").strip()

    # Keep normal mode clean: hide internal fallback diagnostics unless debug is enabled.
    warnings = list(raw_warnings)
    if not debug:
        warnings = [
            w
            for w in warnings
            if not (
                str(w).startswith("llm_fallback_used:")
                or str(w).startswith("llm_used_for_structured_query")
                or str(w).startswith("controlled_warning_missing_requested_analytes")
                or str(w).startswith("missing_requested_analyte:")
                or "sensitive query should generally return anonymized" in str(w).lower()
            )
        ]

    print("\nRéponse :")
    print(result.get("answer") or "")

    if debug:
        print("\nDebug :")
        print(query)
        print(f"- request_id: {result.get('request_id')}")
        print(f"- query_received: {result.get('query_received')}")
        print(f"- query_used_for_retrieval: {result.get('query_used_for_retrieval')}")
        print(f"- query_used_for_prompt: {result.get('query_used_for_prompt')}")
        print(f"- detected_analytes: {result.get('detected_analytes')}")
        print(f"- generation_mode: {result.get('generation_mode')}")
        print("\nValidation :")
        print(f"- status : {validation.get('validation_status')}")
        print(f"- pii_leak_detected : {str(bool(validation.get('pii_leak_detected'))).lower()}")
        print(f"- warnings : {len(warnings)}")
        if warnings:
            for w in warnings:
                print(f"  - {w}")
    elif warnings:
        print("\nAvertissements validation :")
        for w in warnings:
            print(f"- {w}")

    show_llm_error = bool(llm_error)
    if not debug and generation_mode in {
        "llm_writer_error_fallback",
        "llm_writer_format_fallback",
        "llm_writer_quality_fallback",
        "deterministic_structured_renderer",
        "llm_error_fallback_template",
    }:
        # Fallback already produced a valid grounded answer; don't surface low-level LLM errors in normal mode.
        show_llm_error = False

    if show_llm_error:
        print("\nErreur LLM :")
        print(f"- {llm_error}")

    if show_context:
        _print_evidence(result.get("evidence_pack") or [])


def _clear_screen() -> None:
    cmd = "cls" if os.name == "nt" else "clear"
    try:
        os.system(cmd)
    except Exception:
        pass


@contextlib.contextmanager
def _suppress_stdout_stderr(enabled: bool):
    if not enabled:
        yield
        return
    with open(os.devnull, "w", encoding="utf-8") as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
        yield


def _handle_command(raw: str, state: ChatState) -> bool:
    cmd = raw.strip().lower()
    if cmd in {"/exit", "/quit"}:
        print("Session terminée.")
        return False
    if cmd == "/help":
        _print_help()
        return True
    if cmd == "/settings":
        _print_settings(state)
        return True
    if cmd == "/context on":
        state.show_context = True
        print("Affichage du contexte activé.")
        return True
    if cmd == "/context off":
        state.show_context = False
        print("Affichage du contexte désactivé.")
        return True
    if cmd == "/json on":
        state.json_output = True
        print("Sortie JSON activée.")
        return True
    if cmd == "/json off":
        state.json_output = False
        print("Sortie JSON désactivée.")
        return True
    if cmd == "/clear":
        _clear_screen()
        _startup_banner(state)
        return True

    print("Commande inconnue. Tapez /help pour l’aide.")
    return True


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive terminal chat for Medical RAG generation")
    parser.add_argument("--provider", default="ollama")
    parser.add_argument("--model", default="qwen3:4b")
    parser.add_argument("--mode", default="hybrid", choices=["keyword", "vector", "hybrid"])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-ctx", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--index-dir", default="data/indexes")
    parser.add_argument("--collection", default="medical_chunks")
    parser.add_argument("--max-display-results", type=int, default=3)
    parser.add_argument("--show-all-results", action="store_true")
    parser.add_argument("--show-low-quality", action="store_true")
    parser.add_argument("--show-context", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    state = ChatState(
        provider=args.provider,
        model=args.model,
        mode=args.mode,
        top_k=args.top_k,
        temperature=args.temperature,
        num_ctx=args.num_ctx,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        index_dir=args.index_dir,
        collection=args.collection,
        max_display_results=args.max_display_results,
        show_all_results=args.show_all_results,
        show_low_quality=args.show_low_quality,
        show_context=args.show_context,
        json_output=args.json,
        debug=args.debug,
    )

    _startup_banner(state)
    index_dir = Path(state.index_dir)
    sqlite_path = index_dir / "medical_rag.sqlite"
    qdrant_dir = index_dir / "qdrant"
    search_engine = SearchEngine(
        sqlite_path=sqlite_path,
        qdrant_dir=qdrant_dir,
        collection=state.collection,
    )
    llm_client = LLMClient(provider=state.provider)
    conversation_state = ConversationState()

    try:
        while True:
            try:
                raw = input("\n> ")
            except EOFError:
                print("\nSession terminée (EOF).")
                return 0
            except KeyboardInterrupt:
                print("\nInterruption détectée. Session terminée.")
                return 0

            question = raw.strip()
            if not question:
                continue

            if question.startswith("/"):
                keep_running = _handle_command(question, state)
                if not keep_running:
                    return 0
                continue

            print("Génération en cours...")
            try:
                qu = parse_query_understanding(question)
                with _suppress_stdout_stderr(not state.debug):
                    result = run_generation(
                        query=question,
                        top_k=state.top_k,
                        mode=state.mode,
                        provider=state.provider,
                        model=state.model,
                        temperature=state.temperature,
                        num_ctx=state.num_ctx,
                        max_tokens=state.max_tokens,
                        timeout=state.timeout,
                        index_dir=state.index_dir,
                        collection=state.collection,
                        search_engine=search_engine,
                        llm_client=llm_client,
                        max_display_results=state.max_display_results,
                        show_all_results=state.show_all_results,
                        show_low_quality=state.show_low_quality,
                        previous_structured_evidence_pack=conversation_state.last_evidence_pack if qu.intent == "response_transform" else None,
                    )
            except KeyboardInterrupt:
                print("\nInterruption détectée pendant la génération.")
                continue
            except Exception as exc:
                print(f"Erreur génération: {exc}")
                if "Ollama service unavailable" in str(exc):
                    print("Ollama indisponible. Vérifiez: systemctl status ollama")
                continue

            if state.json_output:
                print(json.dumps(result, ensure_ascii=False, indent=2))
            else:
                _print_human_result(question, result, state.show_context, state.debug)

            if str((result.get("validation") or {}).get("validation_status") or "").lower() in {"pass", "warning"}:
                conversation_state.last_user_question = question
                conversation_state.last_query_understanding = result.get("query_understanding") or {}
                conversation_state.last_evidence_pack = result.get("structured_evidence_pack") or {}
                conversation_state.last_answer = str(result.get("answer") or "")
                conversation_state.last_output_format = str((result.get("query_understanding") or {}).get("output_format") or "")
                conversation_state.last_rendered_rows = list((result.get("structured_evidence_pack") or {}).get("evidences") or [])
                conversation_state.last_sources = list(result.get("sources") or [])
                conversation_state.last_visualization = dict(result.get("visualization") or {}) if isinstance(
                    result.get("visualization"), dict
                ) else None
                conversation_state.last_intent = str((result.get("query_understanding") or {}).get("intent") or "")
                conversation_state.last_requested_columns = list((result.get("query_understanding") or {}).get("requested_table_columns") or [])

            llm_error = str(result.get("llm_error") or "")
            generation_mode = str(result.get("generation_mode") or "")
            if (
                "Ollama service unavailable" in llm_error
                and (
                    state.debug
                    or generation_mode
                    not in {
                        "llm_writer_error_fallback",
                        "llm_writer_format_fallback",
                        "llm_writer_quality_fallback",
                        "deterministic_structured_renderer",
                        "llm_error_fallback_template",
                    }
                )
            ):
                print("Ollama indisponible. Vérifiez: systemctl status ollama")
    finally:
        search_engine.close()


if __name__ == "__main__":
    raise SystemExit(main())
