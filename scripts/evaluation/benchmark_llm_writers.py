from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib import error, request

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_OUTPUT_DIR = Path("reports")
DEFAULT_MODELS = [
    "llama3.2:latest",
    "qwen2.5:7b-instruct",
    "mistral:7b-instruct-q4_0",
    "gemma3:4b",
]
DEFAULT_QUESTIONS = [
    ("Q1", "Fais une synthèse médico-biologique du report 12 en 6 lignes maximum, en séparant les anomalies et les résultats rassurants. Ne donne pas de diagnostic."),
    ("Q2", "Résume le report 24 comme une note courte pour un médecin, en restant strictement descriptif et sans diagnostic."),
    ("Q3", "Dans le report 10, explique les anomalies biologiques les plus importantes par priorité technique, avec une justification courte pour chaque anomalie. Ne pose pas de diagnostic."),
    ("Q4", "Le bilan thyroïdien du report 16 est-il compatible avec une hyperthyroïdie primaire ? Explique prudemment à partir de TSH, T3, T4 et anticorps, sans conclure à un diagnostic."),
    ("Q5", "Résume les résultats de pharmacotoxicologie urinaire du report 27 en distinguant les résultats sous seuil et ceux au-dessus du seuil. Ne donne aucune interprétation clinique."),
]

CSV_FIELDS = [
    "model",
    "benchmark_model_requested",
    "question_id",
    "question",
    "answer",
    "generation_mode",
    "generation_writer",
    "validation_status",
    "quality_final_status",
    "fallback_reason",
    "selected_route",
    "llm_provider",
    "llm_model_requested",
    "llm_model_effective",
    "ollama_model",
    "model_verified",
    "llm_model_override_applied",
    "llm_writer_attempted",
    "llm_writer_accepted",
    "hard_gate_rejected",
    "repair_attempted",
    "repair_success",
    "llm_candidate_validation_status",
    "llm_candidate_validation_errors",
    "llm_candidate_validation_warnings",
    "validation_errors",
    "validation_warnings",
    "llm_writer_ms",
    "response_time",
    "displayed_count",
    "sources_count",
    "score",
    "debug_model_match",
    "error",
]

DIAGNOSTIC_PATTERNS = re.compile(
    r"\bdiagnos|\bdiagnostique|compatible avec|est compatible|est une|est un|confirme|conclu|conclusion|probable|vraisemblablement|suggère",
    flags=re.IGNORECASE,
)
TREATMENT_PATTERNS = re.compile(
    r"\btraitement|médicament|medicament|prescrire|posologie|dose|traiter|soin|hospitalisation|antibi|anticoagulant|cortico|kinésith|chirurgie",
    flags=re.IGNORECASE,
)


def _http_json(*, method: str, url: str, token: str, payload: dict[str, Any] | None, timeout_s: int = 180) -> dict[str, Any]:
    body = None
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {token}",
    }
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = request.Request(url=url, data=body, headers=headers, method=method.upper())
    try:
        with request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw.strip() else {}
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} on {url}: {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Network error on {url}: {exc}") from exc


def _ensure_conversation_id(*, base_url: str, token: str, conv_id_env: str | None) -> str:
    if str(conv_id_env or "").strip():
        return str(conv_id_env).strip()
    created = _http_json(
        method="POST",
        url=f"{base_url.rstrip('/')}/conversations",
        token=token,
        payload={"title": "LLM writer benchmark"},
    )
    conv_id = str(created.get("id") or "").strip()
    if not conv_id:
        raise RuntimeError("Failed to create conversation: missing id in /conversations response.")
    return conv_id


def _get_installed_ollama_models() -> set[str] | None:
    ollama_exe = shutil.which("ollama")
    if not ollama_exe:
        return None
    try:
        proc = subprocess.run(
            [ollama_exe, "list"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        return None
    except FileNotFoundError:
        return None

    if proc.returncode != 0:
        return None

    models: set[str] = set()
    for line in proc.stdout.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("NAME"):
            continue
        parts = re.split(r"\s{2,}", stripped)
        if parts:
            models.add(parts[0].strip())
    return models


def _model_installed(model: str) -> bool | None:
    installed_models = _get_installed_ollama_models()
    if installed_models is None:
        return None
    return model in installed_models


def _normalize_string(value: Any) -> str:
    return str(value or "").strip()


def _extract_response_fields(model: str, question_id: str, question: str, response: dict[str, Any]) -> dict[str, Any]:
    debug = dict(response.get("debug") or {})
    validation = dict(response.get("validation") or {})
    debug_validation = dict(debug.get("validation") or {})
    score = 0
    validation_status = _normalize_string(response.get("validation_status") or validation.get("validation_status"))
    fallback_reason = _normalize_string(debug.get("fallback_reason")) or None
    llm_candidate_validation_errors = debug.get("llm_candidate_validation_errors")
    quality_final_status = _normalize_string((response.get("quality_report") or {}).get("final_status"))
    generation_writer = _normalize_string(response.get("generation_writer") or debug.get("generation_writer"))
    answer = str(response.get("answer") or "").strip()
    answer_norm = answer.lower()

    if validation_status == "pass":
        score += 2
    if not fallback_reason:
        score += 2
    if not llm_candidate_validation_errors:
        score += 2
    if quality_final_status == "pass":
        score += 1
    if generation_writer == "llm_writer":
        score += 1
    if not DIAGNOSTIC_PATTERNS.search(answer):
        score += 1
    if not TREATMENT_PATTERNS.search(answer):
        score += 1

    llm_candidate_validation_status = _normalize_string(debug.get("llm_candidate_validation_status")) or None
    llm_candidate_validation_warnings = list(debug.get("llm_candidate_validation_warnings") or [])
    validation_errors = list(debug_validation.get("errors") or [])
    validation_warnings = list(debug_validation.get("warnings") or [])
    displayed_evidences = list(response.get("displayed_evidences") or [])
    sources = list(response.get("sources") or [])
    llm_provider = _normalize_string(debug.get("llm_provider")) or None
    llm_model_override_applied = bool(debug.get("llm_model_override_applied"))
    llm_model_requested = _normalize_string(debug.get("llm_model_requested")) or None
    llm_model_effective = _normalize_string(debug.get("llm_model_effective")) or None
    ollama_model = _normalize_string(debug.get("ollama_model")) or None
    debug_model_match = bool(ollama_model and ollama_model == model)
    model_verified: bool = bool(
        llm_model_override_applied
        and llm_model_requested
        and llm_model_effective
        and llm_model_requested == llm_model_effective
        and llm_model_effective == model
        and debug_model_match
    )
    llm_writer_attempted = bool(debug.get("llm_writer_attempted") or debug.get("llm_writer_allowed") or debug.get("llm_writer_used"))
    llm_writer_accepted = str(generation_writer).strip().lower() == "llm_writer"
    hard_gate_rejected = bool(debug.get("hard_gate_rejected") or debug.get("hard_gate_triggered"))
    repair_attempted = bool(debug.get("repair_attempted") or debug.get("llm_repair_attempted") or debug.get("llm_candidate_repair_used"))
    repair_success = repair_attempted and not hard_gate_rejected and llm_writer_accepted

    return {
        "model": model,
        "benchmark_model_requested": model,
        "question_id": question_id,
        "question": question,
        "answer": answer,
        "generation_mode": _normalize_string(response.get("generation_mode")) or None,
        "generation_writer": generation_writer or None,
        "validation_status": validation_status or None,
        "quality_final_status": quality_final_status or None,
        "fallback_reason": fallback_reason,
        "selected_route": _normalize_string(debug.get("selected_route")) or None,
        "llm_provider": llm_provider,
        "llm_model_requested": llm_model_requested,
        "llm_model_effective": llm_model_effective,
        "ollama_model": ollama_model,
        "model_verified": model_verified,
        "llm_model_override_applied": llm_model_override_applied,
        "llm_writer_attempted": llm_writer_attempted,
        "llm_writer_accepted": llm_writer_accepted,
        "hard_gate_rejected": hard_gate_rejected,
        "repair_attempted": repair_attempted,
        "repair_success": repair_success,
        "llm_candidate_validation_status": llm_candidate_validation_status,
        "llm_candidate_validation_errors": llm_candidate_validation_errors or [],
        "llm_candidate_validation_warnings": llm_candidate_validation_warnings,
        "validation_errors": validation_errors,
        "validation_warnings": validation_warnings,
        "llm_writer_ms": float((debug.get("stage_timings_ms") or {}).get("llm_writer_ms") or 0.0),
        "response_time": float(response.get("response_time") or 0.0),
        "displayed_count": len(displayed_evidences),
        "sources_count": len(sources),
        "score": min(max(score, 0), 10),
        "debug_model_match": debug_model_match,
        "error": None,
    }


def _score_result_is_llm_writer(result: dict[str, Any]) -> bool:
    return str(result.get("generation_writer") or "").strip().lower() == "llm_writer"


def _format_errors(failures: list[str]) -> str:
    return ", ".join(str(item) for item in failures if str(item).strip())


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            flat = {key: row.get(key, "") for key in CSV_FIELDS}
            writer.writerow(flat)


def _build_summary(path: Path, results: list[dict[str, Any]], models: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary_lines: list[str] = ["# LLM Writer Benchmark Summary", ""]

    model_rows: dict[str, list[dict[str, Any]]] = {model: [] for model in models}
    for row in results:
        if row.get("error"):
            continue
        model_rows.setdefault(row["model"], []).append(row)

    ranking: list[tuple[str, float]] = []
    metrics_by_model: dict[str, dict[str, Any]] = {}
    for model, rows in model_rows.items():
        q_rows = [row for row in rows if row["question_id"] in {"Q1", "Q2", "Q3", "Q4"}]
        total = max(1, len(q_rows))
        llm_writer_count = sum(1 for row in q_rows if row.get("llm_writer_accepted"))
        fallback_count = sum(1 for row in q_rows if row.get("fallback_reason"))
        hard_gate_rejection_count = sum(1 for row in q_rows if row.get("hard_gate_rejected"))
        repair_attempt_count = sum(1 for row in q_rows if row.get("repair_attempted"))
        repair_success_count = sum(1 for row in q_rows if row.get("repair_success"))
        avg_score = round(sum(row["score"] for row in q_rows) / total, 2)
        validation_fail_count = sum(1 for row in q_rows if str(row.get("validation_status") or "").lower() == "fail")
        validation_warning_count = sum(1 for row in q_rows if str(row.get("validation_status") or "").lower() == "warning")
        avg_time = round(sum(row.get("response_time") or 0.0 for row in q_rows) / total, 3)
        avg_llm_writer_ms = round(sum(row.get("llm_writer_ms") or 0.0 for row in q_rows) / total, 3)
        ranking.append((model, avg_score))
        metrics_by_model[model] = {
            "avg_score": avg_score,
            "llm_writer_rate": round(llm_writer_count / total * 100.0, 1),
            "fallback_rate": round(fallback_count / total * 100.0, 1),
            "hard_gate_rejection_rate": round(hard_gate_rejection_count / total * 100.0, 1),
            "repair_success_rate": round(repair_success_count / max(1, repair_attempt_count) * 100.0, 1) if repair_attempt_count else 0.0,
            "avg_llm_writer_ms": avg_llm_writer_ms,
            "avg_response_time": round(sum(row.get("response_time") or 0.0 for row in q_rows) / total, 3),
            "fallback_count": fallback_count,
            "validation_fail_count": validation_fail_count,
            "validation_warning_count": validation_warning_count,
            "avg_time": avg_time,
            "q_count": len(q_rows),
        }

    ranking.sort(key=lambda item: item[1], reverse=True)
    model_verified_failure = any(
        (
            row.get("question_id") in {"Q1", "Q2", "Q3", "Q4"}
            and (
                row.get("model_verified") is not True
                or row.get("debug_model_match") is not True
                or _normalize_string(row.get("llm_model_effective")) != _normalize_string(row.get("benchmark_model_requested"))
            )
        )
        for row in results
    )
    summary_lines.append("## Classement global par modèle")
    summary_lines.append("")
    if model_verified_failure:
        summary_lines.append(
            "**Benchmark non concluant pour le choix du modèle.** "
            "Au moins un modèle n'a pas été vérifié sur Q1-Q4."
        )
        summary_lines.append("")
    for rank, (model, score) in enumerate(ranking, start=1):
        summary_lines.append(f"{rank}. **{model}** — score moyen Q1-Q4 : {score}")
    summary_lines.append("")

    summary_lines.append("## Résumé des scores et du temps")
    summary_lines.append("")
    summary_lines.append("| Modèle | Score moyen Q1-Q4 | llm_writer_rate | fallback_rate | hard_gate_rejection_rate | repair_success_rate | avg_llm_writer_ms | avg_response_time (s) | Fallbacks Q1-Q4 | Fail Q1-Q4 | Warning Q1-Q4 | Temps moyen (s) |")
    summary_lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for model, metrics in metrics_by_model.items():
        summary_lines.append(
            f"| {model} | {metrics['avg_score']} | {metrics['llm_writer_rate']}% | {metrics['fallback_rate']}% | {metrics['hard_gate_rejection_rate']}% | {metrics['repair_success_rate']}% | {metrics['avg_llm_writer_ms']} | {metrics['avg_response_time']} | {metrics['fallback_count']} | {metrics['validation_fail_count']} | {metrics['validation_warning_count']} | {metrics['avg_time']} |"
        )
    summary_lines.append("")

    summary_lines.append("## Détails des fallbacks et warnings")
    summary_lines.append("")
    fallback_reason_counts: dict[str, int] = {}
    warnings_by_type: dict[str, int] = {}
    for row in results:
        fallback_reason = str(row.get("fallback_reason") or "").strip()
        if fallback_reason:
            fallback_reason_counts[fallback_reason] = fallback_reason_counts.get(fallback_reason, 0) + 1
        for warning in row.get("validation_warnings") or []:
            warnings_by_type[str(warning)] = warnings_by_type.get(str(warning), 0) + 1
        for warning in row.get("llm_candidate_validation_warnings") or []:
            warnings_by_type[str(warning)] = warnings_by_type.get(str(warning), 0) + 1
    if fallback_reason_counts:
        summary_lines.append("### fallback_by_reason")
        for reason, count in sorted(fallback_reason_counts.items(), key=lambda x: x[1], reverse=True):
            summary_lines.append(f"- {reason}: {count}")
        summary_lines.append("")
    if warnings_by_type:
        summary_lines.append("### warnings_by_type")
        for warning, count in sorted(warnings_by_type.items(), key=lambda x: x[1], reverse=True):
            summary_lines.append(f"- {warning}: {count}")
        summary_lines.append("")

    summary_lines.append("## Erreurs fréquentes")
    summary_lines.append("")
    error_counts: dict[str, int] = {}
    warning_counts: dict[str, int] = {}
    for row in results:
        for error in row.get("validation_errors") or []:
            error_counts[str(error)] = error_counts.get(str(error), 0) + 1
        for error in row.get("llm_candidate_validation_errors") or []:
            error_counts[str(error)] = error_counts.get(str(error), 0) + 1
        for warning in row.get("validation_warnings") or []:
            warning_counts[str(warning)] = warning_counts.get(str(warning), 0) + 1
        for warning in row.get("llm_candidate_validation_warnings") or []:
            warning_counts[str(warning)] = warning_counts.get(str(warning), 0) + 1
    if not error_counts and not warning_counts:
        summary_lines.append("Aucune erreur ou avertissement identifié sur les requêtes analysées.")
    else:
        if error_counts:
            summary_lines.append("### Erreurs")
            for error, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                summary_lines.append(f"- {error}: {count}")
        if warning_counts:
            summary_lines.append("")
            summary_lines.append("### Avertissements")
            for warning, count in sorted(warning_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                summary_lines.append(f"- {warning}: {count}")
    summary_lines.append("")

    summary_lines.append("## Détails par modèle / question")
    summary_lines.append("")
    summary_lines.append("| Modèle | Question | Score | Route | Writer | Validation | Qualité | Fallback | Temps | Answer preview |")
    summary_lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for row in results:
        if row.get("error"):
            preview = f"ERROR: {row['error']}"
        else:
            answer_preview = str(row.get("answer") or "").replace("\n", " ")[:120]
            preview = answer_preview
        summary_lines.append(
            f"| {row['model']} | {row['question_id']} | {row['score']} | {row.get('selected_route') or ''} | {row.get('generation_writer') or ''} | {row.get('validation_status') or ''} | {row.get('quality_final_status') or ''} | {row.get('fallback_reason') or ''} | {round(float(row.get('response_time') or 0.0), 3)} | {preview} |"
        )
    summary_lines.append("")

    summary_lines.append("## Check non-régression Q5")
    summary_lines.append("")
    summary_lines.append("Les résultats de Q5 sont inclus dans le JSON et le CSV, mais le score global LLM writer est calculé uniquement sur Q1-Q4.")
    summary_lines.append("")

    path.write_text("\n".join(summary_lines), encoding="utf-8")


def _build_report_files(output_dir: Path, results: list[dict[str, Any]]) -> None:
    json_path = output_dir / "llm_writer_benchmark_results.json"
    csv_path = output_dir / "llm_writer_benchmark_results.csv"
    md_path = output_dir / "llm_writer_benchmark_summary.md"
    _write_json(json_path, results)
    _write_csv(csv_path, results)
    _build_summary(md_path, results, sorted({row.get("model") for row in results if row.get("model")}))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark local Ollama LLM writers against the Medical RAG /chat API.")
    parser.add_argument(
        "--models",
        nargs="*",
        default=DEFAULT_MODELS,
        help="Liste des modèles à tester. Exemple: llama3.2:latest qwen2.5:7b-instruct mistral:7b-instruct-q4_0 gemma3:4b",
    )
    parser.add_argument(
        "--questions",
        nargs="*",
        default=["default"],
        help="Liste des questions à tester ou 'default' pour les 5 questions standard.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Répertoire de sortie pour les rapports JSON/CSV/MD.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Affiche les modèles et questions sans appeler l'API.",
    )
    parser.add_argument(
        "--token",
        default=os.getenv("TOKEN") or os.getenv("MEDICAL_RAG_TOKEN"),
        help="Jeton Bearer pour l'API. Par défaut lu depuis TOKEN ou MEDICAL_RAG_TOKEN.",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("BASE_URL", DEFAULT_BASE_URL),
        help="URL de base de l'API backend. Par défaut http://127.0.0.1:8000.",
    )
    parser.add_argument(
        "--conv-id",
        default=os.getenv("CONV_ID"),
        help="Conversation ID à utiliser. Si absent, une nouvelle conversation est créée.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    models = args.models or DEFAULT_MODELS
    questions = DEFAULT_QUESTIONS
    if len(args.questions) == 1 and args.questions[0] == "default":
        questions = DEFAULT_QUESTIONS
    elif args.questions:
        raise SystemExit("Only 'default' question set is supported at the moment.")

    token = _normalize_string(args.token)
    base_url = str(args.base_url).rstrip("/")

    if args.dry_run:
        conv_id = str(args.conv_id or "").strip() or "<will create if not provided>"
    else:
        if not token:
            raise SystemExit("Missing TOKEN environment variable or --token argument.")
        conv_id = _ensure_conversation_id(base_url=base_url, token=token, conv_id_env=args.conv_id)

    print("LLM Writer Benchmark")
    print(f"BASE_URL={base_url}")
    print(f"CONV_ID={conv_id}")
    print(f"Models: {models}")
    print(f"Questions: {[qid for qid, _ in questions]}")
    print(f"Output directory: {output_dir}")
    if args.dry_run:
        print("\nDry run mode: aucune requête API ne sera envoyée.")
        return 0

    if not shutil.which("ollama"):
        print("Warning: ollama CLI is not available in PATH. Model install checks will be skipped.")

    results: list[dict[str, Any]] = []
    for model in models:
        installed = _model_installed(model)
        if installed is False:
            print(f"Model not found. Run: ollama pull {model}")
            continue
        if installed is None:
            print(f"ollama CLI unavailable, skipping install check for model {model}.")

        print(f"\n=== Testing model: {model} ===")
        for question_id, question in questions:
            print(f" - {question_id}: {question}")
            try:
                payload = {
                    "conversation_id": conv_id,
                    "message": question,
                    "history": [],
                    "mode": "general",
                    "llm_model_override": model,
                }
                response = _http_json(
                    method="POST",
                    url=f"{base_url}/chat",
                    token=token,
                    payload=payload,
                    timeout_s=180,
                )
                result = _extract_response_fields(model=model, question_id=question_id, question=question, response=response)
                if not result["model_verified"]:
                    print(
                        "   WARN: model override not verified "
                        f"(requested={result.get('benchmark_model_requested')}, "
                        f"llm_model_requested={result.get('llm_model_requested')}, "
                        f"llm_model_effective={result.get('llm_model_effective')}, "
                        f"ollama_model={result.get('ollama_model')}, "
                        f"override_applied={result.get('llm_model_override_applied')})"
                    )
                results.append(result)
            except Exception as exc:
                error_result = {
                    "model": model,
                    "benchmark_model_requested": model,
                    "question_id": question_id,
                    "question": question,
                    "answer": "",
                    "generation_mode": None,
                    "generation_writer": None,
                    "validation_status": None,
                    "quality_final_status": None,
                    "fallback_reason": None,
                    "selected_route": None,
                    "llm_provider": None,
                    "llm_model_requested": None,
                    "llm_model_effective": None,
                    "ollama_model": None,
                    "model_verified": False,
                    "llm_model_override_applied": False,
                    "llm_writer_attempted": False,
                    "llm_writer_accepted": False,
                    "hard_gate_rejected": False,
                    "repair_attempted": False,
                    "repair_success": False,
                    "llm_candidate_validation_status": None,
                    "llm_candidate_validation_errors": [],
                    "llm_candidate_validation_warnings": [],
                    "validation_errors": [],
                    "validation_warnings": [],
                    "llm_writer_ms": 0.0,
                    "response_time": 0.0,
                    "displayed_count": 0,
                    "sources_count": 0,
                    "score": 0,
                    "debug_model_match": False,
                    "error": str(exc),
                }
                results.append(error_result)
                print(f"   ERROR: {exc}")

    _build_report_files(output_dir, results)
    print(f"\nReports written to {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
