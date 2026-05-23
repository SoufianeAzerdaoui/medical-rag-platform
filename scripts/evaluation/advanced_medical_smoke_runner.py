from __future__ import annotations

import json
import os
import re
import sys
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, request


DEFAULT_BASE_URL = "http://127.0.0.1:8000"
REPORTS_DIR = Path("reports")
JSON_REPORT_PATH = REPORTS_DIR / "advanced_medical_smoke_results.json"
MD_REPORT_PATH = REPORTS_DIR / "advanced_medical_smoke_results.md"


@dataclass(frozen=True)
class SmokeCase:
    idx: int
    question: str


CASES: list[SmokeCase] = [
    SmokeCase(1, "Quelle est la plage physiologique de l’acide urique chez l’homme dans les rapports disponibles ?"),
    SmokeCase(2, "Quelle est la plage normale de l’acide urique chez la femme selon les rapports ?"),
    SmokeCase(3, "Dans le report 24, quelle est la valeur d’acide urique et est-elle dans la référence pour une femme ?"),
    SmokeCase(4, "Dans le report 29, l’acide urique est-il bas par rapport à la référence féminine ? Donne valeur, référence et statut."),
    SmokeCase(5, "Dans tous les rapports disponibles, quels documents montrent un acide urique en dessous de la référence ? Donne document, valeur, référence et statut."),
    SmokeCase(6, "Quels rapports comportent une recherche de toxiques urinaires, et quelles familles sont testées ?"),
    SmokeCase(7, "Quels rapports contiennent une pharmacotoxicologie sanguine, et quels paramètres sont recherchés ?"),
    SmokeCase(8, "Dans le report 27, quels résultats de pharmacotoxicologie urinaire dépassent leur seuil de référence ?"),
    SmokeCase(9, "Dans le report 25, les toxiques urinaires sont-ils majoritairement sous les seuils ? Donne une réponse technique sans diagnostic."),
    SmokeCase(10, "Dans le report 12, la phosphatase alcaline à 40 UI/L est-elle hors référence chez une femme adulte ?"),
]


def _norm(text: str) -> str:
    s = str(text or "").strip().lower().replace("µ", "u")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.replace("’", "'")
    s = re.sub(r"\s+", " ", s)
    return s


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
        payload={"title": "Advanced Medical Smoke Runner"},
    )
    conv_id = str(created.get("id") or "").strip()
    if not conv_id:
        raise RuntimeError("Failed to create conversation: missing id in /conversations response.")
    return conv_id


def _extract_fields(resp: dict[str, Any]) -> dict[str, Any]:
    debug = dict(resp.get("debug") or {})
    debug_validation = dict(debug.get("validation") or {})
    qu = dict(debug.get("query_understanding") or {})
    stage = dict(debug.get("stage_timings_ms") or {})
    displayed_evidences = list(resp.get("displayed_evidences") or [])
    answer = str(resp.get("answer") or "")

    status_codes: list[str] = []
    for ev in displayed_evidences:
        if not isinstance(ev, dict):
            continue
        raw = str(
            ev.get("technical_status_code")
            or ev.get("interpretation_status")
            or ev.get("status")
            or ""
        ).strip().lower()
        if raw:
            status_codes.append(raw)
    return {
        "selected_route": str(debug.get("selected_route") or "").strip() or None,
        "generation_mode": str(resp.get("generation_mode") or "").strip() or None,
        "validation_status": str(resp.get("validation_status") or "").strip() or None,
        "quality_final_status": str((resp.get("quality_report") or {}).get("final_status") or "").strip() or None,
        "response_time": float(resp.get("response_time") or 0.0),
        "retrieval_ms": float(stage.get("retrieval_ms") or 0.0),
        "llm_writer_ms": float(stage.get("llm_writer_ms") or 0.0),
        "requested_analytes": list(debug.get("requested_analytes") or qu.get("requested_analytes") or []),
        "requested_doc_ids": list(debug.get("requested_doc_ids") or qu.get("requested_doc_ids") or []),
        "technical_condition": str(debug.get("technical_condition") or qu.get("technical_condition") or "").strip() or None,
        "displayed_count": len(displayed_evidences),
        "displayed_status_codes": status_codes,
        "sources_count": len(list(resp.get("sources") or [])),
        "validation_errors": list(debug_validation.get("errors") or []),
        "validation_warnings": list(debug_validation.get("warnings") or []),
        "answer": answer,
    }


def _contains_range(answer_norm: str, lo: str, hi: str, unit: str) -> bool:
    # Accept 35-72 mg/l, 35 - 72 mg/l, 35–72 mg/l etc.
    pat = rf"{re.escape(lo)}\s*[-–]\s*{re.escape(hi)}\s*{re.escape(unit)}"
    return re.search(pat, answer_norm) is not None


def _general_fail_reasons(*, case: SmokeCase, fields: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    answer = str(fields.get("answer") or "")
    answer_norm = _norm(answer)
    selected_route = str(fields.get("selected_route") or "")
    generation_mode = str(fields.get("generation_mode") or "")
    validation_status = str(fields.get("validation_status") or "")
    displayed_count = int(fields.get("displayed_count") or 0)

    if validation_status == "fail":
        reasons.append("validation_status=fail")
    if generation_mode.startswith("llm"):
        reasons.append("generation_mode=llm")
    if not selected_route:
        reasons.append("selected_route manquant")
    if "chunk_id" in answer_norm or "chk_report_" in answer_norm:
        reasons.append("chunk_id visible dans answer")
    viewer_doc_id_ok = bool(re.search(r"/viewer/pdf\?doc_id=report_[0-9]+", answer_norm))
    raw_doc_id_leak = ("doc_id=" in answer_norm) and (not viewer_doc_id_ok)
    if "sqlite_deterministic" in answer_norm or raw_doc_id_leak:
        reasons.append("source brute/internal visible dans answer")
    if (
        ("je ne peux pas fournir une reponse fiable" in answer_norm or "je ne peux pas fournir une réponse fiable" in answer.lower())
        and displayed_count > 0
    ):
        reasons.append("safe_error générique alors que des evidences existent")
    if case.idx in {3, 4, 10} and displayed_count > 6:
        reasons.append("single-analyte semble retourner tout le bilan")
    if case.idx in {6, 7, 8, 9} and any(x in answer_norm for x in ["cristaux", "ecbu", "cytologie urinaire"]):
        reasons.append("confusion toxicologie avec cristaux/ECBU/cytologie")
    return reasons


def _evaluate_case(case: SmokeCase, fields: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons = _general_fail_reasons(case=case, fields=fields)
    ans_norm = _norm(str(fields.get("answer") or ""))
    route = str(fields.get("selected_route") or "")
    mode = str(fields.get("generation_mode") or "")
    val_status = str(fields.get("validation_status") or "")
    analytes = [str(a).strip().lower() for a in list(fields.get("requested_analytes") or [])]
    docs = [str(d).strip().lower() for d in list(fields.get("requested_doc_ids") or [])]
    tc = str(fields.get("technical_condition") or "").strip().lower()
    displayed_count = int(fields.get("displayed_count") or 0)
    displayed_status_codes = [str(s).strip().lower() for s in list(fields.get("displayed_status_codes") or []) if str(s).strip()]

    def expect(cond: bool, msg: str) -> None:
        if not cond:
            reasons.append(msg)

    if case.idx == 1:
        expect(route == "reference_range_lookup", "route attendu: reference_range_lookup")
        expect(mode == "deterministic_reference_range_lookup", "mode attendu: deterministic_reference_range_lookup")
        expect(_contains_range(ans_norm, "35", "72", "mg/l"), "réponse doit contenir 35-72 mg/l")
        expect("chunk_id" not in ans_norm, "chunk_id non autorisé")
        expect(val_status != "fail", "validation_status ne doit pas être fail")
    elif case.idx == 2:
        expect(route == "reference_range_lookup", "route attendu: reference_range_lookup")
        expect(mode == "deterministic_reference_range_lookup", "mode attendu: deterministic_reference_range_lookup")
        expect(_contains_range(ans_norm, "26", "60", "mg/l"), "réponse doit contenir 26-60 mg/l")
        expect("chunk_id" not in ans_norm, "chunk_id non autorisé")
        expect(val_status != "fail", "validation_status ne doit pas être fail")
    elif case.idx == 3:
        expect(route == "doc_scoped_single_analyte_status", "route attendu: doc_scoped_single_analyte_status")
        expect(mode == "deterministic_single_analyte_lookup", "mode attendu: deterministic_single_analyte_lookup")
        expect("report_24" in docs, "requested_doc_ids doit contenir report_24")
        expect("acide_urique" in analytes, "requested_analytes doit contenir acide_urique")
        expect(("40 mg/l" in ans_norm) or ("40,00 mg/l" in ans_norm) or ("40.00 mg/l" in ans_norm), "réponse doit contenir 40 mg/l")
        expect(("dans la reference" in ans_norm) or ("within_reference" in ans_norm), "réponse doit indiquer dans la référence")
        expect(displayed_count <= 6, "doit rester centré analyte (pas tout le bilan)")
    elif case.idx == 4:
        expect(route == "doc_scoped_single_analyte_status", "route attendu: doc_scoped_single_analyte_status")
        expect(("11.11 mg/l" in ans_norm) or ("11,11 mg/l" in ans_norm), "réponse doit contenir 11.11 mg/l")
        expect(("en dessous" in ans_norm) or ("below_reference" in ans_norm), "réponse doit indiquer below_reference")
        expect(displayed_count <= 6, "doit rester centré analyte (pas tout le bilan)")
    elif case.idx == 5:
        expect(route == "global_analyte_abnormal_search", "route attendu: global_analyte_abnormal_search")
        expect(mode == "deterministic_global_analyte_abnormal_search", "mode attendu: deterministic_global_analyte_abnormal_search")
        expect("acide_urique" in analytes, "requested_analytes doit contenir acide_urique")
        expect(tc == "below_reference", "technical_condition attendu: below_reference")
        expect(val_status != "fail", "validation_status ne doit pas être fail")
        if displayed_count <= 0:
            reasons.append("no_evidence inattendu (displayed_count=0)")
    elif case.idx == 6:
        expect(route == "global_toxicology_search", "route attendu: global_toxicology_search")
        expect(mode == "deterministic_global_toxicology_search", "mode attendu: deterministic_global_toxicology_search")
        expect(any(k in ans_norm for k in ["amphetamine", "benzodiazepine", "cocaine", "ecstasy", "opiaces", "phencyclidine"]), "familles toxicologiques urinaires attendues")
        expect("cristaux" not in ans_norm, "cristaux ne doivent pas être preuve toxicologique")
        expect(val_status != "fail", "validation_status ne doit pas être fail")
    elif case.idx == 7:
        expect(route == "global_toxicology_search", "route attendu: global_toxicology_search")
        expect(mode == "deterministic_global_toxicology_search", "mode attendu: deterministic_global_toxicology_search")
        expect(any(k in ans_norm for k in ["ethanol", "valpro", "carbamazep", "lithium"]), "paramètres sanguins attendus")
        expect(val_status != "fail", "validation_status ne doit pas être fail")
    elif case.idx == 8:
        expect(route == "doc_scoped_toxicology_threshold_search", "route attendu: doc_scoped_toxicology_threshold_search")
        expect(mode == "deterministic_doc_scoped_toxicology_threshold_search", "mode attendu: deterministic_doc_scoped_toxicology_threshold_search")
        expect(val_status != "fail", "validation_status ne doit pas être fail")
        expect("phencyclidine" in ans_norm, "PHENCYCLIDINE attendue si applicable")
        if displayed_status_codes:
            expect(
                all(code == "above_reference" for code in displayed_status_codes),
                "displayed_evidences doit contenir uniquement des lignes above_reference",
            )
        expect("sous seuil" not in ans_norm, "réponse threshold ne doit pas lister les lignes sous seuil")
    elif case.idx == 9:
        expect(route == "doc_scoped_toxicology_summary", "route attendu: doc_scoped_toxicology_summary")
        expect(mode == "deterministic_doc_scoped_toxicology_summary", "mode attendu: deterministic_doc_scoped_toxicology_summary")
        expect(any(k in ans_norm for k in ["sous seuil", "au-dessus", "ambigu"]), "synthèse sous/au-dessus/ambigu attendue")
        expect(all(k not in ans_norm for k in ["diagnostic", "traitement recommande", "prescrire"]), "pas de diagnostic/traitement")
        expect(val_status != "fail", "validation_status ne doit pas être fail")
    elif case.idx == 10:
        expect(route in {"doc_scoped_single_analyte_status", "single_analyte_lookup", "doc_scoped_results"}, "route ciblée analyte attendue")
        expect("phosphatase_alcaline" in analytes, "requested_analytes doit contenir phosphatase_alcaline")
        expect(("40 ui/l" in ans_norm) or ("40.0 ui/l" in ans_norm) or ("40,0 ui/l" in ans_norm), "valeur 40 UI/L attendue")
        expect(_contains_range(ans_norm, "40", "150", "ui/l"), "référence femme adulte 40-150 UI/L attendue")
        expect("au-dessus de la reference" not in ans_norm, "ne doit pas dire au-dessus de la référence")
        expect(val_status != "fail", "validation_status ne doit pas être fail")

    return (len(reasons) == 0), reasons


def _print_case_result(case: SmokeCase, fields: dict[str, Any], passed: bool, reasons: list[str]) -> None:
    print(f"\n[{case.idx}/10] {case.question}")
    print(f"  selected_route      : {fields.get('selected_route')}")
    print(f"  generation_mode     : {fields.get('generation_mode')}")
    print(f"  validation_status   : {fields.get('validation_status')}")
    print(f"  quality_final_status: {fields.get('quality_final_status')}")
    print(f"  response_time       : {fields.get('response_time')}")
    print(f"  retrieval_ms        : {fields.get('retrieval_ms')}")
    print(f"  llm_writer_ms       : {fields.get('llm_writer_ms')}")
    print(f"  requested_analytes  : {fields.get('requested_analytes')}")
    print(f"  requested_doc_ids   : {fields.get('requested_doc_ids')}")
    print(f"  technical_condition : {fields.get('technical_condition')}")
    print(f"  displayed_count     : {fields.get('displayed_count')}")
    print(f"  sources_count       : {fields.get('sources_count')}")
    print(f"  validation_errors   : {fields.get('validation_errors')}")
    print(f"  validation_warnings : {fields.get('validation_warnings')}")
    print(f"  RESULT              : {'PASS' if passed else 'FAIL'}")
    if not passed:
        print(f"  FAIL_REASON         : {'; '.join(reasons)}")


def _to_md_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| # | Status | selected_route | generation_mode | validation_status | quality | response_time | retrieval_ms | llm_writer_ms | displayed | sources |",
        "|---|---|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['test_number']} | {row['status']} | {row['selected_route'] or ''} | "
            f"{row['generation_mode'] or ''} | {row['validation_status'] or ''} | {row['quality_final_status'] or ''} | "
            f"{row['response_time']:.3f} | {row['retrieval_ms']:.3f} | {row['llm_writer_ms']:.3f} | {row['displayed_count']} | {row['sources_count']} |"
        )
    return "\n".join(lines)


def main() -> int:
    base_url = str(os.getenv("BASE_URL") or DEFAULT_BASE_URL).strip().rstrip("/")
    token = str(os.getenv("TOKEN") or "").strip()
    conv_id_env = str(os.getenv("CONV_ID") or "").strip() or None

    if not token:
        print("ERROR: TOKEN environment variable is required.", file=sys.stderr)
        return 2

    started = time.perf_counter()
    try:
        conversation_id = _ensure_conversation_id(base_url=base_url, token=token, conv_id_env=conv_id_env)
    except Exception as exc:
        print(f"ERROR: cannot resolve conversation id: {exc}", file=sys.stderr)
        return 2

    print(f"BASE_URL={base_url}")
    print(f"CONV_ID={conversation_id}")

    results: list[dict[str, Any]] = []
    passed_count = 0

    for case in CASES:
        payload = {
            "conversation_id": conversation_id,
            "message": case.question,
            "history": [],
            "mode": "general",
        }
        try:
            raw = _http_json(
                method="POST",
                url=f"{base_url}/chat",
                token=token,
                payload=payload,
            )
            fields = _extract_fields(raw)
            passed, fail_reasons = _evaluate_case(case, fields)
        except Exception as exc:
            fields = {
                "selected_route": None,
                "generation_mode": None,
                "validation_status": "fail",
                "quality_final_status": None,
                "response_time": 0.0,
                "retrieval_ms": 0.0,
                "llm_writer_ms": 0.0,
                "requested_analytes": [],
                "requested_doc_ids": [],
                "technical_condition": None,
                "displayed_count": 0,
                "displayed_status_codes": [],
                "sources_count": 0,
                "validation_errors": [f"runner_exception:{exc}"],
                "validation_warnings": [],
                "answer": "",
            }
            passed, fail_reasons = False, [f"runner_exception: {exc}"]

        if passed:
            passed_count += 1

        row = {
            "test_number": case.idx,
            "question": case.question,
            "selected_route": fields.get("selected_route"),
            "generation_mode": fields.get("generation_mode"),
            "validation_status": fields.get("validation_status"),
            "quality_final_status": fields.get("quality_final_status"),
            "response_time": float(fields.get("response_time") or 0.0),
            "retrieval_ms": float(fields.get("retrieval_ms") or 0.0),
            "llm_writer_ms": float(fields.get("llm_writer_ms") or 0.0),
            "requested_analytes": list(fields.get("requested_analytes") or []),
            "requested_doc_ids": list(fields.get("requested_doc_ids") or []),
            "technical_condition": fields.get("technical_condition"),
            "displayed_count": int(fields.get("displayed_count") or 0),
            "sources_count": int(fields.get("sources_count") or 0),
            "validation_errors": list(fields.get("validation_errors") or []),
            "validation_warnings": list(fields.get("validation_warnings") or []),
            "status": "PASS" if passed else "FAIL",
            "fail_reasons": fail_reasons,
            "answer_preview": str(fields.get("answer") or "")[:600],
        }
        results.append(row)
        _print_case_result(case, fields, passed, fail_reasons)

    total = len(CASES)
    score = f"{passed_count}/{total}"
    elapsed = round(time.perf_counter() - started, 3)

    print("\nSummary")
    print(f"  Score : {score}")
    print(f"  Time  : {elapsed}s")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_payload = {
        "base_url": base_url,
        "conversation_id": conversation_id,
        "started_at_epoch": started,
        "elapsed_seconds": elapsed,
        "score": score,
        "passed": passed_count,
        "total": total,
        "results": results,
    }
    JSON_REPORT_PATH.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = [
        "# Advanced Medical Smoke Results",
        "",
        f"- Base URL: `{base_url}`",
        f"- Conversation ID: `{conversation_id}`",
        f"- Score: **{score}**",
        f"- Elapsed: `{elapsed}s`",
        "",
        _to_md_table(results),
        "",
        "## Fail Details",
    ]
    any_fail = False
    for row in results:
        if row["status"] == "FAIL":
            any_fail = True
            md_lines.append(f"- Test {row['test_number']}: {'; '.join(row['fail_reasons'])}")
    if not any_fail:
        md_lines.append("- None")
    MD_REPORT_PATH.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"  JSON report: {JSON_REPORT_PATH}")
    print(f"  MD report  : {MD_REPORT_PATH}")

    return 0 if passed_count == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
