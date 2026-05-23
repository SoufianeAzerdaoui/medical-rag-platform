from __future__ import annotations

import json
import os
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
GENERATION_ROOT = SCRIPTS_ROOT / "generation"
for root in (SCRIPTS_ROOT, GENERATION_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from professional_answer_composer import compose_professional_answer
from query_understanding import parse_query_understanding


def _default_model_matrix() -> list[tuple[str, str]]:
    # provider, model
    return [
        ("ollama", "llama3.2:latest"),
        ("ollama", "qwen3:4b"),
        ("lmstudio", "gemini-1.5-flash"),
    ]


def _parse_matrix_from_env() -> list[tuple[str, str]]:
    """
    Format:
      WRITER_SNAPSHOT_MODELS="ollama|llama3.2:latest,ollama|qwen3:4b,lmstudio|gemini-1.5-flash"
    """
    raw = (os.getenv("WRITER_SNAPSHOT_MODELS") or "").strip()
    if not raw:
        return _default_model_matrix()
    out: list[tuple[str, str]] = []
    for part in [p.strip() for p in raw.split(",") if p.strip()]:
        if "|" not in part:
            continue
        provider, model = part.split("|", 1)
        provider = provider.strip()
        model = model.strip()
        if provider and model:
            out.append((provider, model))
    return out or _default_model_matrix()


class TestWriterModelSnapshots(unittest.TestCase):
    def test_snapshot_writer_stability_multi_model(self) -> None:
        if os.getenv("RUN_LLM_SNAPSHOT", "0") != "1":
            self.skipTest("Set RUN_LLM_SNAPSHOT=1 to run live multi-model writer snapshot test.")

        query = "Quelle est la plage normale AMH pour une femme de 30-34 ans ? avec source cliquable"
        qu = parse_query_understanding(query)
        evidence_pack = {
            "question": query,
            "intent": "reference_range_lookup",
            "requested_doc_ids": ["report_1"],
            "requested_analytes": ["AMH"],
            "output_format": "list",
            "answer_style": "standard",
            "evidences": [
                {
                    "doc_id": "report_1",
                    "filename": "report (1).pdf",
                    "page": 1,
                    "row": 4,
                    "analyte": "AMH",
                    "analyte_norm": "amh",
                    "current_value": "8",
                    "unit": "ng/ml",
                    "reference": "3,03-3,87 ng/ml",
                    "technical_status_code": "above_reference",
                    "technical_status": "au-dessus de la référence",
                    "source_label": "report (1).pdf — page 1, ligne 4",
                    "viewer_url": "/viewer/pdf?doc_id=report_1&page=1",
                }
            ],
            "missing_items": [],
        }
        source_citations = [
            {
                "doc_id": "report_1",
                "filename": "report (1).pdf",
                "page": 1,
                "row": 4,
                "label": "report (1).pdf — page 1, ligne 4",
                "viewer_url": "/viewer/pdf?doc_id=report_1&page=1",
            }
        ]

        snapshots: list[dict[str, object]] = []
        ran = 0
        for provider, model in _parse_matrix_from_env():
            with self.subTest(provider=provider, model=model):
                out = compose_professional_answer(
                    user_question=query,
                    query_understanding=qu,
                    evidence_pack=evidence_pack,
                    mode="auto",
                    source_citations=source_citations,
                    provider=provider,
                    model=model,
                    temperature=0.0,
                    max_tokens=260,
                    timeout=25,
                )
                answer = str(out.get("answer") or "")
                llm_error = str(out.get("llm_error") or "")
                mode = str(out.get("mode") or "")

                # Unavailable models are recorded and skipped from assertions.
                if mode == "llm_writer_error_fallback":
                    snapshots.append(
                        {
                            "provider": provider,
                            "model": model,
                            "status": "unavailable",
                            "mode": mode,
                            "llm_error": llm_error,
                        }
                    )
                    continue

                ran += 1
                self.assertIn("AMH", answer)
                self.assertTrue(("3,03-3,87 ng/ml" in answer) or ("3,03–3,87 ng/ml" in answer))
                self.assertIn("Sources", answer)
                self.assertNotIn("résultat(s)", answer.lower())
                self.assertNotIn("correspondant(s)", answer.lower())

                snapshots.append(
                    {
                        "provider": provider,
                        "model": model,
                        "status": "ok",
                        "mode": mode,
                        "answer_length": len(answer),
                        "first_200": answer[:200],
                    }
                )

        if ran == 0:
            self.skipTest("No configured model was available for live snapshot run.")

        out_dir = PROJECT_ROOT / "tests" / "artifacts"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "writer_model_snapshots.json"
        payload = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "query": query,
            "models": snapshots,
        }
        out_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

