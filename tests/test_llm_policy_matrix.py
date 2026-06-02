from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import patch
from pathlib import Path
from dataclasses import replace

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
GENERATION_DIR = SCRIPTS_DIR / "generation"
for module_path in (str(SCRIPTS_DIR), str(GENERATION_DIR)):
    if module_path not in sys.path:
        sys.path.insert(0, module_path)

import generate_answer
from professional_answer_composer import PROFESSIONAL_WRITER_SYSTEM_PROMPT, compose_professional_answer
from policy_matrix import (
    DETERMINISTIC_ONLY_ROUTES,
    DETERMINISTIC_PREFERRED_ROUTES,
    LLM_ALLOWED_ROUTES,
    SAFETY_ONLY_ROUTES,
    get_intent_policy,
    get_llm_route_class,
)
from query_understanding import parse_query_understanding


class TestLlmPolicyMatrix(unittest.TestCase):
    class _PromptCaptureLlm:
        def __init__(self) -> None:
            self.last_prompt = ""

        def generate(self, prompt: str, **kwargs: object) -> str:
            system_prompt = str(kwargs.get("system_prompt") or "")
            user_prompt = str(kwargs.get("user_prompt") or "")
            combined = "\n\n".join(part for part in [system_prompt, user_prompt or prompt] if part)
            self.last_prompt = combined
            if "Réponds en JSON strict uniquement" in combined:
                return '{"title":"Synthèse","points":["Point 1","Point 2","Point 3"],"limitations":null}'
            return "Faits techniques : test.\nLimites : test.\nConclusion technique : test."

    def test_llm_allowed_routes_are_explicit(self) -> None:
        self.assertIn("doc_scoped_biological_summary", LLM_ALLOWED_ROUTES)
        self.assertIn("doc_scoped_medical_interpretation_guarded", LLM_ALLOWED_ROUTES)
        self.assertIn("open_grounded_medical_question", LLM_ALLOWED_ROUTES)
        self.assertIn("response_transform", LLM_ALLOWED_ROUTES)
        self.assertNotIn("context_summary_render", LLM_ALLOWED_ROUTES)
        self.assertNotIn("doc_scoped_single_analyte_status", LLM_ALLOWED_ROUTES)
        self.assertNotIn("reference_range_lookup", LLM_ALLOWED_ROUTES)

    def test_deterministic_only_routes_explicitly_block_llm(self) -> None:
        self.assertIn("doc_scoped_single_analyte_status", DETERMINISTIC_ONLY_ROUTES)
        policy = get_intent_policy("doc_scoped_single_analyte_status")
        self.assertEqual(str(policy.get("selected_policy") or ""), "deterministic_only")
        self.assertFalse(bool(policy.get("llm_writer_allowed")))

    def test_safety_only_routes_explicitly_block_llm(self) -> None:
        self.assertIn("diagnostic_safety_question", SAFETY_ONLY_ROUTES)
        self.assertIn("treatment_safety_question", SAFETY_ONLY_ROUTES)
        policy = get_intent_policy("diagnostic_safety_question")
        self.assertEqual(str(policy.get("selected_policy") or ""), "safety_only")
        self.assertFalse(bool(policy.get("llm_writer_allowed")))

    def test_biological_summary_routes_are_llm_allowed_by_default(self) -> None:
        self.assertNotIn("doc_scoped_biological_summary", DETERMINISTIC_PREFERRED_ROUTES)
        policy = get_intent_policy("doc_scoped_biological_summary")
        self.assertEqual(str(policy.get("generation_strategy") or ""), "llm_writer_expected")
        self.assertTrue(bool(policy.get("llm_writer_allowed")))

    def test_unknown_route_defaults_to_non_llm(self) -> None:
        policy = get_intent_policy("unknown_route")
        self.assertEqual(str(policy.get("generation_strategy") or ""), "deterministic_preferred")
        self.assertFalse(bool(policy.get("llm_writer_allowed")))
        self.assertEqual(get_llm_route_class("unknown_route", policy), "deterministic_preferred")

    def test_llm_route_class_maps_explicit_policies(self) -> None:
        self.assertEqual(
            get_llm_route_class("doc_scoped_single_analyte_status", get_intent_policy("doc_scoped_single_analyte_status")),
            "deterministic_only",
        )
        self.assertEqual(
            get_llm_route_class("doc_scoped_biological_summary", get_intent_policy("doc_scoped_biological_summary")),
            "llm_allowed",
        )
        self.assertEqual(
            get_llm_route_class(
                "doc_scoped_medical_interpretation_guarded",
                get_intent_policy("doc_scoped_medical_interpretation_guarded"),
            ),
            "llm_allowed",
        )
        self.assertEqual(
            get_llm_route_class("diagnostic_safety_question", get_intent_policy("diagnostic_safety_question")),
            "safety_only",
        )

    def test_global_llm_kill_switch_forces_fallback_writer_mode(self) -> None:
        qu = parse_query_understanding("Fais une synthèse biologique du report 24")
        with patch.object(generate_answer, "_llm_global_enabled", return_value=False):
            self.assertEqual(generate_answer._hybrid_writer_mode(qu), "fallback")

    def test_global_llm_kill_switch_disables_llm_query_understanding(self) -> None:
        qu = parse_query_understanding("TSH normale ?")
        with patch.object(generate_answer, "_llm_global_enabled", return_value=False):
            merged, debug = generate_answer._llm_assisted_query_understanding(
                query="TSH normale ?",
                base_qu=qu,
                llm_client=None,
                provider="ollama",
                model="dummy",
                timeout=10,
            )
        self.assertEqual(merged.intent, qu.intent)
        self.assertFalse(bool(debug.get("used")))
        self.assertEqual(str(debug.get("error") or ""), "llm_globally_disabled")

    def test_guarded_route_micro_prompt_contains_writer_only_guardrails(self) -> None:
        qu = parse_query_understanding("Le bilan thyroïdien du report 16 est-il cohérent ?")
        client = self._PromptCaptureLlm()
        out = generate_answer._compose_level2_micro_prompt_answer(
            selected_route="doc_scoped_medical_interpretation_guarded",
            query_understanding=qu,
            llm_pack={
                "evidences": [
                    {
                        "analyte": "TSH",
                        "value_with_unit": "55,00 mUI/L",
                        "reference_short": "0,35 à 4,94 mUI/L",
                        "status": "above_reference",
                    }
                ]
            },
            llm_client=client,
            provider="ollama",
            model="dummy",
            num_ctx=2048,
        )
        prompt = client.last_prompt
        self.assertIn("Tu es uniquement un writer/summarizer/rephraser technique.", prompt)
        self.assertIn("Tu n'es ni routeur, ni planner, ni answerability gate.", prompt)
        self.assertIn("Tu ne décides jamais du routing, du scope, de l'answerability ou des sources à garder.", prompt)
        self.assertIn("Tu ne donnes jamais de diagnostic.", prompt)
        self.assertEqual(str(out.get("llm_prompt_policy_version") or ""), "v2")

    def test_open_grounded_route_micro_prompt_requires_limits_not_clinical_conclusion(self) -> None:
        qu = parse_query_understanding("Le bilan rénal est-il normal ?")
        client = self._PromptCaptureLlm()
        generate_answer._compose_level2_micro_prompt_answer(
            selected_route="open_grounded_medical_question",
            query_understanding=qu,
            llm_pack={
                "evidences": [
                    {
                        "analyte": "Créatinine",
                        "value_with_unit": "23 mg/L",
                        "reference_short": "7,2 - 12,5 mg/L",
                        "status": "above_reference",
                    }
                ]
            },
            llm_client=client,
            provider="ollama",
            model="dummy",
            num_ctx=2048,
        )
        prompt = client.last_prompt
        self.assertIn("Structure obligatoire :", prompt)
        self.assertIn("- Faits techniques", prompt)
        self.assertIn("- Limites", prompt)
        self.assertIn("- Conclusion technique", prompt)
        self.assertIn("Si des éléments manquent pour conclure, écris explicitement que le contexte est insuffisant.", prompt)
        self.assertIn("N'emploie jamais 'probablement', 'suggère une maladie', 'traitement recommandé'.", prompt)

    def test_timeout_circuit_opens_only_for_configured_routes(self) -> None:
        with patch.dict(
            os.environ,
            {
                "MEDICAL_RAG_LLM_TIMEOUT_CIRCUIT_ENABLED": "1",
                "MEDICAL_RAG_LLM_TIMEOUT_CIRCUIT_ROUTES": "doc_scoped_medical_interpretation_guarded",
                "MEDICAL_RAG_LLM_TIMEOUT_CIRCUIT_TTL_S": "300",
            },
            clear=False,
        ):
            generate_answer._LLM_TIMEOUT_CIRCUIT_STATE.clear()
            self.assertFalse(
                generate_answer._is_llm_timeout_circuit_open(
                    "doc_scoped_medical_interpretation_guarded",
                    "llama3.2:latest",
                )
            )
            generate_answer._open_llm_timeout_circuit(
                "doc_scoped_medical_interpretation_guarded",
                "llama3.2:latest",
            )
            self.assertTrue(
                generate_answer._is_llm_timeout_circuit_open(
                    "doc_scoped_medical_interpretation_guarded",
                    "llama3.2:latest",
                )
            )
            self.assertFalse(
                generate_answer._is_llm_timeout_circuit_open(
                    "doc_scoped_biological_summary",
                    "llama3.2:latest",
                )
            )
            generate_answer._LLM_TIMEOUT_CIRCUIT_STATE.clear()

    def test_professional_writer_system_prompt_states_non_router_role(self) -> None:
        self.assertIn("Tu n'es ni routeur, ni planner, ni answerability gate.", PROFESSIONAL_WRITER_SYSTEM_PROMPT)
        self.assertIn("Tu ne décides jamais si la requête est answerable, ambiguë ou unsafe.", PROFESSIONAL_WRITER_SYSTEM_PROMPT)

    def test_response_transform_prompt_uses_strict_writer_guardrails(self) -> None:
        qu = parse_query_understanding("Reformule la réponse précédente en style professionnel.")
        pack = {
            "intent": "response_transform",
            "output_format": "paragraph",
            "evidences": [
                {
                    "doc_id": "report_29",
                    "analyte": "Créatinine",
                    "analyte_norm": "creatinine",
                    "current_value": "1.11",
                    "unit": "mg/l",
                    "reference": "5,7 - 11,1 mg/l",
                    "technical_status_code": "below_reference",
                    "technical_status": "en dessous de la référence",
                    "page": 2,
                    "row": 15,
                }
            ],
        }
        client = self._PromptCaptureLlm()
        compose_professional_answer(
            user_question="Reformule la réponse précédente en style professionnel.",
            query_understanding=qu,
            evidence_pack=pack,
            mode="llm_professional_writer",
            source_citations=[],
            llm_client=client,
            provider="ollama",
            model="dummy",
        )
        prompt = client.last_prompt
        self.assertIn("RÈGLE STRICTE: reformule uniquement les facts de results_locked.", prompt)
        self.assertIn("RÈGLES SPÉCIFIQUES response_transform :", prompt)
        self.assertIn("Tu transforms uniquement la forme de la réponse précédente, jamais le fond.", prompt)
        self.assertIn("Tu conserves strictement les mêmes résultats, les mêmes sources et le même périmètre documentaire.", prompt)
        self.assertIn("INTERDIT: ajouter/supprimer/modifier analyte, valeur, unité, référence, statut ou source.", prompt)
        self.assertIn("INTERDIT: recalculer, diagnostiquer, proposer un traitement", prompt)
        self.assertIn("Si une donnée manque, écrire 'non présent' ou 'non disponible'.", prompt)

    def test_context_summary_render_llm_prompt_is_strict_json_and_grounded(self) -> None:
        client = self._PromptCaptureLlm()
        points, limitation = generate_answer._try_llm_grounded_summary(
            llm_client=client,
            provider="ollama",
            model="dummy",
            timeout=10,
            context_type="medical_qualitative_comment",
            subject="cortisol",
            display_text="Commentaire technique sur cortisol. Contrôle conseillé selon le contexte.",
            evidence_pack={"evidences": [{"comment_text": "Commentaire technique sur cortisol."}]},
            sources=[{"label": "report (20).pdf — page 1, ligne 1"}],
            requested_summary_points=3,
        )
        prompt = client.last_prompt
        self.assertIsNotNone(points)
        self.assertIsNone(limitation)
        self.assertIn("Tu ne dois pas ajouter de connaissance médicale externe.", prompt)
        self.assertIn("Tu ne dois pas poser de diagnostic.", prompt)
        self.assertIn("Tu ne dois pas inventer de valeur, source, patient, rapport ou interprétation.", prompt)
        self.assertIn("Réponds en JSON strict uniquement", prompt)
        self.assertIn("requested_summary_points", prompt)

    def test_multi_doc_comparison_prompt_uses_route_specific_guardrails(self) -> None:
        qu = parse_query_understanding("Compare report 10 et 12 vite fait.")
        pack = {
            "intent": "multi_doc_comparison",
            "output_format": "table",
            "evidences": [
                {"doc_id": "report_10", "analyte": "Créatinine", "current_value": "4", "unit": "mg/l"},
                {"doc_id": "report_12", "analyte": "Créatinine", "current_value": "23", "unit": "mg/l"},
            ],
        }
        client = self._PromptCaptureLlm()
        compose_professional_answer(
            user_question="Compare report 10 et 12 vite fait.",
            query_understanding=qu,
            evidence_pack=pack,
            mode="llm_professional_writer",
            source_citations=[],
            llm_client=client,
            provider="ollama",
            model="dummy",
        )
        prompt = client.last_prompt
        self.assertIn("RÈGLES SPÉCIFIQUES comparaison multi-doc :", prompt)
        self.assertIn("Tu conserves strictement la séparation par document.", prompt)
        self.assertIn("Tu ne permutes jamais report A et report B.", prompt)

    def test_cohort_search_prompt_uses_route_specific_guardrails(self) -> None:
        qu = replace(
            parse_query_understanding("Listez tous les rapports qui ont la créatinine supérieure à 10."),
            intent="cohort_search",
        )
        pack = {
            "intent": "cohort_search",
            "output_format": "table",
            "evidences": [
                {"doc_id": "report_12", "analyte": "Créatinine", "current_value": "23", "unit": "mg/l"},
                {"doc_id": "report_29", "analyte": "Créatinine", "current_value": "1.11", "unit": "mg/l"},
            ],
        }
        client = self._PromptCaptureLlm()
        compose_professional_answer(
            user_question="Listez tous les rapports qui ont la créatinine supérieure à 10.",
            query_understanding=qu,
            evidence_pack=pack,
            mode="llm_professional_writer",
            source_citations=[],
            llm_client=client,
            provider="ollama",
            model="dummy",
        )
        prompt = client.last_prompt
        self.assertIn("RÈGLES SPÉCIFIQUES cohort/global search :", prompt)
        self.assertIn("Tu conserves chaque ligne de résultat comme une observation distincte.", prompt)
        self.assertIn("tu ne réduis pas arbitrairement la liste à un seul cas", prompt.lower())


if __name__ == "__main__":
    unittest.main()
