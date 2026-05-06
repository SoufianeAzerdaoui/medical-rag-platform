#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from typing import Any, List, Dict, Optional
from dataclasses import dataclass

try:
    from .models import SearchResponse, RetrievalResult
    from .search import SearchEngine
    from .generation import AnswerGenerator, MedicalPrompter
except ImportError:
    from models import SearchResponse, RetrievalResult
    from search import SearchEngine
    from generation import AnswerGenerator, MedicalPrompter

@dataclass
class Evidence:
    id: str
    text: str
    doc_id: str
    page: Optional[int]
    confidence: float
    metadata: Dict[str, Any]

class EvidencePack:
    def __init__(self, query: str, evidences: List[Evidence]):
        self.query = query
        self.evidences = evidences

    def to_text(self) -> str:
        lines = []
        for i, ev in enumerate(self.evidences, 1):
            source = f"Source: {ev.doc_id}"
            if ev.page:
                source += f", Page: {ev.page}"
            lines.append(f"--- EVIDENCE {i} [{source}] ---\n{ev.text}")
        return "\n\n".join(lines)

class AnswerValidator:
    """
    Vérifie la fidélité de la réponse par rapport au pack de preuves.
    """
    def validate(self, answer: str, pack: EvidencePack) -> str:
        # 0. Ignorer les messages d'erreur technique
        if "Erreur de connexion" in answer or "Read timed out" in answer:
            return answer

        # 1. Vérification des citations
        if "[" not in answer and "]" not in answer:
            answer += "\n\n⚠️ ATTENTION : Cette réponse manque de citations explicites."

        # 2. Vérification des hallucinations numériques simples
        # On extrait tous les nombres de la réponse (ex: 24,00)
        numbers_in_answer = re.findall(r"\d+[\.,]\d+", answer)
        numbers_in_answer += re.findall(r"\b\d+\b", answer)
        
        # On extrait tous les nombres du pack de preuves
        all_evidence_text = pack.to_text()
        numbers_in_evidence = set(re.findall(r"\d+[\.,]\d+", all_evidence_text))
        numbers_in_evidence.update(re.findall(r"\b\d+\b", all_evidence_text))

        hallucinations = []
        for n in numbers_in_answer:
            if n not in numbers_in_evidence and len(n) > 1: # On ignore les chiffres isolés < 10 si besoin
                hallucinations.append(n)

        if hallucinations:
            unique_h = set(hallucinations)
            answer += f"\n\n🚨 ALERTE HALLUCINATION : Les valeurs suivantes ne sont pas dans les sources : {', '.join(unique_h)}"

        return answer

class MedicalRagPipeline:
    """
    Orchestrateur du pipeline RAG Médical strict.
    """
    def __init__(self):
        self.search_engine = SearchEngine()
        self.generator = AnswerGenerator()
        self.prompter = MedicalPrompter()
        self.validator = AnswerValidator()

    def normalize_query(self, query: str) -> str:
        """
        Étape 2: Query normalization
        """
        q = query.lower().strip()
        synonyms = {
            "leuco": "leucocytes",
            "pnn": "neutrophiles",
            "hb": "hemoglobine",
            "vgm": "volume globulaire moyen",
            "tsh": "tshus thyroid",
            "gly": "glycemie"
        }
        for k, v in synonyms.items():
            if k == q: q = v
        return q

    def build_evidence_pack(self, response: SearchResponse) -> EvidencePack:
        """
        Étape 5: Evidence pack builder
        Structure les résultats bruts en un pack de preuves utilisable.
        """
        evidences = []
        chunks = response.context_chunks if response.context_chunks else response.top_results
        
        for chunk in chunks:
            evidences.append(Evidence(
                id=chunk.chunk_id,
                text=chunk.text,
                doc_id=chunk.doc_id,
                page=chunk.page_number,
                confidence=float((chunk.metadata or {}).get("evidence_score", 0.0)),
                metadata=chunk.metadata or {}
            ))
        
        return EvidencePack(query=response.query, evidences=evidences)

    def run(self, user_question: str) -> Dict[str, Any]:
        """
        ÉCHELLE DU PIPELINE (9 ÉTAPES)
        """
        # 1. User Question (Reçue en argument)
        
        # 2. Query normalization
        normalized_query = self.normalize_query(user_question)

        # 3 & 4. Clinical-aware retrieval & Context builder
        search_response = self.search_engine.search(
            query=normalized_query,
            top_k=5
        )

        # 5. Evidence pack builder
        evidence_pack = self.build_evidence_pack(search_response)

        # 6. Prompt builder
        context_text = evidence_pack.to_text()
        full_prompt = f"CONTEXTE MÉDICAL (EVIDENCE PACK) :\n{context_text}\n\nQUESTION : {normalized_query}\n\nRÉPONSE MÉDICALE DÉTAILLÉE (avec citations) :"

        # 7. LLM client
        raw_answer = self.generator.generate_from_full_prompt(full_prompt)

        # 8. Answer validator
        final_answer = self.validator.validate(raw_answer, evidence_pack)

        # 9. Final answer with citations/provenance
        return {
            "query": user_question,
            "normalized_query": normalized_query,
            "answer": final_answer,
            "evidence_count": len(evidence_pack.evidences),
            "sources": [
                {
                    "doc_id": ev.doc_id,
                    "page_number": ev.page,
                    "confidence": ev.confidence
                } for ev in evidence_pack.evidences
            ]
        }

    def close(self):
        self.search_engine.close()

if __name__ == "__main__":
    print("Pipeline RAG Médical chargé.")
