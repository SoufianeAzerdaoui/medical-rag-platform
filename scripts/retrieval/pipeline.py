#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from typing import Any, List, Dict, Optional
from dataclasses import dataclass

import sys
from pathlib import Path

# Handle standalone vs package execution
if __name__ == "__main__" or "." not in __name__:
    current_dir = Path(__file__).parent.absolute()
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))

try:
    from .models import SearchResponse, RetrievalResult
    from .search import SearchEngine
    from .generation import AnswerGenerator, MedicalPrompter
    from .config import DEFAULT_TOP_K, DEFAULT_CONTEXT_MAX_CHUNKS
except (ImportError, ValueError):
    from models import SearchResponse, RetrievalResult
    from search import SearchEngine
    from generation import AnswerGenerator, MedicalPrompter
    from config import DEFAULT_TOP_K, DEFAULT_CONTEXT_MAX_CHUNKS

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
        # Groupement par document pour une meilleure isolation
        docs: Dict[str, List[Evidence]] = {}
        for ev in self.evidences:
            if ev.doc_id not in docs:
                docs[ev.doc_id] = []
            docs[ev.doc_id].append(ev)

        output = []
        for doc_id, ev_list in docs.items():
            doc_section = [f"=== DOCUMENT: {doc_id} ==="]
            for i, ev in enumerate(ev_list, 1):
                page_info = f", Page: {ev.page}" if ev.page else ""
                doc_section.append(f"[EXTRAIT {i}{page_info}]\n{ev.text}")
            output.append("\n".join(doc_section))
        
        return "\n\n" + "\n\n".join(output)

class AnswerValidator:
    """
    Vérifie la fidélité de la réponse par rapport au pack de preuves.
    """
    def validate(self, answer: str, pack: EvidencePack) -> str:
        # 0. Ignorer les messages d'erreur technique
        if "Erreur de connexion" in answer or "Read timed out" in answer:
            return answer

        # 1. Vérification des citations
        if "[" not in answer and "]" not in answer and "Source:" not in answer:
            answer += "\n\n⚠️ ATTENTION : Cette réponse manque de citations explicites."

        # 2. Vérification des hallucinations numériques
        # On extrait tous les nombres de la réponse (ex: 24,00 ou 10)
        numbers_in_answer = re.findall(r"\d+[\.,]\d+", answer)
        numbers_in_answer += re.findall(r"\b\d+\b", answer)
        
        # On extrait les IDs de documents pour les exclure de la détection d'hallucination
        doc_ids_in_pack = {ev.doc_id for ev in pack.evidences}
        # On extrait aussi les nombres contenus dans les doc_ids (ex: 'report_31' -> '31')
        numbers_in_doc_ids = set()
        for doc_id in doc_ids_in_pack:
            numbers_in_doc_ids.update(re.findall(r"\d+", doc_id))

        # On extrait tous les nombres du pack de preuves
        all_evidence_text = pack.to_text()
        numbers_in_evidence = set(re.findall(r"\d+[\.,]\d+", all_evidence_text))
        numbers_in_evidence.update(re.findall(r"\b\d+\b", all_evidence_text))
        
        # Liste blanche additionnelle (ex: années courantes, numéros de page cités légitimement)
        whitelist = {"2024", "2025", "1", "2", "3", "4", "5"} 
        whitelist.update(numbers_in_doc_ids)

        hallucinations = []
        for n in numbers_in_answer:
            # On considère une hallucination si le nombre n'est ni dans les preuves, 
            # ni dans les noms de docs, ni dans la whitelist
            if n not in numbers_in_evidence and n not in whitelist:
                # On ignore les nombres très simples comme 0 ou les petits entiers s'ils sont dans le pack
                if len(n) > 1 or n not in numbers_in_evidence:
                     hallucinations.append(n)

        if hallucinations:
            unique_h = sorted(list(set(hallucinations)))
            answer += f"\n\n🚨 ALERTE HALLUCINATION : Les valeurs suivantes ne semblent pas provenir des sources : {', '.join(unique_h)}"

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

    def run(self, user_question: str, history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        ÉCHELLE DU PIPELINE (9 ÉTAPES)
        """
        history = history or []
        # 1. User Question (Reçue en argument)
        
        # 2. Query normalization
        normalized_query = self.normalize_query(user_question)

        # 3 & 4. Clinical-aware retrieval & Context builder
        search_response = self.search_engine.search(
            query=normalized_query,
            top_k=DEFAULT_TOP_K,
            max_context_chunks=DEFAULT_CONTEXT_MAX_CHUNKS
        )

        # 5. Evidence pack builder
        evidence_pack = self.build_evidence_pack(search_response)

        # 6. Generate answer using prompter and generator
        # Note: We use the specialized prompter with history
        raw_answer = self.generator.generate_answer(search_response, history=history)

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
