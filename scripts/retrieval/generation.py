#!/usr/bin/env python3
from __future__ import annotations

import json
import requests
from typing import Any, List, Optional, Dict
from pathlib import Path

try:
    from .config import DEFAULT_LLM_MODEL, DEFAULT_OLLAMA_URL
    from .models import SearchResponse, RetrievalResult
except ImportError:
    from config import DEFAULT_LLM_MODEL, DEFAULT_OLLAMA_URL
    from models import SearchResponse, RetrievalResult

class MedicalPrompter:
    """
    Construit des prompts spécialisés pour le domaine médical avec traçabilité et mémoire.
    """
    
    SYSTEM_PROMPT = (
        "Tu es un médecin expert en biologie clinique. Ta mission est d'analyser les données patients avec une rigueur absolue.\n\n"
        "### RÈGLES D'OR POUR ÉVITER LE VAGUE :\n"
        "1. **CHIFFRES AVANT TOUT** : Ne dis jamais 'votre taux est normal' sans citer la valeur exacte trouvée ET la plage de référence. Exemple : 'Votre taux de Glucose est de 0.95 g/L (Norme : 0.70 - 1.10)'.\n"
        "2. **PAS DE GÉNÉRALITÉS** : Évite les phrases comme 'vos analyses sont bonnes'. Analyse chaque ligne du tableau de résultats.\n"
        "3. **ANALYSE COMPARATIVE** : Si plusieurs dates sont disponibles, calcule la différence (ex: +15%) ou décris la tendance précisément.\n"
        "4. **DÉTECTION D'ANOMALIE** : Si une valeur est hors norme, mets-la en évidence (gras) et explique sa position par rapport aux bornes.\n"
        "5. **STRUCTURE OBLIGATOIRE** :\n"
        "   - Un tableau Markdown pour les résultats bruts.\n"
        "   - Une analyse textuelle point par point.\n"
        "   - Un bloc [CHART_DATA: ...] si une évolution est détectée.\n"
        "6. **CITATIONS** : Accole [Source: DOC_ID] à chaque donnée numérique.\n"
        "7. **MÉMOIRE** : Si l'utilisateur pose une question de suivi (ex: 'et pour le fer ?'), utilise l'historique pour comprendre le contexte mais base-toi TOUJOURS sur les documents fournis.\n"
        "8. **LIMITES** : Si une donnée est manquante, dis précisément quel examen manque.\n\n"
        "Ton style doit être factuel, précis, et purement médical."
    )

    def build_rag_prompt(self, query: str, context_chunks: List[RetrievalResult], history: List[Dict[str, str]] = None) -> str:
        history = history or []
        
        # Formatage de l'historique
        history_text = ""
        if history:
            history_text = "### HISTORIQUE DE LA CONVERSATION :\n"
            for msg in history[-5:]: # On garde les 5 derniers messages
                role = "Patient" if msg["role"] == "user" else "Médecin"
                history_text += f"{role}: {msg['content']}\n"
            history_text += "\n"

        context_text = ""
        # Groupement par document pour aider l'isolation
        docs = {}
        for chunk in context_chunks:
            if chunk.doc_id not in docs:
                docs[chunk.doc_id] = []
            docs[chunk.doc_id].append(chunk)

        for doc_id, chunks in docs.items():
            context_text += f"=== DOCUMENT: {doc_id} ===\n"
            for i, chunk in enumerate(chunks, start=1):
                page = f", Page: {chunk.page_number}" if chunk.page_number else ""
                context_text += f"[Extrait {i}{page}]\n{chunk.text}\n"
            context_text += "\n"

        prompt = (
            f"{history_text}"
            f"### DONNÉES SOURCES MÉDICALES :\n{context_text}\n"
            f"### QUESTION ACTUELLE : {query}\n\n"
            f"INSTRUCTION : Analyse les documents en tenant compte de l'historique si nécessaire. "
            f"Réponds de manière technique et précise :"
        )
        return prompt

class AnswerGenerator:
    """
    Gère la communication avec Ollama pour générer des réponses.
    """
    
    def __init__(
        self, 
        model: str = DEFAULT_LLM_MODEL, 
        url: str = DEFAULT_OLLAMA_URL
    ) -> None:
        self.model = model
        self.url = url
        self.prompter = MedicalPrompter()

    def generate_answer(self, response: SearchResponse, history: List[Dict[str, str]] = None) -> str:
        if not response.context_chunks and not response.top_results:
            return "Désolé, aucun contexte médical n'a été trouvé pour répondre à cette question."

        chunks = response.context_chunks if response.context_chunks else response.top_results
        
        full_prompt = self.prompter.build_rag_prompt(response.query, chunks, history=history)
        return self.generate_from_full_prompt(full_prompt)

    def generate_from_full_prompt(self, full_prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": f"{self.prompter.SYSTEM_PROMPT}\n\n{full_prompt}",
            "stream": False,
            "options": {
                "temperature": 0.05,
                "top_p": 0.9,
                "num_ctx": 3072,
                "num_predict": 1024
            }
        }

        try:
            res = requests.post(self.url, json=payload, timeout=300)
            res.raise_for_status()
            data = res.json()
            return data.get("response", "Erreur : aucune réponse générée.")
        except requests.exceptions.RequestException as e:
            return f"Erreur de connexion à Ollama ({self.model}) : {str(e)}."

if __name__ == "__main__":
    print("Module de génération RAG (Mémoire Active) chargé.")
