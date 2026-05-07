#!/usr/bin/env python3
from __future__ import annotations

import json
import requests
from typing import Any, List, Optional
from pathlib import Path

try:
    from .config import DEFAULT_LLM_MODEL, DEFAULT_OLLAMA_URL
    from .models import SearchResponse, RetrievalResult
except ImportError:
    from config import DEFAULT_LLM_MODEL, DEFAULT_OLLAMA_URL
    from models import SearchResponse, RetrievalResult

class MedicalPrompter:
    """
    Construit des prompts spécialisés pour le domaine médical avec traçabilité.
    """
    
    SYSTEM_PROMPT = (
        "Tu es un expert en analyse de rapports biologiques médicaux. Ton objectif est d'extraire et synthétiser les données avec une précision absolue.\n\n"
        "DIRECTIVES DE RÉPONSE :\n"
        "1. RAISONNEMENT : Avant de répondre, analyse mentalement si une valeur est 'hors référence' en comparant strictement le nombre avec la plage fournie [Min - Max].\n"
        "2. PRÉCISION NUMÉRIQUE : Ne modifie jamais une valeur (ex: 11.0 ne devient pas 11).\n"
        "3. CITATIONS OBLIGATOIRES : Chaque fait médical doit être suivi de sa source au format : [Source: DOC_ID, Page: X].\n"
        "4. ABSENCE DE DONNÉES : Si la réponse n'est pas dans le contexte, réponds : 'L'information n'est pas présente dans les documents fournis'.\n"
        "5. COMPARAISON TEMPORELLE : Si on demande une évolution, compare les dates des rapports si disponibles.\n"
        "6. STYLE : Professionnel, factuel, et exclusivement en français."
    )

    def build_rag_prompt(self, query: str, context_chunks: List[RetrievalResult]) -> str:
        context_text = ""
        for i, chunk in enumerate(context_chunks, start=1):
            source_info = f"ID: {chunk.doc_id}, Page: {chunk.page_number or '?'}"
            context_text += f"--- EXTRAIT {i} [Source: {source_info}] ---\n"
            context_text += f"{chunk.text}\n\n"

        prompt = (
            f"VOICI LES DONNÉES SOURCES :\n{context_text}\n"
            f"QUESTION : {query}\n\n"
            f"Analyse étape par étape pour vérifier les valeurs par rapport aux références, puis donne ta réponse finale avec citations :"
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

    def generate_answer(self, response: SearchResponse) -> str:
        if not response.context_chunks and not response.top_results:
            return "Désolé, aucun contexte médical n'a été trouvé pour répondre à cette question."

        # Priorité aux chunks de contexte étendus, sinon top results
        chunks = response.context_chunks if response.context_chunks else response.top_results
        
        full_prompt = self.prompter.build_rag_prompt(response.query, chunks)
        return self.generate_from_full_prompt(full_prompt)

    def generate_from_full_prompt(self, full_prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": f"{self.prompter.SYSTEM_PROMPT}\n\n{full_prompt}",
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
            }
        }

        try:
            res = requests.post(self.url, json=payload, timeout=180)
            res.raise_for_status()
            data = res.json()
            return data.get("response", "Erreur : aucune réponse générée.")
        except requests.exceptions.RequestException as e:
            return f"Erreur de connexion à Ollama ({self.model}) : {str(e)}."

if __name__ == "__main__":
    # Test rapide si lancé directement
    print("Module de génération RAG chargé.")
