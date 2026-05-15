#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import json
from pathlib import Path

import os

# Add the current directory to sys.path to handle standalone execution
current_dir = Path(__file__).parent.absolute()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

try:
    from .pipeline import MedicalRagPipeline
    from .config import DEFAULT_TOP_K, DEFAULT_LLM_MODEL
except (ImportError, ValueError):
    from pipeline import MedicalRagPipeline
    from config import DEFAULT_TOP_K, DEFAULT_LLM_MODEL

# Couleurs ANSI pour le terminal
BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"

def main():
    parser = argparse.ArgumentParser(description="Medical RAG - Pipeline Strict")
    parser.add_argument("query", nargs="?", help="Votre question médicale (laisser vide pour le mode interactif)")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Nombre de chunks")
    parser.add_argument("--json", action="store_true", help="Sortie JSON")
    args = parser.parse_args()

    # 1. Initialisation du pipeline
    try:
        pipeline = MedicalRagPipeline()
    except Exception as e:
        print(f"{RED}Erreur d'initialisation : {e}{RESET}")
        return 1

    print(f"\n{BOLD}{GREEN}🚀 Pipeline RAG Médical (Strict Provenance) prêt.{RESET}")
    
    try:
        if args.query:
            run_pipeline_once(pipeline, args.query, args.json)
        else:
            print(f"{BLUE}Entrez votre question (ou 'exit' pour quitter) :{RESET}")
            while True:
                try:
                    query = input(f"\n{BOLD}Question 🩺 > {RESET}").strip()
                    if query.lower() in ("exit", "quit", "q"):
                        break
                    if not query:
                        continue
                    run_pipeline_once(pipeline, query, args.json)
                except KeyboardInterrupt:
                    break
    finally:
        pipeline.close()
    
    return 0

def run_pipeline_once(pipeline, query, is_json):
    try:
        result = pipeline.run(query)
        if is_json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print("\n" + "─"*60)
            print(f"{BOLD}{BLUE}🔍 ANALYSE DE LA QUESTION{RESET}")
            print(f"Brute      : {result['query']}")
            if result['normalized_query'] != result['query']:
                print(f"Normalisée : {result['normalized_query']}")
            print("─"*60)
            
            print(f"\n{BOLD}{GREEN}⚕️ RÉPONSE (Évidences : {result['evidence_count']}){RESET}")
            # Coloration des alertes dans la réponse
            answer = result['answer']
            answer = answer.replace("🚨 ALERTE HALLUCINATION", f"{RED}{BOLD}🚨 ALERTE HALLUCINATION{RESET}{RED}")
            answer = answer.replace("⚠️ ATTENTION", f"{YELLOW}{BOLD}⚠️ ATTENTION{RESET}{YELLOW}")
            print(f"{answer}{RESET}")
            
            print("\n" + "─"*60)
            print(f"{BOLD}{YELLOW}📚 SOURCES ET PROVENANCE{RESET}")
            seen_docs = set()
            for s in result['sources']:
                doc_key = (s['doc_id'], s['page_number'])
                if doc_key not in seen_docs:
                    conf_str = f" [Conf: {s['confidence']:.2f}]" if s['confidence'] > 0 else ""
                    print(f"  • {BOLD}{s['doc_id']}{RESET} (Page {s['page_number'] or '?'}){conf_str}")
                    seen_docs.add(doc_key)
            print("─"*60 + "\n")
    except Exception as e:
        print(f"{RED}Erreur lors de l'exécution : {e}{RESET}")

if __name__ == "__main__":
    sys.exit(main())
