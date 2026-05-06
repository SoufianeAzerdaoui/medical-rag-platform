#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import json
from pathlib import Path

try:
    from .pipeline import MedicalRagPipeline
    from .config import DEFAULT_TOP_K, DEFAULT_LLM_MODEL
except ImportError:
    from pipeline import MedicalRagPipeline
    from config import DEFAULT_TOP_K, DEFAULT_LLM_MODEL

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
        print(f"Erreur d'initialisation : {e}")
        return 1

    print("\n🚀 Pipeline RAG Médical prêt.")
    
    try:
        if args.query:
            # Mode commande unique
            run_pipeline_once(pipeline, args.query, args.json)
        else:
            # Mode interactif
            print("Entrez votre question (ou 'exit' pour quitter) :")
            while True:
                try:
                    query = input("\nQuestion 🩺 > ").strip()
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
            print("\n" + "="*60)
            print(f"QUESTION : {result['query']}")
            if result['normalized_query'] != result['query']:
                print(f"NORMALISÉ : {result['normalized_query']}")
            print("="*60)
            print(f"\nRÉPONSE (Evidence Pack Size: {result['evidence_count']}) :\n")
            print(result['answer'])
            print("\n" + "="*60)
            print("SOURCES (Citations provenance) :")
            seen_docs = set()
            for s in result['sources']:
                doc_key = (s['doc_id'], s['page_number'])
                if doc_key not in seen_docs:
                    print(f"- {s['doc_id']} (Page {s['page_number'] or '?'})")
                    seen_docs.add(doc_key)
            print("="*60 + "\n")
    except Exception as e:
        print(f"Erreur lors de l'exécution : {e}")

if __name__ == "__main__":
    sys.exit(main())
