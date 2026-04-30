import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb

DB_DIR = Path(__file__).resolve().parents[2] / "data" / "vector_db"
COLLECTION_NAME = "medical_reports"
MODEL_NAME = "intfloat/multilingual-e5-base"

def main():
    print(f"Chargement du modèle d'embedding: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    
    print(f"Connexion à ChromaDB dans: {DB_DIR}")
    client = chromadb.PersistentClient(path=str(DB_DIR))
    
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception as e:
        print(f"Erreur: La collection '{COLLECTION_NAME}' n'existe pas. Veuillez lancer build_index.py d'abord.")
        return

    print("\n" + "="*50)
    print("🤖 Test interactif de recherche RAG Médical 🤖")
    print("="*50)
    print("Tapez 'exit' ou 'quit' pour quitter.\n")
    
    while True:
        try:
            query = input("\n📝 Posez votre question clinique : ")
            if query.lower() in ['exit', 'quit']:
                break
            if not query.strip():
                continue
                
            print("🔍 Recherche en cours...")
            # E5 exige le préfixe 'query: ' pour la question
            query_embedding = model.encode(["query: " + query]).tolist()
            
            results = collection.query(
                query_embeddings=query_embedding,
                n_results=3, # Top 3
                include=["documents", "metadatas", "distances"]
            )
            
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            distances = results["distances"][0]
            
            print(f"\n✅ Voici les {len(docs)} résultats les plus pertinents :\n")
            for i in range(len(docs)):
                score = 1.0 - distances[i] # Convert cosine distance to similarity score
                print(f"--- Résultat {i+1} (Score de similarité: {score:.2f}) ---")
                print(f"📜 Texte : {docs[i]}")
                print(f"🏷️  Métadonnées : Patient ID: {metas[i].get('patient_id', 'N/A')} | Analyte: {metas[i].get('analyte', 'N/A')} | Date: {metas[i].get('report_date', 'N/A')} | Type: {metas[i].get('chunk_type', 'N/A')}")
                print("-" * 50)
                
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()
