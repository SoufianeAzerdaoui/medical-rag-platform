import os
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not installed
    def tqdm(iterable, **kwargs):
        return iterable

# Configurations
CHUNKS_FILE = Path(__file__).resolve().parents[2] / "data" / "chunks_output" / "all_chunks.jsonl"
DB_DIR = Path(__file__).resolve().parents[2] / "data" / "vector_db"
COLLECTION_NAME = "medical_reports"
MODEL_NAME = "intfloat/multilingual-e5-base"

def main():
    if not CHUNKS_FILE.exists():
        print(f"Fichier {CHUNKS_FILE} introuvable. Avez-vous lancé le chunking global ?")
        return

    print(f"Chargement du modèle d'embedding: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    
    print(f"Initialisation de ChromaDB dans: {DB_DIR}")
    client = chromadb.PersistentClient(path=str(DB_DIR))
    
    # Recréer la collection (pour repartir de zéro)
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print("Collection précédente supprimée.")
    except Exception:
        pass
    
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"} # BGE models usually use cosine similarity
    )

    print("Lecture des chunks...")
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]

    print(f"{len(chunks)} chunks trouvés. Début de l'indexation...")
    
    # ChromaDB préfère des listes séparées pour ids, documents, metadatas
    ids = []
    documents = []
    metadatas = []
    embeddings = []

    # On encode tout le texte d'un coup avec le modèle BGE
    print("Calcul des vecteurs (Embeddings)... Cela peut prendre quelques instants...")
    # Le modèle E5 requiert le préfixe 'passage: ' pour les documents
    texts_to_embed = ["passage: " + c["text"] for c in chunks]
    # Le modèle encode la liste de textes
    vectors = model.encode(texts_to_embed, show_progress_bar=True).tolist()

    unique_ids = set()
    for i, c in enumerate(chunks):
        base_id = c["chunk_id"]
        c_id = base_id
        counter = 1
        while c_id in unique_ids:
            c_id = f"{base_id}_{counter}"
            counter += 1
        unique_ids.add(c_id)
        
        ids.append(c_id)
        documents.append(c["text"])
        embeddings.append(vectors[i])
        
        # ChromaDB n'accepte que des strings, ints, floats, bools dans les métadonnées.
        # On doit aplatir/filtrer les dictionnaires ou listes
        meta = {}
        for k, v in c.get("metadata", {}).items():
            if v is None:
                continue
            if isinstance(v, (str, int, float, bool)):
                meta[k] = v
            elif isinstance(v, list):
                meta[k] = ", ".join(map(str, v)) # Convertir les listes en strings (ex: analytes)
            else:
                meta[k] = str(v)
        
        meta["chunk_type"] = c["chunk_type"]
        metadatas.append(meta)

    # Batch insert to avoid Memory/Payload limits
    batch_size = 100
    print("Insertion dans ChromaDB...")
    for i in tqdm(range(0, len(ids), batch_size), desc="Indexation"):
        collection.add(
            ids=ids[i:i+batch_size],
            documents=documents[i:i+batch_size],
            embeddings=embeddings[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size]
        )
        
    print(f"✅ Indexation terminée avec succès dans {DB_DIR} !")

if __name__ == "__main__":
    main()
