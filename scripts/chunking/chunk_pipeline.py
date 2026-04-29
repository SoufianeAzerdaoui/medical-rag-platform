import json
import re
from datetime import date
from pathlib import Path


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def compute_age(birth_date_str: str, reference_date_str: str) -> int | None:
    """Recalcule l'âge réel depuis birth_date + request_date."""
    try:
        if not birth_date_str or not reference_date_str:
            return None
        bd = date.fromisoformat(birth_date_str)
        rd = date.fromisoformat(reference_date_str[:10])
        age = rd.year - bd.year
        if (rd.month, rd.day) < (bd.month, bd.day):
            age -= 1
        return age
    except Exception:
        return None


# ─────────────────────────────────────────────
# Générateurs de chunks
# ─────────────────────────────────────────────

def make_base_metadata(doc: dict) -> dict:
    """Métadonnées communes à tous les chunks du document."""
    patient = doc.get("patient", {})
    report = doc.get("report", {})
    facility = doc.get("facility", {})

    birth_date = patient.get("birth_date")
    request_date = (report.get("request_date") or "")[:10]
    
    # Utiliser l'âge recalculé ou celui déjà calculé par l'extraction
    real_age = compute_age(birth_date, request_date)
    if real_age is None:
        real_age = patient.get("computed_age_at_request_date")

    return {
        "doc_id": doc.get("doc_id"),
        "patient_id": patient.get("patient_id"),
        "patient_sex": patient.get("sex"),
        "birth_date": birth_date,
        "age": real_age,                          
        "report_date": report.get("report_date"),
        "report_id": report.get("report_id"),
        "sample_type": report.get("sample_type"),
        "prescriber": report.get("prescriber"),
        "facility": facility.get("organization"),
        "laboratory": facility.get("laboratory"),
        "validated_by": report.get("validated_by"),
        "validation_date": report.get("validation_date"),
    }


def chunk_patient_context(doc: dict, base_meta: dict) -> dict:
    """Chunk 1 : contexte patient."""
    p = doc.get("patient", {})
    r = doc.get("report", {})
    anonymized_name = f"Patient_{p.get('patient_id', 'Anonyme')}"
    text = (
        f"Patient : {anonymized_name}, "
        f"né(e) le {p.get('birth_date', 'Inconnu')}, "
        f"sexe {p.get('sex_raw', 'Inconnu')}. "
        f"Prescripteur : {r.get('prescriber', 'Inconnu')}. "
        f"Échantillon {r.get('sample_id', 'Inconnu')}, nature {r.get('sample_type', 'Inconnu')}, "
        f"reçu le {(r.get('received_date') or '')[:10]}."
    )
    return {
        "chunk_id": f"{doc['doc_id']}_patient",
        "chunk_type": "patient_context",
        "is_indexable": True,
        "index_role": "context_only",
        "text": text,
        "metadata": {**base_meta, "section": "patient"}
    }


def chunk_results_table(doc: dict, base_meta: dict) -> dict:
    """Chunk 2 : tous les résultats en un seul bloc (retrieval général)."""
    p = doc.get("patient", {})
    p_name = f"Patient_{p.get('patient_id', 'Anonyme')}"
    r_date = doc.get("report", {}).get("report_date", "Inconnu")
    
    # Ajout du contexte patient AU DEBUT du chunk pour l'embedding
    parts = [f"Résultats de laboratoire complets pour le patient {p_name} en date du {r_date} :"]
    
    for r in doc.get("results", []):
        ref_text = r.get("reference_range", {}).get("text")
        line = f"- {r.get('analyte', 'Inconnu')} : {r.get('value_raw', '')} {r.get('unit', '')}"
        if ref_text:
            line += f" (Valeurs usuelles : {ref_text})"
        parts.append(line)

    return {
        "chunk_id": f"{doc['doc_id']}_results_table",
        "chunk_type": "results_table",
        "is_indexable": True,
        "index_role": "full_results",
        "text": "\n".join(parts),
        "metadata": {
            **base_meta,
            "section": "results",
            "analytes": [r.get("analyte", "Inconnu") for r in doc.get("results", [])]
        }
    }


def chunk_per_analyte(doc: dict, base_meta: dict) -> list[dict]:
    """Chunks 3..N : un chunk par analyte (retrieval précis)."""
    chunks = []
    p = doc.get("patient", {})
    r_date = doc.get("report", {}).get("report_date", "Inconnu")
    
    # Contexte à injecter dans chaque chunk
    anonymized_name = f"Patient_{p.get('patient_id', 'Anonyme')}"
    patient_context = f"Pour le patient {anonymized_name} (Sexe: {p.get('sex', '?')}, Âge: {base_meta.get('age', '?')}), le {r_date} :"

    for r in doc.get("results", []):
        ref_text = r.get("reference_range", {}).get("text")

        text = f"{patient_context} Le résultat pour l'analyse de {r.get('analyte', 'Inconnu')} est de {r.get('value_raw', '')} {r.get('unit', '')}."
        if ref_text:
            text += f" Les valeurs de référence sont : {ref_text}."

        analyte_key = r.get("analyte", "inconnu").lower().replace(" ", "_").replace("(", "").replace(")", "")
        chunks.append({
            "chunk_id": f"{doc['doc_id']}_result_{analyte_key}",
            "chunk_type": "single_result",
            "is_indexable": True,
            "index_role": "analyte_result",
            "text": text,
            "metadata": {
                **base_meta,
                "section": "results",
                "analyte": r.get("analyte", "Inconnu"),
                "value": r.get("value_numeric"),
                "unit": r.get("unit"),
                "confidence": r.get("confidence"),
                "confidence_score": r.get("confidence_score"),
                "dedup_key": r.get("dedup_key"),
            }
        })
    return chunks


def chunk_validation(doc: dict, base_meta: dict) -> dict:
    """Chunk final : bloc de validation."""
    v = doc.get("validation", {})
    r = doc.get("report", {})
    text = (
        f"Rapport de biologie validé par {v.get('validated_by')} "
        f"le {(r.get('validation_date') or '')[:10]}. "
        f"Statut du rapport : {v.get('status')}."
    )
    return {
        "chunk_id": f"{doc['doc_id']}_validation",
        "chunk_type": "validation",
        "is_indexable": False,
        "index_role": "audit",
        "text": text,
        "metadata": {**base_meta, "section": "validation"}
    }


# ─────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────

def chunk_document(doc: dict) -> list[dict]:
    """Génère tous les chunks pour un document JSON."""
    base_meta = make_base_metadata(doc)
    chunks = []
    chunks.append(chunk_patient_context(doc, base_meta))
    chunks.append(chunk_results_table(doc, base_meta))
    chunks.extend(chunk_per_analyte(doc, base_meta))
    chunks.append(chunk_validation(doc, base_meta))
    return chunks


def process_json_file(input_path: str, output_path: str):
    """Lit un fichier JSON et écrit les chunks en JSONL."""
    with open(input_path, "r", encoding="utf-8") as f:
        doc = json.load(f)

    chunks = chunk_document(doc)

    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"✅ {len(chunks)} chunks générés → {output_path}")


def process_nested_folders(root_dir: str, output_dir: str):
    """
    Parcourt récursivement 'root_dir' pour trouver chaque 'document.json'
    et génère les chunks correspondants.
    """
    root_path = Path(root_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_chunks = []
    
    # Utilisation de rglob pour chercher "document.json" dans TOUS les sous-dossiers
    json_files = list(root_path.rglob("document.json"))
    
    if not json_files:
        print(f"Aucun 'document.json' trouvé dans {root_dir}")
        return

    for json_file in json_files:
        # On utilise le nom du dossier parent (ex: report_1) pour l'ID unique
        folder_name = json_file.parent.name
        print(f"📄 Traitement du dossier : {folder_name}")

        with open(json_file, "r", encoding="utf-8") as f:
            doc = json.load(f)
        
        # On s'assure que le doc_id dans le JSON correspond au dossier si nécessaire
        if not doc.get("doc_id"):
            doc["doc_id"] = folder_name

        # Appel de la fonction de chunking
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)

        # Sauvegarde d'un fichier JSONL directement dans le dossier du rapport
        out_file = json_file.parent / "chunks.jsonl"
        with open(out_file, "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    # Export global de tous les chunks
    global_out = output_path / "all_chunks.jsonl"
    with open(global_out, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"\n📦 Terminé ! {len(all_chunks)} chunks générés au total.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 3:
        inp, out = sys.argv[1], sys.argv[2]
        if inp.endswith(".json"):
            process_json_file(inp, out)
        else:
            process_nested_folders(inp, out)
    else:
        print("Usage:")
        print("  Un fichier  : python chunker.py report_31.json output/chunks.jsonl")
        print("  Un dossier  : python chunker.py data/jsons/  output/chunks/")
