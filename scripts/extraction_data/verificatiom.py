def check_text_quality(text: str) -> dict:
    import re
    if not text:
        return {"score": 0, "issues": ["texte vide"]}

    issues = []
    score = 100

    invalid_ratio = len(re.findall(r'[^\w\s\.,;:!\?\-\'\"()àâéèêëîïôùûüç]', text)) / max(len(text), 1)
    if invalid_ratio > 0.05:
        issues.append(f"bruit OCR: {invalid_ratio:.1%} caractères suspects")
        score -= 30

    if len(text.strip()) < 50:
        issues.append("texte trop court")
        score -= 20

    words = text.split()
    short_garbage = sum(1 for w in words if len(w) > 10 and not w.isalpha())
    if short_garbage / max(len(words), 1) > 0.1:
        issues.append("fragments non-mots détectés")
        score -= 20

    return {"score": max(score, 0), "issues": issues}


def check_table_quality(table: list) -> dict:
    issues = []
    score = 100

    if not table or len(table) < 2:
        return {"score": 0, "issues": ["tableau vide ou insuffisant"]}

    total_cells = sum(len(row) for row in table)
    empty_cells = sum(1 for row in table for cell in row if not cell or str(cell).strip() == "")
    empty_ratio = empty_cells / max(total_cells, 1)

    if empty_ratio > 0.4:
        issues.append(f"{empty_ratio:.0%} cellules vides")
        score -= 40

    col_counts = [len(row) for row in table]
    if len(set(col_counts)) > 2:
        issues.append("nombre de colonnes incohérent entre les lignes")
        score -= 30

    return {"score": max(score, 0), "issues": issues}