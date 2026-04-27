import cv2
import numpy as np

def detect_cachet(pix) -> dict:
    img_array = np.frombuffer(pix.samples, dtype=np.uint8)
    img = img_array.reshape(pix.height, pix.width, pix.n)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    result = {"found": False, "method": None, "confidence": 0.0}

    # 1. Détection de cercles (Hough)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2,
        minDist=50, param1=50, param2=30,
        minRadius=30, maxRadius=200
    )

    if circles is not None:
        result["found"] = True
        result["method"] = "hough_circle"
        result["confidence"] = 0.75
        result["circles_count"] = len(circles[0])

    return result


def verify_cachet_content(pix, ocr_text: str) -> dict:
    """Vérifie que le cachet contient les infos réglementaires"""
    import re
    checks = {
        "rpps": bool(re.search(r'\b\d{11}\b', ocr_text)),
        "adresse": bool(re.search(r'\d+.*(rue|avenue|bd|place)', ocr_text, re.I)),
        "nom_medecin": bool(re.search(r'(Dr|Docteur|Pr\.?)\s+\w+', ocr_text, re.I)),
        "specialite": bool(re.search(r'(médecin|cardiologue|généraliste|chirurgien)', ocr_text, re.I))
    }
    score = sum(checks.values()) / len(checks)
    return {"checks": checks, "completeness_score": score}