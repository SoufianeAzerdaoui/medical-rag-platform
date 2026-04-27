import anthropic
import base64

client = anthropic.Anthropic()

IMAGE_TYPES = ["logo", "signature", "en_tete", "clinical_chart", "autre"]

def classify_image(pix) -> dict:
    img_bytes = pix.tobytes("png")
    b64 = base64.standard_b64encode(img_bytes).decode()

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=200,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": b64}
                },
                {
                    "type": "text",
                    "text": f"""Analyse cette image extraite d'un document médical.
Classe-la dans une de ces catégories : {IMAGE_TYPES}

Réponds UNIQUEMENT en JSON :
{{"type": "<catégorie>", "confidence": <0.0-1.0>, "reason": "<explication courte>"}}"""
                }
            ]
        }]
    )

    import json
    return json.loads(response.content[0].text)