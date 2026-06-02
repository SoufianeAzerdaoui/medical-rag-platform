# Backend Environment Setup

Le backend lit les variables d’environnement depuis :

1. les variables système (`os.environ`) ;
2. le fichier `.env` à la racine du projet (recommandé) ;
3. en fallback dev, `scripts/generation/.env` (sans override).

Le backend principal est lancé depuis la racine :

```bash
python -m uvicorn backend_api:app --host 127.0.0.1 --port 8000 --reload
```

## Fichier recommandé

Créer/éditer `.env` à la racine :

```env
ENABLE_FEATURE_FLAG_ADMIN_API=true
ADMIN_EMAILS=simo@test.ma
JWT_SECRET=change-me
FRONTEND_ORIGIN=http://localhost:3000
MEDICAL_RAG_LLM_PROVIDER=ollama
MEDICAL_RAG_LLM_MODEL=qwen2.5:7b-instruct
MEDICAL_RAG_LLM_TEMPERATURE=0.1
MEDICAL_RAG_LLM_NUM_CTX=4096
MEDICAL_RAG_LLM_MAX_TOKENS=384
MEDICAL_RAG_LLM_TIMEOUT=180
MEDICAL_RAG_SUMMARY_TIMEOUT_CAP_S=120
MEDICAL_RAG_SUMMARY_MAX_TOKENS=220
MEDICAL_RAG_ALLOW_MODEL_OVERRIDE=false
MEDICAL_RAG_OLLAMA_URL=http://127.0.0.1:11434
MEDICAL_RAG_OLLAMA_MODEL=qwen2.5:7b-instruct
MEDICAL_RAG_GEMINI_MODEL=gemini-2.5-flash
GEMINI_API_KEY=your_api_key_here
```

Notes :
- ne pas committer de vraies clés/secrets ;
- `ADMIN_EMAILS` accepte une liste séparée par virgules.

## Test rapide

```bash
curl -H "Authorization: Bearer $TOKEN" http://127.0.0.1:8000/auth/me
curl -H "Authorization: Bearer $TOKEN" http://127.0.0.1:8000/feature-flags
```

## Toggle feature flag runtime

Désactiver :

```bash
curl -X PATCH "http://127.0.0.1:8000/feature-flags/REFERENCE_RANGE_STRICT_MODE" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"enabled": false}'
```

Réactiver :

```bash
curl -X PATCH "http://127.0.0.1:8000/feature-flags/REFERENCE_RANGE_STRICT_MODE" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"enabled": true}'
```
