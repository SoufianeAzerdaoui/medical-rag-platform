# Generation Phase (Medical RAG)

## Rôle
La phase `generation` transforme les résultats retrieval en réponse médicale **courte, fidèle, sourcée**, sans hallucination évidente et sans fuite PII/PHI.

Pipeline implémenté :

User question  
→ Query normalization  
→ Clinical-aware retrieval (`scripts/retrieval`)  
→ Evidence pack builder  
→ Prompt builder  
→ LLM client (Ollama local ou Gemini selon la route)  
→ Answer validator  
→ Final answer + citations/provenance

## Architecture
- `generate_answer.py`: orchestrateur CLI.
- `evidence_builder.py`: construit un evidence pack structuré à partir du retrieval.
- `prompt_builder.py`: construit un prompt médical strict (grounded).
- `llm_client.py`: client Ollama local (`/api/generate`).
- `answer_validator.py`: garde-fous anti-hallucination/anti-PII.
- `citation_builder.py`: formatte les citations provenance.
- `validate_generation.py`: tests end-to-end + rapport JSON.

## Dépendance Ollama
Prérequis :
- service `ollama` actif
- modèle `qwen2.5:7b-instruct` installé

Vérification rapide :

```bash
curl http://127.0.0.1:11434/api/tags
ollama list
```

## Modèle local
- Provider: `ollama`
- Modèle: `qwen2.5:7b-instruct`
- CPU-only compatible (plus lent)

## Pourquoi `think=false` est obligatoire
Qwen3 peut renvoyer un champ `thinking` sans texte final exploitable.  
Le client impose :
- `stream=false`
- `think=false` (top-level)

Payload utilisé :

```json
{
  "model": "qwen3:4b",
  "prompt": "...",
  "stream": false,
  "think": false,
  "options": {
    "temperature": 0.0,
    "top_p": 0.8,
    "num_ctx": 4096,
    "num_predict": 800
  }
}
```

## Commandes Ollama (test)

```bash
curl -s http://127.0.0.1:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5:7b-instruct",
    "prompt": "Réponds en français: test.",
    "stream": false,
    "think": false,
    "options": {
      "temperature": 0.0,
      "num_ctx": 4096,
      "num_predict": 200
    }
  }'
```

## Commandes génération

```bash
python scripts/generation/generate_answer.py \
  --query "Quel est le résultat de la calcitonine ?" \
  --provider ollama \
  --model qwen2.5:7b-instruct \
  --top-k 5
```

JSON :

```bash
python scripts/generation/generate_answer.py \
  --query "Quels résultats ont un résultat antérieur ?" \
  --provider ollama \
  --model qwen2.5:7b-instruct \
  --top-k 5 \
  --json
```

Afficher le contexte :

```bash
python scripts/generation/generate_answer.py \
  --query "Quel parasite a été détecté ?" \
  --provider ollama \
  --model qwen2.5:7b-instruct \
  --top-k 5 \
  --show-context
```

### Inspection locale type `run_q`

Pour contrôler le routing, le provider effectif, le modèle et les timings runtime avant de passer au serveur :

```bash
python scripts/ops/run_q.py \
  --query "Quel est le résultat de la calcitonine ?" \
  --provider ollama \
  --model qwen2.5:7b-instruct \
  --show-context \
  --raw-debug
```

Ajoute `--json` si tu veux le rapport complet en JSON.

## Validation génération

```bash
python scripts/generation/validate_generation.py \
  --provider ollama \
  --model qwen2.5:7b-instruct \
  --report data/generation/generation_validation_report.json
```

## Sécurité médicale implémentée
- grounded generation (contexte uniquement)
- refus d’extrapolation
- garde-fou anti-diagnostic définitif
- garde-fou anti-prescription
- scan anti-PII/PHI
- citations obligatoires si evidence disponible
- refus explicite si contexte insuffisant

## Limites CPU-only
- génération plus lente (plusieurs secondes à dizaines de secondes)
- `top_k` élevé + `num_ctx` élevé augmentent la latence
- `max_tokens` élevé augmente le temps de réponse

## Recommandations pratiques
- `top_k=5`
- `num_ctx=4096`
- `max_tokens=600-800`
- `temperature=0.0`

## Troubleshooting Ollama
- Service down: `systemctl status ollama`
- Modèle absent: `ollama pull qwen2.5:7b-instruct`
- Réponse vide avec thinking: vérifier `think=false` top-level
- Timeout: augmenter `timeout` côté client ou réduire `max_tokens`

## Exemples de questions supportées
- "Quel est le résultat de la calcitonine ?"
- "Quels résultats sont supérieurs à la référence ?"
- "Quels résultats ont un résultat antérieur ?"
- "Quel parasite a été détecté ?"

## Limites production-ready
Cette implémentation vise la qualité PFE (robuste, claire, sûre) mais n’est pas encore production-ready :
- pas d’observabilité avancée
- pas de gestion de charge
- pas de contrôle d’accès robuste côté API
- pas de versioning/rollback LLM complet

## Terminal interactive mode

Lancer le mode terminal interactif :

```bash
python scripts/generation/chat_cli.py
```

Ou avec paramètres explicites :

```bash
python scripts/generation/chat_cli.py \
  --provider ollama \
  --model qwen2.5:7b-instruct \
  --top-k 5
```

Commandes disponibles dans le chat :
- `/help`
- `/exit` ou `/quit`
- `/context on`
- `/context off`
- `/json on`
- `/json off`
- `/settings`
- `/clear`

Exemples de questions :
- `Quel est le résultat de la calcitonine ?`
- `Quels résultats sont supérieurs à la référence ?`
- `Quels résultats ont un résultat antérieur ?`
- `Quel parasite a été détecté ?`
- `Quel traitement faut-il donner ?`
- `Quel est le nom du patient ?`

Limites CPU-only :
- la latence peut être élevée sur `qwen3:4b`
- privilégier `top_k=3` ou `top_k=5` pour garder des réponses rapides
