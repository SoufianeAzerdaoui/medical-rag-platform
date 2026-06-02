# DigitalOcean Deployment

This project now supports two deployment styles:

1. `docker compose` with a local Ollama service for `llama3.2:latest`
2. `systemd` services on a host with Ollama installed natively

## Recommended setup

For the production path, use:

```env
APP_ENV=production
PROD_READINESS_ENFORCE=true
MEDICAL_RAG_PLANNER_SHADOW_MODE=1
MEDICAL_RAG_PLANNER_ENABLE_TAKEOVER=0
MEDICAL_RAG_LLM_PROVIDER=ollama
MEDICAL_RAG_LLM_MODEL=llama3.2:latest
MEDICAL_RAG_ALLOW_MODEL_OVERRIDE=false
MEDICAL_RAG_OLLAMA_URL=http://ollama:11434
MEDICAL_RAG_OLLAMA_MODEL=llama3.2:latest
MEDICAL_RAG_GEMINI_MODEL=gemini-2.5-flash
MEDICAL_RAG_REQUEST_SUMMARY_LOG_LEVEL=info
MEDICAL_RAG_TRACE_LLM=0
GEMINI_API_KEY=your_gemini_api_key
GOOGLE_API_KEY=your_gemini_api_key
NEXT_PUBLIC_RAG_API_URL=http://YOUR_SERVER_IP:8000
FRONTEND_ORIGIN=http://YOUR_SERVER_IP:3000
OLLAMA_BIND_HOST=0.0.0.0:11434
OLLAMA_CONTEXT_LENGTH=4096
OLLAMA_NUM_PARALLEL=1
OLLAMA_MAX_LOADED_MODELS=1
JWT_SECRET=replace_with_a_long_random_secret
APP_DB_PATH=/app/data/app_state.sqlite3
LOGS_DIR=/app/logs
```

The repository includes a ready-to-edit template at `deploy/digitalocean.env.example`.

## Docker Compose

Recommended if you want the simplest runtime on a single droplet:

```bash
cp deploy/digitalocean.env.example .env
docker compose up -d ollama
docker compose exec ollama ollama pull llama3.2:latest
docker compose up -d --build
docker compose ps
```

Notes:
- the backend uses `MEDICAL_RAG_OLLAMA_URL` to reach the Ollama service inside the Compose network
- Gemini remains external and is used only when the route profile allows it
- `4 GB` RAM is usable for validation, but `8 GB` is safer once Ollama is active
- `deploy/deploy_docker_compose.sh` wraps the same sequence for repeatable runs
- model overrides stay disabled in production unless `MEDICAL_RAG_ALLOW_MODEL_OVERRIDE=true`

## systemd

Use the files in `deploy/systemd/` if you want native services instead of containers.

Typical order:

```bash
sudo systemctl enable --now ollama
sudo systemctl enable --now medical-rag-backend
sudo systemctl enable --now medical-rag-frontend
```

## Model policy

- exact/technical routes stay local-first on Ollama
- narrative / transformation routes can use Gemini when credentials are present
- fallback remains deterministic if the writer is rejected or unavailable
