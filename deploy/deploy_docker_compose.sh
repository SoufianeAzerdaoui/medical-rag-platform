#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.yml}"
ENV_FILE="${ENV_FILE:-.env}"
OLLAMA_MODEL="${MEDICAL_RAG_OLLAMA_MODEL:-llama3.2:latest}"
ENABLE_OBSERVABILITY="${ENABLE_OBSERVABILITY:-0}"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing ${ENV_FILE}. Copy deploy/digitalocean.env.example to .env first." >&2
  exit 1
fi

compose_args=(
  --env-file "${ENV_FILE}"
  -f "${COMPOSE_FILE}"
)

if [[ "${ENABLE_OBSERVABILITY}" == "1" ]]; then
  compose_args+=(--profile observability)
fi

compose() {
  docker compose "${compose_args[@]}" "$@"
}

echo "Starting Ollama..."
compose up -d ollama

echo "Pulling model ${OLLAMA_MODEL}..."
compose exec -T ollama ollama pull "${OLLAMA_MODEL}"

echo "Building and starting the stack..."
compose up -d --build

echo "Current service status:"
compose ps
