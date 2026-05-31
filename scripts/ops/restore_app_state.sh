#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <backup_dir>"
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BACKUP_DIR="$1"

if [[ ! -d "${BACKUP_DIR}" ]]; then
  echo "Backup directory not found: ${BACKUP_DIR}"
  exit 1
fi

echo "WARNING: This will overwrite current local state in data/."
echo "Backup source: ${BACKUP_DIR}"
read -r -p "Type YES to continue: " CONFIRM
if [[ "${CONFIRM}" != "YES" ]]; then
  echo "Restore cancelled."
  exit 1
fi

mkdir -p "${ROOT_DIR}/data/indexes"

if [[ -f "${BACKUP_DIR}/app_state.sqlite3" ]]; then
  cp "${BACKUP_DIR}/app_state.sqlite3" "${ROOT_DIR}/data/app_state.sqlite3"
fi

if [[ -f "${BACKUP_DIR}/medical_rag.sqlite" ]]; then
  cp "${BACKUP_DIR}/medical_rag.sqlite" "${ROOT_DIR}/data/indexes/medical_rag.sqlite"
fi

if [[ -d "${BACKUP_DIR}/qdrant" ]]; then
  rm -rf "${ROOT_DIR}/data/indexes/qdrant"
  cp -R "${BACKUP_DIR}/qdrant" "${ROOT_DIR}/data/indexes/qdrant"
fi

echo "Restore completed from: ${BACKUP_DIR}"
