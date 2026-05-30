#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
BACKUP_DIR="${ROOT_DIR}/backups/app_state/${STAMP}"

mkdir -p "${BACKUP_DIR}"

APP_DB="${ROOT_DIR}/data/app_state.sqlite3"
INDEX_DB="${ROOT_DIR}/data/indexes/medical_rag.sqlite"
QDRANT_DIR="${ROOT_DIR}/data/indexes/qdrant"

if [[ -f "${APP_DB}" ]]; then
  cp "${APP_DB}" "${BACKUP_DIR}/app_state.sqlite3"
fi
if [[ -f "${INDEX_DB}" ]]; then
  cp "${INDEX_DB}" "${BACKUP_DIR}/medical_rag.sqlite"
fi
if [[ -d "${QDRANT_DIR}" ]]; then
  cp -R "${QDRANT_DIR}" "${BACKUP_DIR}/qdrant"
fi

cat > "${BACKUP_DIR}/manifest.txt" <<EOF
created_at_utc=${STAMP}
root_dir=${ROOT_DIR}
app_db_present=$( [[ -f "${APP_DB}" ]] && echo "yes" || echo "no" )
index_db_present=$( [[ -f "${INDEX_DB}" ]] && echo "yes" || echo "no" )
qdrant_present=$( [[ -d "${QDRANT_DIR}" ]] && echo "yes" || echo "no" )
EOF

echo "Backup completed: ${BACKUP_DIR}"
