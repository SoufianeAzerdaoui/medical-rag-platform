# Go-Live P0 Checklist (Blocking)

Use this checklist before production release.

- [ ] ClamAV actif (`ANTIVIRUS_REQUIRED=true`, `clamscan --version` OK).
- [ ] Chiffrement activé (`DATA_ENCRYPTION_ENABLED=true`) + clé valide (`DATA_ENCRYPTION_KEY`).
- [ ] JWT fort + rotation (`JWT_SECRET_PREVIOUS`) + TTL validé.
- [ ] RBAC ops/admin validé pour override doublon et purge.
- [ ] Audit trail immuable validé (triggers no update/delete).
- [ ] Jobs asynchrones persistants validés après redémarrage backend.
- [ ] Resync registre/index exécuté (`POST /documents/resync-registry`).
- [ ] Backup/restore DB validé:
  - `bash scripts/ops/backup_app_state.sh`
  - `bash scripts/ops/restore_app_state.sh <backup_dir>`
- [ ] Health check dépendances critiques validé (`GET /health`).
- [ ] E2E critique validé (upload -> ingestion -> indexation -> question chat avec source).

## Automated P0 report

```bash
python scripts/ops/run_p0_readiness.py
```

API equivalent (ops/admin token):

```http
GET /admin/go-live/p0-check
```

Optional hard gate at backend startup in production:

- `APP_ENV=production`
- `PROD_READINESS_ENFORCE=true`
