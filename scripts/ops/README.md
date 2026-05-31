# Ops scripts

## P0 readiness report

```bash
PYTHONPATH=. python scripts/ops/run_p0_readiness.py
```

Exit code:

- `0`: all blocking checks passed
- `2`: at least one blocking check failed

## Backup app state

```bash
bash scripts/ops/backup_app_state.sh
```

Creates a timestamped backup under `backups/app_state/<UTC_TIMESTAMP>/`.

## Restore app state

```bash
bash scripts/ops/restore_app_state.sh backups/app_state/<UTC_TIMESTAMP>
```

Restore requires explicit `YES` confirmation and overwrites local `data/`.
