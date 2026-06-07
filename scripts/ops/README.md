# Ops scripts

## P0 readiness report

```bash
PYTHONPATH=. python scripts/ops/run_p0_readiness.py
```

Exit code:

- `0`: all blocking checks passed
- `2`: at least one blocking check failed

## Summary quality check

Backend suite for report summary quality, LLM writer acceptance, document identity, and clickable-source consistency.

```bash
export BASE_URL="http://127.0.0.1:8000"
export TOKEN="..."
bash scripts/ops/summary_quality_check.sh
```

Interactive mode:

```bash
source scripts/ops/summary_quality_check.sh
export CONV_ID="$(create_summary_quality_conversation)"
run_q "Fais une synthèse biologique courte du report 12..."
```

Outputs are written under `reports/summary_quality_check/<timestamp>/` and the last raw response is copied to `/tmp/last_chat_response.json`.

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
