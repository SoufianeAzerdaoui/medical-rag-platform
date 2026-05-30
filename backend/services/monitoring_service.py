from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from backend.database import db_connect, now_iso


def _labels_key(labels: dict[str, Any] | None = None) -> str:
    return json.dumps(labels or {}, ensure_ascii=False, sort_keys=True)


def inc(metric: str, value: float = 1.0, labels: dict[str, Any] | None = None) -> None:
    _upsert(metric=metric, metric_type="counter", delta=float(value), labels=labels)


def gauge(metric: str, value: float, labels: dict[str, Any] | None = None) -> None:
    _upsert(metric=metric, metric_type="gauge", absolute=float(value), labels=labels)


def observe_duration(metric_prefix: str, seconds: float, labels: dict[str, Any] | None = None) -> None:
    safe = max(0.0, float(seconds))
    inc(f"{metric_prefix}_count", 1.0, labels=labels)
    inc(f"{metric_prefix}_sum", safe, labels=labels)


def _upsert(
    *,
    metric: str,
    metric_type: str,
    delta: float | None = None,
    absolute: float | None = None,
    labels: dict[str, Any] | None = None,
) -> None:
    key = _labels_key(labels)
    now = now_iso()
    conn = db_connect()
    try:
        if metric_type == "gauge":
            conn.execute(
                """
                INSERT INTO monitoring_metrics (metric_name, metric_type, labels_json, value, updated_at)
                VALUES (?, 'gauge', ?, ?, ?)
                ON CONFLICT(metric_name, labels_json) DO UPDATE SET
                    value = excluded.value,
                    updated_at = excluded.updated_at
                """,
                (metric, key, float(absolute or 0.0), now),
            )
        else:
            conn.execute(
                """
                INSERT INTO monitoring_metrics (metric_name, metric_type, labels_json, value, updated_at)
                VALUES (?, 'counter', ?, ?, ?)
                ON CONFLICT(metric_name, labels_json) DO UPDATE SET
                    value = monitoring_metrics.value + excluded.value,
                    updated_at = excluded.updated_at
                """,
                (metric, key, float(delta or 0.0), now),
            )
        conn.commit()
    finally:
        conn.close()


def snapshot() -> list[dict[str, Any]]:
    conn = db_connect()
    try:
        rows = conn.execute(
            """
            SELECT metric_name, metric_type, labels_json, value, updated_at
            FROM monitoring_metrics
            ORDER BY metric_name, labels_json
            """
        ).fetchall()
    finally:
        conn.close()
    out: list[dict[str, Any]] = []
    for row in rows:
        labels_json = str(row["labels_json"] or "{}")
        try:
            labels = json.loads(labels_json)
            if not isinstance(labels, dict):
                labels = {}
        except Exception:
            labels = {}
        out.append(
            {
                "metric": str(row["metric_name"] or ""),
                "type": str(row["metric_type"] or "counter"),
                "labels": labels,
                "value": float(row["value"] or 0.0),
                "updated_at": str(row["updated_at"] or ""),
            }
        )
    return out


def render_prometheus() -> str:
    lines: list[str] = []
    for row in snapshot():
        metric = row["metric"]
        labels = row["labels"]
        value = row["value"]
        label_txt = ""
        if labels:
            rendered = []
            for key, val in sorted(labels.items()):
                escaped = str(val).replace("\\", "\\\\").replace('"', '\\"')
                rendered.append(f'{key}="{escaped}"')
            label_txt = "{" + ",".join(rendered) + "}"
        lines.append(f"{metric}{label_txt} {value}")
    if not lines:
        lines.append("medical_rag_metrics_up 1")
    return "\n".join(lines) + "\n"


def compute_summary() -> dict[str, Any]:
    rows = snapshot()
    by_name: dict[str, float] = {}
    for row in rows:
        metric = str(row["metric"])
        by_name[metric] = by_name.get(metric, 0.0) + float(row["value"])
    count = float(by_name.get("ingestion_pipeline_duration_seconds_count", 0.0))
    total = float(by_name.get("ingestion_pipeline_duration_seconds_sum", 0.0))
    avg = (total / count) if count > 0 else 0.0
    return {
        "avg_pipeline_seconds": round(avg, 3),
        "pipeline_success_total": int(by_name.get("ingestion_pipeline_success_total", 0.0)),
        "pipeline_failure_total": int(by_name.get("ingestion_pipeline_failure_total", 0.0)),
        "indexing_errors_total": int(by_name.get("ingestion_indexing_errors_total", 0.0)),
        "queue_depth": int(by_name.get("ingestion_queue_depth", 0.0)),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

