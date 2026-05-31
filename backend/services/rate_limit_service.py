from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from backend.database import db_connect, now_iso


@dataclass
class RateLimitResult:
    allowed: bool
    limit: int
    count: int
    window_seconds: int
    retry_after_seconds: int


def _window_epoch(now_ts: int, window_seconds: int) -> int:
    return now_ts - (now_ts % max(1, int(window_seconds)))


def enforce_limit(*, scope: str, key: str, limit: int, window_seconds: int) -> RateLimitResult:
    normalized_scope = str(scope or "").strip() or "default"
    normalized_key = str(key or "").strip() or "anonymous"
    safe_limit = max(1, int(limit))
    safe_window = max(1, int(window_seconds))
    now_ts = int(datetime.now(timezone.utc).timestamp())
    bucket_start = _window_epoch(now_ts, safe_window)
    bucket_end = bucket_start + safe_window
    retry_after = max(1, bucket_end - now_ts)

    conn = db_connect()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO rate_limit_hits (scope, rl_key, window_start_epoch, count, updated_at)
            VALUES (?, ?, ?, 1, ?)
            ON CONFLICT(scope, rl_key, window_start_epoch)
            DO UPDATE SET
                count = count + 1,
                updated_at = excluded.updated_at
            """,
            (normalized_scope, normalized_key, bucket_start, now_iso()),
        )
        row = cur.execute(
            """
            SELECT count
            FROM rate_limit_hits
            WHERE scope = ? AND rl_key = ? AND window_start_epoch = ?
            """,
            (normalized_scope, normalized_key, bucket_start),
        ).fetchone()
        count = int(row["count"] if row else 0)
        # Best-effort cleanup of old buckets.
        cur.execute(
            "DELETE FROM rate_limit_hits WHERE window_start_epoch < ?",
            (bucket_start - (safe_window * 4),),
        )
        conn.commit()
    finally:
        conn.close()

    return RateLimitResult(
        allowed=(count <= safe_limit),
        limit=safe_limit,
        count=count,
        window_seconds=safe_window,
        retry_after_seconds=retry_after,
    )

