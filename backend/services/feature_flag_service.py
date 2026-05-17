from __future__ import annotations

from typing import Any

from backend.database import db_connect, now_iso


DEFAULT_FLAGS: tuple[tuple[str, int, str], ...] = (
    (
        "REFERENCE_RANGE_STRICT_MODE",
        1,
        "Enable strict deterministic reference range lookup flow",
    ),
)


def ensure_feature_flags_seeded() -> None:
    conn = db_connect()
    try:
        cur = conn.cursor()
        for name, enabled, description in DEFAULT_FLAGS:
            cur.execute(
                """
                INSERT INTO feature_flags (name, enabled, description, updated_at, updated_by)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(name) DO NOTHING
                """,
                (name, int(enabled), description, now_iso(), "system:feature_flag_seed"),
            )
        conn.commit()
    finally:
        conn.close()


def get_feature_flag(name: str) -> bool:
    ensure_feature_flags_seeded()
    conn = db_connect()
    try:
        row = conn.execute("SELECT enabled FROM feature_flags WHERE name = ? LIMIT 1", (str(name),)).fetchone()
        if row is None:
            return False
        return bool(int(row["enabled"]))
    finally:
        conn.close()


def set_feature_flag(name: str, enabled: bool, updated_by: str | None = None) -> None:
    ensure_feature_flags_seeded()
    conn = db_connect()
    try:
        existing = conn.execute("SELECT description FROM feature_flags WHERE name = ? LIMIT 1", (str(name),)).fetchone()
        description = str(existing["description"] or "") if existing else ""
        conn.execute(
            """
            INSERT INTO feature_flags (name, enabled, description, updated_at, updated_by)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                enabled=excluded.enabled,
                updated_at=excluded.updated_at,
                updated_by=excluded.updated_by
            """,
            (
                str(name),
                1 if bool(enabled) else 0,
                description,
                now_iso(),
                str(updated_by) if updated_by else None,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def list_feature_flags() -> list[dict[str, Any]]:
    ensure_feature_flags_seeded()
    conn = db_connect()
    try:
        rows = conn.execute(
            """
            SELECT name, enabled, description, updated_at, updated_by
            FROM feature_flags
            ORDER BY name ASC
            """
        ).fetchall()
        return [
            {
                "name": str(r["name"]),
                "enabled": bool(int(r["enabled"])),
                "description": str(r["description"] or ""),
                "updated_at": str(r["updated_at"] or ""),
                "updated_by": str(r["updated_by"] or ""),
            }
            for r in rows
        ]
    finally:
        conn.close()
