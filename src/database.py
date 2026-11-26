"""
SQLite event log (standard library only - nothing to install).

One table, `events`, stores the result of every module:
    id, created_at (UTC ISO-8601), module, label, confidence, input_text, details (JSON)
The file is created automatically at database/surveillance.db.
"""
import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.config import DATABASE_DIR, DATABASE_PATH

VALID_MODULES = ("mask", "activity", "incident", "safety")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  TEXT    NOT NULL,
    module      TEXT    NOT NULL,
    label       TEXT    NOT NULL,
    confidence  REAL,
    input_text  TEXT,
    details     TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_module  ON events(module);
CREATE INDEX IF NOT EXISTS idx_events_created ON events(created_at);
"""


def _connect() -> sqlite3.Connection:
    DATABASE_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(DATABASE_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create the table if it does not exist (safe to call many times)."""
    with closing(_connect()) as conn, conn:
        conn.executescript(_SCHEMA)


def log_event(
    module: str,
    label: str,
    confidence: Optional[float] = None,
    input_text: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
) -> int:
    """Insert one event and return its id."""
    if module not in VALID_MODULES:
        raise ValueError(f"module must be one of {VALID_MODULES}")
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    details_json = json.dumps(details, default=str) if details is not None else None
    with closing(_connect()) as conn, conn:
        cur = conn.execute(
            "INSERT INTO events (created_at, module, label, confidence, input_text, details) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (created_at, module, label, confidence, input_text, details_json),
        )
        return int(cur.lastrowid)


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    d = dict(row)
    d["details"] = json.loads(d["details"]) if d["details"] else None
    return d


def get_events(limit: int = 50, module: Optional[str] = None) -> List[Dict[str, Any]]:
    """Newest events first, optionally filtered by module."""
    limit = max(1, min(int(limit), 1000))
    query, params = "SELECT * FROM events", []
    if module:
        if module not in VALID_MODULES:
            raise ValueError(f"module must be one of {VALID_MODULES}")
        query += " WHERE module = ?"
        params.append(module)
    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    with closing(_connect()) as conn:
        return [_row_to_dict(r) for r in conn.execute(query, params).fetchall()]


def get_stats() -> Dict[str, Any]:
    """Totals for the dashboard: overall, per module, and per (module, label)."""
    with closing(_connect()) as conn:
        total = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        by_module = {
            r["module"]: r["n"]
            for r in conn.execute("SELECT module, COUNT(*) AS n FROM events GROUP BY module")
        }
        by_label = [
            {"module": r["module"], "label": r["label"], "count": r["n"]}
            for r in conn.execute(
                "SELECT module, label, COUNT(*) AS n FROM events "
                "GROUP BY module, label ORDER BY n DESC"
            )
        ]
    return {"total": total, "by_module": by_module, "by_label": by_label}


def clear_events() -> int:
    """Delete all events; returns how many were removed."""
    with closing(_connect()) as conn, conn:
        return conn.execute("DELETE FROM events").rowcount


init_db()