"""SQLite storage for classified tickets."""
import sqlite3
from contextlib import closing
from datetime import datetime, timezone


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str) -> None:
    with closing(_connect(db_path)) as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS tickets (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   text TEXT NOT NULL,
                   category TEXT NOT NULL,
                   priority TEXT NOT NULL,
                   confidence REAL NOT NULL,
                   created_at TEXT NOT NULL
               )"""
        )
        conn.commit()


def save_ticket(db_path: str, text: str, result: dict) -> int:
    with closing(_connect(db_path)) as conn:
        cur = conn.execute(
            "INSERT INTO tickets (text, category, priority, confidence, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (text, result["category"], result["priority"], result["confidence"],
             datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        return cur.lastrowid


def get_ticket(db_path: str, ticket_id: int):
    with closing(_connect(db_path)) as conn:
        row = conn.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,)).fetchone()
        return dict(row) if row else None


def list_tickets(db_path: str, limit: int = 50) -> list:
    with closing(_connect(db_path)) as conn:
        rows = conn.execute(
            "SELECT * FROM tickets ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]
