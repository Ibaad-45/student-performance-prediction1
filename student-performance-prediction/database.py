"""
database.py
------------
Lightweight SQLite persistence layer using Python's built-in `sqlite3`
module (no ORM dependency needed for a single-table use case). Stores
every prediction made through the web app so the dashboard can display
a history table and aggregate stats.

Design notes:
    - A fresh connection is opened per operation (SQLite + Flask's
      threaded dev server play best with short-lived connections).
    - `check_same_thread=False` combined with per-call connections
      avoids the classic "SQLite objects created in a thread" error
      under Flask's development server.
"""

import sqlite3
import os
import json
from datetime import datetime, timezone

from config import Config


def get_connection():
    """Opens a new SQLite connection with row access by column name."""
    os.makedirs(os.path.dirname(Config.DATABASE_PATH), exist_ok=True)
    conn = sqlite3.connect(Config.DATABASE_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Creates the `predictions` table if it doesn't already exist.
    Safe to call every time the app starts (idempotent)."""
    conn = get_connection()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                input_json TEXT NOT NULL,
                predicted_score REAL NOT NULL,
                predicted_grade TEXT NOT NULL,
                pass_fail TEXT NOT NULL,
                pass_probability REAL NOT NULL
            )
        """)
        conn.commit()
    finally:
        conn.close()


def insert_prediction(input_data: dict, result: dict) -> int:
    """Stores one prediction record. Returns the new row's id."""
    conn = get_connection()
    try:
        cursor = conn.execute(
            """INSERT INTO predictions
               (created_at, input_json, predicted_score, predicted_grade, pass_fail, pass_probability)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                datetime.now(timezone.utc).isoformat(),
                json.dumps(input_data),
                result["predicted_score"],
                result["predicted_grade"],
                result["pass_fail"],
                result["pass_probability"],
            ),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def get_recent_predictions(limit: int = 50) -> list:
    """Returns the most recent predictions, newest first, as a list of dicts."""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        history = []
        for row in rows:
            record = dict(row)
            record["input_json"] = json.loads(record["input_json"])
            history.append(record)
        return history
    finally:
        conn.close()


def get_summary_stats() -> dict:
    """Returns aggregate stats over all stored predictions for the
    dashboard's live counters (total predictions made, pass rate, etc.)."""
    conn = get_connection()
    try:
        total = conn.execute("SELECT COUNT(*) AS c FROM predictions").fetchone()["c"]
        if total == 0:
            return {"total_predictions": 0, "average_score": 0, "pass_rate_pct": 0}

        avg_score = conn.execute("SELECT AVG(predicted_score) AS a FROM predictions").fetchone()["a"]
        pass_count = conn.execute(
            "SELECT COUNT(*) AS c FROM predictions WHERE pass_fail = 'Pass'"
        ).fetchone()["c"]

        return {
            "total_predictions": total,
            "average_score": round(avg_score, 2),
            "pass_rate_pct": round((pass_count / total) * 100, 2),
        }
    finally:
        conn.close()


def clear_history():
    """Deletes all prediction history. Exposed via a guarded API route
    for demo/reset purposes."""
    conn = get_connection()
    try:
        conn.execute("DELETE FROM predictions")
        conn.commit()
    finally:
        conn.close()
