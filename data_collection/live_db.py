"""
Shared SQLite writer for live-fetched data.

Writes into `data/live_extension.db` — a SEPARATE file from the static Kaggle
export `data/raceform.db`, with the identical `data` table schema, so:
  - the static export is never mutated (it stays a clean, reproducible base)
  - features.py / common.py just need to UNION both files at load time
    (see docs/2026-09-14_live_data_pipeline_plan.md, "Wiring into features.py")

Rows are upserted on (date, race_id, horse) so re-running a fetch (e.g. the
evening racecard pull, then a pre-race declarations refresh) is idempotent.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "live_extension.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS data (
    date        NUMERIC,
    course      TEXT,
    race_id     INTEGER,
    off         TEXT,
    race_name   TEXT,
    type        TEXT,
    class       TEXT,
    pattern     TEXT,
    rating_band TEXT,
    age_band    TEXT,
    sex_rest    TEXT,
    dist        TEXT,
    going       TEXT,
    ran         INTEGER,
    num         INTEGER,
    pos         INTEGER,
    draw        INTEGER,
    ovr_btn     NUMERIC,
    btn         NUMERIC,
    horse       TEXT,
    age         INTEGER,
    sex         TEXT,
    wgt         TEXT,
    hg          TEXT,
    time        TEXT,
    sp          TEXT,
    jockey      TEXT,
    trainer     TEXT,
    prize       INTEGER,
    [or]        INTEGER,
    rpr         INTEGER,
    ts          INTEGER,
    sire        TEXT,
    dam         TEXT,
    damsire     TEXT,
    owner       TEXT,
    comment     TEXT,
    fetched_at  TEXT,
    PRIMARY KEY (date, race_id, horse)
);
"""

COLUMNS = [
    "date", "course", "race_id", "off", "race_name", "type", "class",
    "pattern", "rating_band", "age_band", "sex_rest", "dist", "going",
    "ran", "num", "pos", "draw", "ovr_btn", "btn", "horse", "age", "sex",
    "wgt", "hg", "time", "sp", "jockey", "trainer", "prize", "or", "rpr",
    "ts", "sire", "dam", "damsire", "owner", "comment", "fetched_at",
]


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(SCHEMA)
    return conn


def upsert_rows(conn: sqlite3.Connection, rows: list[dict]) -> int:
    """Upsert row dicts (missing keys -> NULL) keyed on (date, race_id, horse)."""
    if not rows:
        return 0
    placeholders = ", ".join("?" for _ in COLUMNS)
    col_list = ", ".join(f"[{c}]" for c in COLUMNS)
    update_clause = ", ".join(f"[{c}]=excluded.[{c}]" for c in COLUMNS if c not in ("date", "race_id", "horse"))
    sql = (
        f"INSERT INTO data ({col_list}) VALUES ({placeholders}) "
        f"ON CONFLICT(date, race_id, horse) DO UPDATE SET {update_clause}"
    )
    values = [tuple(r.get(c) for c in COLUMNS) for r in rows]
    conn.executemany(sql, values)
    conn.commit()
    return len(values)
