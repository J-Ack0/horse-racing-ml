"""
Tests for live_db.py — schema creation and idempotent upsert-on-(date,race_id,horse).
Each test gets its own throwaway sqlite file (never touches the real
data/live_extension.db).
"""
import sqlite3

import pytest

import live_db


@pytest.fixture
def conn(tmp_path, monkeypatch):
    db_path = tmp_path / "test_live.db"
    monkeypatch.setattr(live_db, "DB_PATH", db_path)
    c = live_db.connect()
    yield c
    c.close()


def make_row(**overrides):
    row = {c: None for c in live_db.COLUMNS}
    row.update({"date": "2026-09-16", "race_id": "r1", "horse": "Great Blasket"})
    row.update(overrides)
    return row


def test_connect_creates_schema(conn):
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    assert ("data",) in tables


def test_upsert_inserts_new_row(conn):
    n = live_db.upsert_rows(conn, [make_row(jockey="Jason Hart")])
    assert n == 1
    row = conn.execute("SELECT jockey FROM data WHERE race_id='r1' AND horse='Great Blasket'").fetchone()
    assert row == ("Jason Hart",)


def test_upsert_same_key_updates_not_duplicates(conn):
    live_db.upsert_rows(conn, [make_row(jockey="Jason Hart", draw=9)])
    live_db.upsert_rows(conn, [make_row(jockey="New Jockey", draw=9)])  # declaration change
    rows = conn.execute("SELECT jockey FROM data WHERE race_id='r1' AND horse='Great Blasket'").fetchall()
    assert len(rows) == 1
    assert rows[0] == ("New Jockey",)


def test_upsert_different_horses_same_race_both_kept(conn):
    live_db.upsert_rows(conn, [make_row(horse="Horse A"), make_row(horse="Horse B")])
    count = conn.execute("SELECT COUNT(*) FROM data WHERE race_id='r1'").fetchone()[0]
    assert count == 2


def test_upsert_same_horse_different_race_both_kept(conn):
    live_db.upsert_rows(conn, [make_row(race_id="r1"), make_row(race_id="r2")])
    count = conn.execute("SELECT COUNT(*) FROM data WHERE horse='Great Blasket'").fetchone()[0]
    assert count == 2


def test_upsert_empty_list_is_noop(conn):
    n = live_db.upsert_rows(conn, [])
    assert n == 0
    count = conn.execute("SELECT COUNT(*) FROM data").fetchone()[0]
    assert count == 0


def test_upsert_missing_keys_become_null(conn):
    partial_row = {"date": "2026-09-16", "race_id": "r1", "horse": "Great Blasket", "jockey": "Jason Hart"}
    live_db.upsert_rows(conn, [partial_row])
    row = conn.execute("SELECT [or], sire FROM data WHERE race_id='r1'").fetchone()
    assert row == (None, None)


def test_upsert_preserves_fetched_at_on_new_insert(conn):
    live_db.upsert_rows(conn, [make_row(fetched_at="2026-09-16T18:00:00Z")])
    row = conn.execute("SELECT fetched_at FROM data WHERE race_id='r1'").fetchone()
    assert row == ("2026-09-16T18:00:00Z",)


def test_upsert_updates_fetched_at_on_redeclaration(conn):
    live_db.upsert_rows(conn, [make_row(fetched_at="2026-09-16T18:00:00Z")])
    live_db.upsert_rows(conn, [make_row(fetched_at="2026-09-16T13:00:00Z")])  # pre-race refresh
    row = conn.execute("SELECT fetched_at FROM data WHERE race_id='r1'").fetchone()
    assert row == ("2026-09-16T13:00:00Z",)


def test_connect_is_reentrant(tmp_path, monkeypatch):
    db_path = tmp_path / "reentrant.db"
    monkeypatch.setattr(live_db, "DB_PATH", db_path)
    c1 = live_db.connect()
    live_db.upsert_rows(c1, [make_row()])
    c1.close()
    c2 = live_db.connect()  # must not fail / must not wipe the table
    count = c2.execute("SELECT COUNT(*) FROM data").fetchone()[0]
    assert count == 1
    c2.close()
