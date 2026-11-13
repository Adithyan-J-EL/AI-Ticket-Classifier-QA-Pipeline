import sqlite3
from datetime import datetime

import pytest

from app import database

SAMPLE = {"category": "billing", "priority": "high", "confidence": 0.9}


def test_save_and_get_ticket(db_path):
    database.init_db(db_path)
    ticket_id = database.save_ticket(db_path, "refund please", SAMPLE)
    row = database.get_ticket(db_path, ticket_id)
    assert row["text"] == "refund please"
    assert row["category"] == "billing"
    assert row["priority"] == "high"
    assert row["confidence"] == 0.9


def test_created_at_is_valid_utc_iso_timestamp(db_path):
    database.init_db(db_path)
    row = database.get_ticket(db_path, database.save_ticket(db_path, "x ticket", SAMPLE))
    parsed = datetime.fromisoformat(row["created_at"])
    assert parsed.tzinfo is not None


def test_missing_ticket_returns_none(db_path):
    database.init_db(db_path)
    assert database.get_ticket(db_path, 999) is None


def test_list_is_newest_first(db_path):
    database.init_db(db_path)
    database.save_ticket(db_path, "first ticket", SAMPLE)
    database.save_ticket(db_path, "second ticket", SAMPLE)
    texts = [t["text"] for t in database.list_tickets(db_path)]
    assert texts == ["second ticket", "first ticket"]


def test_list_respects_limit(db_path):
    database.init_db(db_path)
    for i in range(5):
        database.save_ticket(db_path, f"ticket {i}", SAMPLE)
    assert len(database.list_tickets(db_path, limit=3)) == 3


def test_ids_are_unique_and_increasing(db_path):
    database.init_db(db_path)
    ids = [database.save_ticket(db_path, f"ticket {i}", SAMPLE) for i in range(3)]
    assert ids == sorted(set(ids))


def test_init_db_is_idempotent_and_keeps_data(db_path):
    database.init_db(db_path)
    database.save_ticket(db_path, "keep me", SAMPLE)
    database.init_db(db_path)  # second call must not wipe the table
    assert len(database.list_tickets(db_path)) == 1


def test_sql_injection_text_is_stored_literally(db_path):
    database.init_db(db_path)
    evil = "x'); DROP TABLE tickets; --"
    ticket_id = database.save_ticket(db_path, evil, SAMPLE)
    assert database.get_ticket(db_path, ticket_id)["text"] == evil
    assert len(database.list_tickets(db_path)) == 1  # table still exists


def test_unicode_roundtrip(db_path):
    database.init_db(db_path)
    text = "Café ☕ 日本語 😀"
    ticket_id = database.save_ticket(db_path, text, SAMPLE)
    assert database.get_ticket(db_path, ticket_id)["text"] == text


def test_not_null_constraints_are_enforced(db_path):
    database.init_db(db_path)
    with pytest.raises((sqlite3.IntegrityError, TypeError, KeyError)):
        database.save_ticket(db_path, None, SAMPLE)
