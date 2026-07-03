"""Tests for the heartbeat service."""

from __future__ import annotations

from datetime import datetime

from app.services.heartbeat import (
    RESULT_ERROR,
    RESULT_NO_NEW_CANDLE,
    RESULT_OK,
    get_latest_heartbeat,
    record_heartbeat,
)


class TestRecordHeartbeat:
    def test_inserts_row(self, db_session):
        hb = record_heartbeat(db_session, cycle_result=RESULT_OK)
        db_session.flush()
        assert hb.id is not None
        assert hb.cycle_result == RESULT_OK
        assert hb.evaluations_created == 0
        assert hb.error_message is None
        assert hb.process_pid is None

    def test_with_all_fields(self, db_session):
        ts = datetime(2025, 1, 1, 12, 0, 0)
        hb = record_heartbeat(
            db_session,
            cycle_result=RESULT_ERROR,
            evaluations_created=3,
            last_candle_time=ts,
            error_message="boom",
            process_pid=42,
        )
        db_session.flush()
        assert hb.cycle_result == RESULT_ERROR
        assert hb.evaluations_created == 3
        assert hb.last_candle_time == ts
        assert hb.error_message == "boom"
        assert hb.process_pid == 42

    def test_no_new_candle_result(self, db_session):
        hb = record_heartbeat(db_session, cycle_result=RESULT_NO_NEW_CANDLE)
        db_session.flush()
        assert hb.cycle_result == RESULT_NO_NEW_CANDLE

    def test_timestamp_naive_utc(self, db_session):
        hb = record_heartbeat(db_session, cycle_result=RESULT_OK)
        db_session.flush()
        assert hb.timestamp_utc.tzinfo is None


class TestGetLatestHeartbeat:
    def test_none_when_empty(self, db_session):
        assert get_latest_heartbeat(db_session) is None

    def test_returns_latest(self, db_session):
        record_heartbeat(
            db_session,
            cycle_result=RESULT_OK,
        )
        db_session.flush()
        # Second one — should be newer due to real-time clock
        import time

        time.sleep(0.01)
        hb2 = record_heartbeat(db_session, cycle_result=RESULT_ERROR, error_message="later")
        db_session.flush()

        latest = get_latest_heartbeat(db_session)
        assert latest is not None
        assert latest.error_message == "later"
        assert latest.id == hb2.id

    def test_single_row(self, db_session):
        hb = record_heartbeat(db_session, cycle_result=RESULT_OK)
        db_session.flush()
        assert get_latest_heartbeat(db_session) is hb
