"""
Shared pytest fixtures.

db_session        — in-memory SQLite via Base.metadata.create_all (fast, for unit tests)
alembic_db_url    — session-scoped temp file DB migrated via alembic upgrade head
alembic_session   — function-scoped session against the migrated DB
"""

import os
import tempfile

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models import (  # noqa: F401 — register all models
    candle, daily_risk_state, order, paper_account,
    position, signal, strategy_config, system_event, trade,
)


# ---------------------------------------------------------------------------
# Fast in-memory DB (unit tests — uses Base.metadata, not alembic)
# ---------------------------------------------------------------------------

@pytest.fixture
def db_session():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()
    engine.dispose()


# ---------------------------------------------------------------------------
# Alembic-migrated DB (repository / migration tests)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def alembic_db_url(tmp_path_factory):
    db_file = str(tmp_path_factory.mktemp("alembic_db") / "test_migrated.db")
    db_url = f"sqlite:///{db_file}"

    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", db_url)
    command.upgrade(cfg, "head")

    return db_url


@pytest.fixture
def alembic_session(alembic_db_url):
    engine = create_engine(
        alembic_db_url,
        connect_args={"check_same_thread": False},
    )
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.rollback()
    session.close()
    engine.dispose()
