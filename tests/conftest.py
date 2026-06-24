"""
Shared pytest fixtures.

db_session      — in-memory SQLite via Base.metadata.create_all (fast, for unit tests)
alembic_session — function-scoped fresh temp-file DB migrated via alembic upgrade head
                  Each test gets its own isolated DB → no cross-test contamination.
"""

import pytest
from alembic.config import Config
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from alembic import command
from app.database import Base
from app.models import (  # noqa: F401 — register all models
    candle,
    daily_risk_state,
    order,
    paper_account,
    position,
    signal,
    strategy_config,
    system_event,
    trade,
)

# ---------------------------------------------------------------------------
# Fast in-memory DB (unit tests — uses Base.metadata, not alembic)
# ---------------------------------------------------------------------------


@pytest.fixture
def db_session():
    from sqlalchemy.pool import StaticPool

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()
    engine.dispose()


# ---------------------------------------------------------------------------
# Alembic-migrated DB — fresh temp file per test (full isolation)
# ---------------------------------------------------------------------------


@pytest.fixture
def alembic_session(tmp_path):
    """
    Each test receives a brand-new SQLite file migrated via alembic upgrade head.
    Tests may commit freely; the file is discarded when the test ends.
    No test can affect another through shared state.
    """
    db_url = f"sqlite:///{tmp_path}/test.db"

    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", db_url)
    command.upgrade(cfg, "head")

    engine = create_engine(db_url, connect_args={"check_same_thread": False})
    Session = sessionmaker(bind=engine)
    session = Session()

    yield session

    session.close()
    engine.dispose()
