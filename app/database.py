import logging
from collections.abc import Generator
from typing import Any

from sqlalchemy import Engine, create_engine, event, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


def _get_engine() -> Engine:
    settings = get_settings()
    connect_args: dict[str, Any] = {}
    if settings.database_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False

    engine = create_engine(
        settings.database_url,
        connect_args=connect_args,
        echo=False,
    )

    if settings.database_url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_connection: Any, connection_record: Any) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return engine


engine = _get_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def run_migrations() -> None:
    """Run Alembic migrations to head. Call this at application startup."""
    import alembic.command as command
    from alembic.config import Config

    settings = get_settings()
    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", settings.database_url)
    command.upgrade(cfg, "head")
    logger.info("Database migrations applied.")


def verify_db_connection() -> None:
    """Quick connectivity check without modifying schema."""
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
