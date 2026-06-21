import logging
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

from alembic import context

# ---------------------------------------------------------------------------
# Alembic Config object — provides access to alembic.ini
# ---------------------------------------------------------------------------
config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# ---------------------------------------------------------------------------
# Import all models so they register with Base.metadata
# ---------------------------------------------------------------------------
from app.database import Base  # noqa: E402
from app.models import (  # noqa: F401, E402
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

target_metadata = Base.metadata


# ---------------------------------------------------------------------------
# Resolve DB URL: ini value (overridden by tests) → Settings fallback
# ---------------------------------------------------------------------------
def _get_url() -> str:
    url = config.get_main_option("sqlalchemy.url")
    if url:
        return url
    from app.config import get_settings

    return get_settings().database_url


# ---------------------------------------------------------------------------
# Migration runners
# ---------------------------------------------------------------------------


def run_migrations_offline() -> None:
    url = _get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,  # required for SQLite ALTER support
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = _get_url()

    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
