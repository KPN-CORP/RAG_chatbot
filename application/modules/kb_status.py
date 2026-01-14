"""Database helpers for updating knowledge base job status rows.

The watcher uses this module to mark a document row as "processing" before
ingesting it into Milvus and "done" after a successful upsert. The table name
is configured via ``config.KB_DB_TABLE`` and the database URL is pulled from
``KB_DATABASE_URL`` in the environment. Filename lookups are optionally
narrowed by ``business_unit`` derived from the file's subfolder.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from sqlalchemy import column, create_engine, func, select, table, update

from config_loader import load_config
from logging_config import get_logger
from modules.utils import sanitize_collection_component

logger = get_logger(__name__)

_TABLE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")


def _get_database_url(config: Any) -> str | None:
    """Return the KB database URL from config or environment."""

    return getattr(config, "KB_DATABASE_URL", None) or os.environ.get("KB_DATABASE_URL")


def _get_table_name(config: Any) -> str | None:
    """Validate and return the configured KB table name."""

    table_name = getattr(config, "KB_DB_TABLE", None)
    if not table_name:
        return None
    if not _TABLE_NAME_PATTERN.match(table_name):
        logger.warning("Invalid KB table name; refusing to run update", extra={"table": table_name})
        return None
    return table_name


def _normalize_label(value: str) -> str:
    """Normalize a label for comparison using the shared sanitizer."""

    return sanitize_collection_component(value).lower()


def _normalize_business_unit_column(column_expr):
    """Normalize the business_unit column similarly to folder sanitization."""

    cleaned = func.lower(func.trim(column_expr))
    cleaned = func.regexp_replace(cleaned, r"\s+", "_")
    cleaned = func.regexp_replace(cleaned, r"[^0-9a-z_]", "_")
    cleaned = func.regexp_replace(cleaned, r"_+", "_")
    return func.trim(cleaned, "_")


def _resolve_business_unit(file_path: str, config: Any) -> str | None:
    """Resolve the business unit from the file path relative to WATCH_DIR."""

    watch_dir = getattr(config, "WATCH_DIR", "")
    if not watch_dir:
        return None
    relative_path = os.path.relpath(file_path, watch_dir)
    if relative_path.startswith(".."):
        return None
    parts = relative_path.split(os.sep)
    if len(parts) < 2:
        return None
    return _normalize_label(parts[0]) or None


def _resolve_row_id(
    conn,
    table_name: str,
    *,
    filename: str,
    business_unit: str | None,
) -> int | None:
    """Look up the KB row id by filename (+ optional business_unit)."""

    # Map the file name first as requested by the ingestion workflow.
    kb_table = table(
        table_name,
        column("id"),
        column("filename"),
        column("business_unit"),
        column("status"),
    )
    select_stmt = select(kb_table.c.id).where(kb_table.c.filename == filename)
    if business_unit:
        select_stmt = select_stmt.where(
            _normalize_business_unit_column(kb_table.c.business_unit) == business_unit
        )
    select_stmt = select_stmt.order_by(kb_table.c.id.desc()).limit(1)
    row = conn.execute(select_stmt).fetchone()
    if row:
        return int(row[0])
    logger.info("KB status mapping failed; filename not found", extra={"filename": filename})
    return None


def update_kb_status(file_path: str, *, status: str, config: Any | None = None) -> None:
    """Update the status column for the row that matches ``file_path``.

    The update is best-effort: failures are logged and do not stop ingestion.
    If the file lives under a subfolder of WATCH_DIR, that folder name is
    normalized and used as the business_unit lookup to disambiguate filenames.
    """

    config = config or load_config()
    table_name = _get_table_name(config)
    db_url = _get_database_url(config)
    if not table_name or not db_url:
        logger.debug(
            "KB status update skipped; missing configuration",
            extra={"table": table_name, "has_url": bool(db_url)},
        )
        return

    filename = Path(file_path).name
    business_unit = _resolve_business_unit(str(file_path), config)
    engine = create_engine(db_url, pool_pre_ping=True)
    try:
        with engine.begin() as conn:
            row_id = _resolve_row_id(
                conn,
                table_name,
                filename=filename,
                business_unit=business_unit,
            )
            if row_id is None:
                return
            kb_table = table(table_name, column("id"), column("status"))
            update_stmt = update(kb_table).where(kb_table.c.id == row_id).values(status=status)
            conn.execute(update_stmt)
        logger.info("KB status updated", extra={"filename": filename, "status": status})
    except Exception as exc:  # pragma: no cover - defensive logging aid
        logger.warning(
            "Failed to update KB status",
            extra={"filename": filename, "status": status, "error": str(exc)},
        )


__all__ = ["update_kb_status"]
