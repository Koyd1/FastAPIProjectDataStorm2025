from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Optional

import psycopg2
from psycopg2 import pool


logger = logging.getLogger(__name__)


class DatabaseService:
    def __init__(self, database_url: str) -> None:
        self._database_url = database_url
        self._pool: Optional[pool.SimpleConnectionPool] = None

    def init(self) -> None:
        if self._pool is not None:
            return
        try:
            self._pool = pool.SimpleConnectionPool(1, 5, self._database_url, sslmode="require")
            logger.info("Database connection pool initialised.")
        except Exception as exc:
            logger.exception("Failed to initialise database connection pool: %s", exc)
            self._pool = None

    @contextmanager
    def _get_connection(self):
        if self._pool is None:
            raise RuntimeError("Database pool is not initialised")
        conn = self._pool.getconn()
        try:
            yield conn
        finally:
            self._pool.putconn(conn)

    def insert_upload_metadata(
        self,
        *,
        object_key: str,
        filename: str,
        row_count: int,
        source: str = "csv",
        notes: Optional[str] = None,
    ) -> None:
        if self._pool is None:
            logger.warning("Database pool not ready; skipping metadata insert.")
            return
        query = """
            insert into anomaly.upload_batches (object_key, filename, row_count, source, notes)
            values (%s, %s, %s, %s, %s)
            on conflict (object_key) do update
                set filename = excluded.filename,
                    row_count = excluded.row_count,
                    source = excluded.source,
                    notes = excluded.notes,
                    updated_at = timezone('utc', now())
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, (object_key, filename, row_count, source, notes))
                    conn.commit()
            logger.info("Metadata for %s inserted into database.", object_key)
        except Exception as exc:
            logger.warning("Failed to insert metadata for %s: %s", object_key, exc)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.closeall()
            self._pool = None
            logger.info("Database connection pool closed.")
