from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Generic, Iterator, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class SqliteStore(Generic[K, V]):
    def __init__(self, sqlite_path: str | os.PathLike[str]):
        self._sqlite_path = str(Path(sqlite_path))
        self._connection = sqlite3.connect(self._sqlite_path)
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS store_entries (id TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self._connection.commit()
        self._value_column = self._resolve_value_column_name()

    def _resolve_value_column_name(self) -> str:
        rows = self._connection.execute("PRAGMA table_info(store_entries)").fetchall()
        column_names = [str(row[1]) for row in rows]
        if "value" in column_names:
            return "value"
        if "payload" in column_names:
            return "payload"
        raise AssertionError(
            f"store_entries must contain a value/payload column, got: {column_names}"
        )

    @staticmethod
    def _encode(value: object) -> str:
        return json.dumps(value)

    @staticmethod
    def _decode(value: str) -> object:
        return json.loads(value)

    @staticmethod
    def _to_store_key(entry_id: object) -> str:
        return str(entry_id)

    def upsert(self, entry_id: K, value: V) -> None:
        key = self._to_store_key(entry_id)
        encoded = self._encode(value)
        self._connection.execute(
            f"INSERT INTO store_entries (id, {self._value_column}) VALUES (?, ?) "
            f"ON CONFLICT(id) DO UPDATE SET {self._value_column}=excluded.{self._value_column}",
            (key, encoded),
        )
        self._connection.commit()

    def get(self, entry_id: K) -> V | None:
        key = self._to_store_key(entry_id)
        row = self._connection.execute(
            f"SELECT {self._value_column} FROM store_entries WHERE id = ?",
            (key,),
        ).fetchone()
        if row is None:
            return None
        return self._decode(str(row[0]))  # type: ignore[return-value]

    def load_all(self) -> Iterator[V]:
        rows = self._connection.execute(
            f"SELECT {self._value_column} FROM store_entries ORDER BY CAST(id AS INTEGER), id"
        )
        for row in rows:
            yield self._decode(str(row[0]))  # type: ignore[misc, return-value]

    def clear(self) -> None:
        self._connection.execute("DELETE FROM store_entries")
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()
