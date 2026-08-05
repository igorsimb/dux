from __future__ import annotations

from enum import Enum


class SQLDialect(str, Enum):
    """Supported SQL dialect identifiers aligned with sqlglot names."""

    SQLITE = "sqlite"
    POSTGRES = "postgres"
    CLICKHOUSE = "clickhouse"
    MSSQL = "tsql"

    def __str__(self) -> str:
        """Return raw dialect token for APIs expecting plain strings."""

        return self.value

    @classmethod
    def supported_values(cls) -> tuple[str, ...]:
        """Return all accepted raw dialect values for validation/messages."""

        return tuple(member.value for member in cls)

    @classmethod
    def from_raw(cls, value: str | SQLDialect) -> SQLDialect:
        """Convert external config input (e.g. JSON) into a validated SQLDialect.

        This is used when reading dialect values from JSON/env/runtime input so the rest of
        the code receives a typed enum instead of untrusted raw strings.
        """

        if isinstance(value, cls):
            return value

        normalized = (value or "").strip().lower()
        for member in cls:
            if member.value == normalized:
                return member

        supported = ", ".join(cls.supported_values())
        raise ValueError(
            f"Unsupported SQL dialect: {value!r}. Supported values: {supported}"
        )
