"""Hard read-only guard for the SQL query tool.

This module provides a drop-in replacement for LangChain's `sql_db_query` tool
(`QuerySQLDatabaseTool`) that prevents any non-read-only SQL from being executed.

It is intended as a *hard* safety net against prompt injection or model mistakes.
Even if an LLM tries to run DDL/DML/admin operations, the guarded tool will
return an error instead of executing the query.

Rules enforced by `_is_safe_readonly_sql(query)`:
- Only one statement is allowed (blocks multi-statement input with `;`).
- After stripping leading SQL comments, the query must start with:
  - `SELECT`, or
  - `WITH`, or
  - `EXPLAIN`
- A small denylist of obvious dangerous keywords is blocked (e.g. `DROP`,
  `ALTER`, `OPTIMIZE`, etc.).

Quick manual checks (Django shell):

1) Open a shell:

```bat
uv run manage.py shell
```

2) Verify allow/deny behavior:

```py
>> from ai.sql_guard import _is_safe_readonly_sql

>> _is_safe_readonly_sql('SELECT 1')
(True, '')

>> _is_safe_readonly_sql('SELECT 1; SELECT 2')
(False, 'multiple statements are not allowed')

>> ok, reason = _is_safe_readonly_sql('OPTIMIZE TABLE analytics.customers FINAL;')
>> print(ok, reason)
False only SELECT/WITH/EXPLAIN queries are allowed
```
"""

import re
from typing import Any, Dict, Sequence, Union

from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from sqlalchemy.engine import Result
from sqlalchemy.exc import SQLAlchemyError
from sqlglot import exp
import sqlglot
from sqlglot.errors import ParseError

from ai.sql_dialect import SQLDialect


_ALLOWED_PREFIX_RE = re.compile(r"(?is)^\s*(select|with|explain)\b")
_FORBIDDEN_KEYWORDS_RE = re.compile(
    r"(?is)\b("
    r"insert|update|delete|drop|alter|create|truncate|"
    r"grant|revoke|attach|detach|optimize|kill|use"
    r")\b"
)


def is_safe_readonly_sql(query: str) -> tuple[bool, str]:
    q = (query or "").strip()
    if not q:
        return False, "empty query"

    # Strip leading SQL comments so "-- ...\nSELECT" is allowed.
    while True:
        q = q.lstrip()
        if q.startswith("--"):
            q = q.split("\n", 1)[1] if "\n" in q else ""
            continue
        if q.startswith("/*"):
            end = q.find("*/")
            if end == -1:
                return False, "unterminated comment"
            q = q[end + 2 :]
            continue
        break

    q = q.strip()
    if not q:
        return False, "empty query"

    # Allow at most one trailing semicolon. Block multi-statement input.
    if ";" in q.rstrip(";"):
        return False, "multiple statements are not allowed"

    if not _ALLOWED_PREFIX_RE.match(q):
        return False, "only SELECT/WITH/EXPLAIN queries are allowed"

    if _FORBIDDEN_KEYWORDS_RE.search(q):
        return False, "forbidden keyword, read-only mode only"

    return True, ""


def _readonly_rejection_message(query: str) -> str | None:
    """Return the guarded-tool error for unsafe SQL, otherwise ``None``."""
    is_safe, reason = is_safe_readonly_sql(query)
    if is_safe:
        return None
    return f"Error: forbidden SQL ({reason}). Only SELECT/WITH/EXPLAIN are allowed."


class GuardedQuerySQLDatabaseTool(QuerySQLDatabaseTool):
    """sql_db_query tool with a hard read-only guard."""

    def _run(
        self,
        query: str,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> Union[str, Sequence[Dict[str, Any]], Result]:
        """Return the LangChain tool-call result for model reasoning.

        LangChain calls this method when the model invokes the SQL query tool directly.
        It is optimized for ToolMessage-style output, not for UI table rendering. Use
        `run_structured` when backend code needs row dictionaries with column names for
        `DataTableBlock` creation.
        """

        rejection_message = _readonly_rejection_message(query)
        if rejection_message is not None:
            return rejection_message
        return super()._run(query, run_manager=run_manager)

    def run_structured(self, query: str) -> str | Sequence[Dict[str, Any]]:
        """Return backend-owned SQL rows for structured table rendering.

        This method is not exposed as an LLM tool. `execute_validated_sql` calls it after
        validation so the backend can build `DataTableBlock` rows without parsing the
        string-oriented `_run` result. It reuses the same read-only guard as `_run` and
        returns database row dictionaries with column names when execution succeeds.
        """

        rejection_message = _readonly_rejection_message(query)
        if rejection_message is not None:
            return rejection_message
        try:
            return self.db._execute(query, fetch="all")
        except SQLAlchemyError as exc:
            return f"Error: {exc}"


def resolve_sql_tables_to_allowed_names_with_ast(
    query: str,
    allowed_full_tables: set[str],
    short_name_index: dict[str, list[str]],
    dialect: SQLDialect,
) -> tuple[exp.Expression | None, str | None, list[str], dict[str, object] | None]:
    """Resolve table references to allowed fully-qualified names and return canonical SQL.

    The function parses SQL in the passed dialect, validates all referenced tables against
    `allowed_full_tables`, and rewrites unqualified table names to `db.table` when the mapping
    is unique. CTE names from `WITH ... AS (...)` are treated as local query aliases and are not
    validated against the physical table allowlist.

    Examples:
    - Input: `SELECT count(*) FROM customers`
      Allowed: `{"analytics.customers"}`
      Output SQL: `SELECT COUNT(*) AS count FROM analytics.customers`

    - Input: `SELECT * FROM default.customers`
      Allowed: `{"analytics.customers"}`
      Output: reject with `TABLE_NOT_ALLOWED`

    - Input: `SELECT * FROM customers`
      Allowed: `{"analytics.customers", "archive.customers"}`
      Output: reject with `AMBIGUOUS_UNQUALIFIED_TABLE`

    - Input: `WITH recent_sales AS (SELECT * FROM analytics.sales_fact) SELECT * FROM recent_sales`
      Allowed: `{"analytics.sales_fact"}`
      Output: SQL is accepted (`recent_sales` is a CTE name, not a physical table)
    """
    try:
        ast = sqlglot.parse_one(query, read=dialect)
    except ParseError as exc:
        return (
            None,
            None,
            [],
            {
                "type": "reject",
                "error_code": "SQL_PARSE_ERROR",
                "reason": str(exc),
            },
        )

    normalized_tables: set[str] = set()
    disallowed: list[str] = []
    ambiguous: list[str] = []
    cte_alias_names: set[str] = set()

    # handle table aliases from CTEs (WITH ... AS (...)) so they don't get validated against
    # the physical table allowlist
    with_expression = ast.args.get("with_")
    if with_expression is not None:
        for cte_expression in with_expression.expressions:
            cte_alias_name = (cte_expression.alias_or_name or "").lower()
            if cte_alias_name:
                cte_alias_names.add(cte_alias_name)

    for table in ast.find_all(exp.Table):
        table_name = (table.name or "").lower()
        if not table_name:
            continue

        db_name = (table.db or "").lower()
        if not db_name and table_name in cte_alias_names:
            continue

        if db_name:
            full_name = f"{db_name}.{table_name}"
            if full_name not in allowed_full_tables:
                disallowed.append(full_name)
                continue
            normalized_tables.add(full_name)
            continue

        matches = short_name_index.get(table_name, [])
        if len(matches) == 0:
            disallowed.append(table_name)
            continue
        if len(matches) > 1:
            ambiguous.append(table_name)
            continue

        full_name = matches[0]
        db_resolved, _ = full_name.split(".", 1)
        table.set("db", exp.to_identifier(db_resolved))
        normalized_tables.add(full_name)

    if disallowed:
        return (
            None,
            None,
            [],
            {
                "type": "reject",
                "error_code": "TABLE_NOT_ALLOWED",
                "reason": f"Disallowed tables: {', '.join(sorted(disallowed))}",
                "allowed_tables": sorted(allowed_full_tables),
            },
        )

    if ambiguous:
        return (
            None,
            None,
            [],
            {
                "type": "reject",
                "error_code": "AMBIGUOUS_UNQUALIFIED_TABLE",
                "reason": f"Ambiguous tables: {', '.join(sorted(ambiguous))}. Use db.table format.",
                "allowed_tables": sorted(allowed_full_tables),
            },
        )

    canonical_sql = ast.sql(dialect=dialect)
    return ast, canonical_sql, sorted(normalized_tables), None
