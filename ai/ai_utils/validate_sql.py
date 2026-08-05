import re
from typing import Any, Callable

from loguru import logger
from sqlglot import exp

from ai.ai_utils.logging_config import build_log_preview, format_log_event
from ai.ai_utils.sql_tools import load_table_metadata
from ai.ai_utils.sql_tools import extract_clickhouse_placeholders
from ai.sql_dialect import SQLDialect
from ai.sql_guard import (
    is_safe_readonly_sql,
    resolve_sql_tables_to_allowed_names_with_ast,
)


def check_query_is_read_only(query: str) -> dict[str, object] | None:
    """Reject non-read-only SQL queries using the hard guard."""
    ok, reason = is_safe_readonly_sql(query)
    if ok:
        return None
    return {"type": "reject", "error_code": "READ_ONLY_GUARD", "reason": reason}


def check_dialect_specific_query_guards(
    query: str, dialect: SQLDialect
) -> dict[str, object] | None:
    """Reject dialect-specific placeholder misuse before SQL AST processing."""
    placeholders = extract_clickhouse_placeholders(query)
    if dialect == SQLDialect.CLICKHOUSE and placeholders:
        return {
            "type": "reject",
            "error_code": "UNBOUND_QUERY_PARAMETER",
            "reason": (
                "ClickHouse placeholders are not supported in this execution flow. "
                "Use SQL literals instead of {name:Type}."
            ),
            "placeholders": placeholders,
        }
    return None


def check_query_uses_only_allowlisted_tables(
    table_name_normalization_error: dict[str, Any] | None,
) -> dict[str, object] | None:
    """Return allowlist rejection if normalization produced TABLE_NOT_ALLOWED."""
    if (
        table_name_normalization_error
        and table_name_normalization_error.get("error_code") == "TABLE_NOT_ALLOWED"
    ):
        return table_name_normalization_error
    return None


def check_and_normalize_table_names(
    query: str,
    dialect: SQLDialect,
    allowed_full_tables: set[str],
    short_name_index: dict[str, list[str]],
) -> tuple[exp.Expression | None, str | None, list[str], dict[str, Any] | None]:
    """Resolve SQL tables against allowlist and normalize unqualified names to db.table."""
    return resolve_sql_tables_to_allowed_names_with_ast(
        query=query,
        allowed_full_tables=allowed_full_tables,
        short_name_index=short_name_index,
        dialect=dialect,
    )


def _collect_filter_columns(parsed_ast: exp.Expression | None) -> list[tuple[str, str]]:
    """Return columns referenced in SELECT filters as (qualifier, column_name).

    We scan `WHERE`, `PREWHERE`, and `HAVING` expressions for each `SELECT`.

    Example output:
    - `[('', 'event_date')]` for `WHERE event_date >= today() - 7`
    - `[('sf', 'event_date'), ('sf', 'region')]` for a date and region filter on alias `sf`
    """
    if parsed_ast is None:
        return []

    columns: list[tuple[str, str]] = []
    for select_expr in parsed_ast.find_all(exp.Select):
        for arg_name in ("where", "prewhere", "having"):
            filter_expr = select_expr.args.get(arg_name)
            if filter_expr is None:
                continue
            for column in filter_expr.find_all(exp.Column):
                column_name = (column.name or "").lower()
                if not column_name:
                    continue
                qualifier = (column.table or "").lower()
                columns.append((qualifier, column_name))
    return columns


def _normalized_table_reference(table: exp.Table) -> tuple[str, str, str] | None:
    table_name = (table.name or "").lower()
    if not table_name:
        return None
    db_name = (table.db or "").lower()
    full_name = f"{db_name}.{table_name}" if db_name else table_name
    return table_name, db_name, full_name


def _build_table_identifier_index(
    parsed_ast: exp.Expression | None,
) -> dict[str, set[str]]:
    """Map table/alias identifiers to normalized full table names.

    Keys include short table names, full names, and aliases when present.

    Example output:
    - `{'sales_fact': {'analytics.sales_fact'}}`
    - `{'sf': {'analytics.sales_fact'}}`
    - `{'analytics.sales_fact': {'analytics.sales_fact'}}`
    """
    identifier_index: dict[str, set[str]] = {}
    if parsed_ast is None:
        return identifier_index

    for table in parsed_ast.find_all(exp.Table):
        table_reference = _normalized_table_reference(table)
        if table_reference is None:
            continue
        table_name, _db_name, full_name = table_reference
        identifier_index.setdefault(table_name, set()).add(full_name)
        identifier_index.setdefault(full_name, set()).add(full_name)

        alias_name = (table.alias_or_name or "").lower()
        if alias_name:
            identifier_index.setdefault(alias_name, set()).add(full_name)

    for cte in parsed_ast.find_all(exp.CTE):
        cte_name = (cte.alias_or_name or "").lower()
        if not cte_name:
            continue

        for table in cte.this.find_all(exp.Table):
            table_reference = _normalized_table_reference(table)
            if table_reference is None:
                continue
            _table_name, db_name, full_name = table_reference
            if not db_name:
                continue
            identifier_index.setdefault(cte_name, set()).add(full_name)

    return identifier_index


def check_required_date_filter_for_tables(
    parsed_ast: exp.Expression | None,
    normalized_referenced_tables: list[str],
) -> dict[str, object] | None:
    """Reject when a table that needs a date filter has none in query filters.

    For each referenced table with `requires_date_filter=true`, this checks whether its
    `date_column` appears in `WHERE`, `PREWHERE`, or `HAVING`.
    """
    metadata = load_table_metadata()
    required_tables: dict[str, str] = {}
    for table_name in normalized_referenced_tables:
        table_meta = metadata.get(table_name)
        if not isinstance(table_meta, dict):
            continue
        if not bool(table_meta.get("requires_date_filter")):
            continue
        date_column = str(table_meta.get("date_column", "")).strip().lower()
        if not date_column:
            continue
        required_tables[table_name] = date_column

    if not required_tables:
        return None

    filter_columns = _collect_filter_columns(parsed_ast)
    if not filter_columns:
        return {
            "type": "reject",
            "error_code": "MISSING_REQUIRED_DATE_FILTER",
            "reason": "Required date filter is missing for one or more tables.",
            "tables_missing_date_filter": sorted(required_tables),
        }

    identifier_index = _build_table_identifier_index(parsed_ast)
    total_referenced_tables = len(set(normalized_referenced_tables))
    covered_tables: set[str] = set()

    for qualifier, column_name in filter_columns:
        if qualifier:
            matched_tables = identifier_index.get(qualifier, set())
            for matched_table in matched_tables:
                required_column = required_tables.get(matched_table)
                if required_column and column_name == required_column:
                    covered_tables.add(matched_table)
            continue

        if total_referenced_tables == 1:
            only_table = next(iter(required_tables))
            if column_name == required_tables[only_table]:
                covered_tables.add(only_table)

    missing_tables = sorted(set(required_tables) - covered_tables)
    if not missing_tables:
        return None

    return {
        "type": "reject",
        "error_code": "MISSING_REQUIRED_DATE_FILTER",
        "reason": "Missing required date predicate on metadata-defined date column.",
        "tables_missing_date_filter": missing_tables,
    }


def _query_references_full_table_name(query: str, full_table_name: str) -> bool:
    """Return True when query text explicitly references an allowlisted full table name."""
    if not query or not full_table_name or "." not in full_table_name:
        return False
    pattern = rf"(?<![A-Za-z0-9_$.]){re.escape(full_table_name)}(?![A-Za-z0-9_$])"
    return re.search(pattern, query, flags=re.IGNORECASE) is not None


def narrow_candidates_to_explicit_table_sources(
    *,
    query: str,
    candidates: list[tuple[str, SQLDialect]],
    load_allowed_tables_for_source: Callable[[str], set[str]],
) -> tuple[list[tuple[str, SQLDialect]], dict[str, set[str]], list[str]]:
    """Limit candidate sources when SQL explicitly references allowlisted full table names.

    `ai/table_descriptions.json` is the table allowlist source of truth. If the query names
    `analytics.some_table` and that full table exists only under `clickhouse_default`, errors from
    unrelated sources should not be allowed to mask the ClickHouse validation result.
    """
    allowed_tables_by_source: dict[str, set[str]] = {}
    referenced_source_ids: set[str] = set()
    referenced_tables: set[str] = set()

    for source_id, _dialect in candidates:
        allowed_tables = load_allowed_tables_for_source(source_id)
        allowed_tables_by_source[source_id] = allowed_tables
        for table_name in allowed_tables:
            if not _query_references_full_table_name(query, table_name):
                continue
            referenced_source_ids.add(source_id)
            referenced_tables.add(table_name)

    if not referenced_source_ids:
        return candidates, allowed_tables_by_source, []

    narrowed_candidates = [
        (source_id, dialect)
        for source_id, dialect in candidates
        if source_id in referenced_source_ids
    ]
    return narrowed_candidates, allowed_tables_by_source, sorted(referenced_tables)


def validate_query_against_source_dialect_candidates(
    *,
    query: str,
    candidates: list[tuple[str, SQLDialect]],
    load_allowed_tables_for_source: Callable[[str], set[str]],
    get_short_name_index_for_source: Callable[[str], dict[str, list[str]]],
    check_dialect_preparse_guard: Callable[[str, SQLDialect], dict[str, object] | None],
    check_normalize: Callable[
        [str, SQLDialect, set[str], dict[str, list[str]]],
        tuple[exp.Expression | None, str | None, list[str], dict[str, Any] | None],
    ],
    check_allowlist: Callable[[dict[str, Any] | None], dict[str, object] | None],
    check_date_filter: Callable[
        [exp.Expression | None, list[str]], dict[str, object] | None
    ],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Validate one SQL query against all allowlisted `(source_id, dialect)` candidates.

    A candidate is a routing option derived from table config, for example:
    - `("clickhouse_default", SQLDialect.CLICKHOUSE)`
    - `("mssql_default", SQLDialect.MSSQL)`

    For each candidate this function:
    1) runs candidate pre-parse guards,
    2) loads source-scoped allowlisted tables,
    3) parses + normalizes table names to canonical `db.table` form,
    4) captures allowlist or normalization errors.

    Returns:
    - `candidate_successes`: entries with `source_id`, `dialect`, canonical SQL and normalized tables.
    - `candidate_errors`: collected reject payloads from failed candidates.

    Example:
    - Query: `SELECT * FROM sales_orders`
    - Candidates: ClickHouse + MSSQL
    - Outcome: ClickHouse fails allowlist, MSSQL succeeds -> one success for `mssql_default`.
    """
    candidate_successes: list[dict[str, object]] = []
    candidate_errors: list[dict[str, object]] = []
    candidates_to_evaluate, allowed_tables_by_source, referenced_tables = narrow_candidates_to_explicit_table_sources(
        query=query,
        candidates=candidates,
        load_allowed_tables_for_source=load_allowed_tables_for_source,
    )
    if len(candidates_to_evaluate) != len(candidates):
        logger.debug(
            format_log_event(
                "tool.sql",
                "candidates_narrowed",
                original=len(candidates),
                narrowed=len(candidates_to_evaluate),
                referenced_tables=",".join(referenced_tables),
            )
        )

    for source_id, dialect in candidates_to_evaluate:
        logger.debug(
            format_log_event("tool.sql", "candidate_started", source=source_id, dialect=dialect)
        )
        syntax_check_error = check_dialect_preparse_guard(query, dialect)
        if syntax_check_error:
            syntax_check_error = {
                **syntax_check_error,
                "source_id": source_id,
                "dialect": str(dialect),
            }
            logger.debug(
                format_log_event(
                    "tool.sql",
                    "candidate_rejected",
                    source=source_id,
                    dialect=dialect,
                    error=syntax_check_error.get("error_code"),
                    reason=build_log_preview(syntax_check_error.get("reason") or ""),
                )
            )
            candidate_errors.append(syntax_check_error)
            continue

        allowed_tables = allowed_tables_by_source.get(source_id)
        if allowed_tables is None:
            allowed_tables = load_allowed_tables_for_source(source_id)
        short_index = get_short_name_index_for_source(source_id)
        logger.debug(
            format_log_event(
                "tool.sql",
                "allowlist_loaded",
                source=source_id,
                dialect=dialect,
                tables=len(allowed_tables),
            )
        )
        (
            parsed_ast,
            canonical_sql,
            normalized_referenced,
            table_name_normalization_error,
        ) = check_normalize(query, dialect, allowed_tables, short_index)

        allowlist_check_error = check_allowlist(table_name_normalization_error)
        if allowlist_check_error:
            allowlist_check_error = {
                **allowlist_check_error,
                "source_id": source_id,
                "dialect": str(dialect),
            }
            logger.debug(
                format_log_event(
                    "tool.sql",
                    "candidate_rejected",
                    source=source_id,
                    dialect=dialect,
                    error=allowlist_check_error.get("error_code"),
                    reason=build_log_preview(allowlist_check_error.get("reason") or ""),
                    allowed_tables=len(allowed_tables),
                )
            )
            candidate_errors.append(allowlist_check_error)
            continue
        if table_name_normalization_error:
            table_name_normalization_error = {
                **table_name_normalization_error,
                "source_id": source_id,
                "dialect": str(dialect),
            }
            logger.debug(
                format_log_event(
                    "tool.sql",
                    "candidate_rejected",
                    source=source_id,
                    dialect=dialect,
                    error=table_name_normalization_error.get("error_code"),
                    reason=build_log_preview(table_name_normalization_error.get("reason") or ""),
                )
            )
            candidate_errors.append(table_name_normalization_error)
            continue

        date_filter_check_error = check_date_filter(parsed_ast, normalized_referenced)
        if date_filter_check_error:
            date_filter_check_error = {
                **date_filter_check_error,
                "source_id": source_id,
                "dialect": str(dialect),
            }
            logger.debug(
                format_log_event(
                    "tool.sql",
                    "candidate_rejected",
                    source=source_id,
                    dialect=dialect,
                    error=date_filter_check_error.get("error_code"),
                    reason=build_log_preview(date_filter_check_error.get("reason") or ""),
                    tables=",".join(normalized_referenced),
                )
            )
            candidate_errors.append(date_filter_check_error)
            continue

        logger.debug(
            format_log_event(
                "tool.sql",
                "candidate_passed",
                source=source_id,
                dialect=dialect,
                tables=",".join(normalized_referenced),
                canonical_len=len(canonical_sql or query),
            )
        )
        candidate_successes.append(
            {
                "source_id": source_id,
                "dialect": str(dialect),
                "canonical_sql": canonical_sql or query,
                "tables": normalized_referenced,
            }
        )

    return candidate_successes, candidate_errors


def select_prioritized_candidate_error(
    candidate_errors: list[dict[str, object]],
) -> dict[str, object] | None:
    """Select one deterministic reject payload from candidate errors by priority.

    Multiple candidates can fail for different reasons. This function keeps the returned
    error stable and actionable so retry behavior is deterministic.

    Priority order:
    1) `TABLE_NOT_ALLOWED`
    2) `AMBIGUOUS_UNQUALIFIED_TABLE`
    3) `MISSING_REQUIRED_DATE_FILTER`
    4) `UNBOUND_QUERY_PARAMETER`
    5) `SQL_PARSE_ERROR`

    Example:
    - Candidate A: `SQL_PARSE_ERROR`
    - Candidate B: `TABLE_NOT_ALLOWED`
    - Returned error: `TABLE_NOT_ALLOWED`.
    """
    error_priority = [
        "TABLE_NOT_ALLOWED",
        "AMBIGUOUS_UNQUALIFIED_TABLE",
        "MISSING_REQUIRED_DATE_FILTER",
        "UNBOUND_QUERY_PARAMETER",
        "SQL_PARSE_ERROR",
    ]
    for error_code in error_priority:
        matched = next(
            (err for err in candidate_errors if err.get("error_code") == error_code),
            None,
        )
        if matched:
            return matched
    return None
