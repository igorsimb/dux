import json
import time
from typing import Annotated
from uuid import uuid4

import sqlglot
from langchain.messages import ToolMessage
from langchain.tools import InjectedToolCallId, ToolRuntime, tool
from langgraph.config import get_stream_writer
from langgraph.types import Command
from loguru import logger
from pydantic import BaseModel, ConfigDict, ValidationError
from sqlglot import exp

from ai.ai_utils.sql_tools import (
    format_sql_for_log,
    get_run_query_tool_for_source,
    get_short_name_index_for_source,
    load_allowed_source_candidates,
    load_allowed_tables_for_source,
    load_table_descriptions,
    load_table_metadata,
)
from ai.ai_utils.logging_config import build_log_preview, build_short_log_id, format_log_event
from ai.ai_utils.structured_output_blocks import (
    DEFAULT_SQL_RESULT_ROW_LIMIT,
    AgentCommentaryResponse,
    DataTableDetails,
    DataTableFacts,
    build_data_table_block,
)
from ai.ai_utils.progress_messages import (
    ExecuteValidatedSqlStage,
    GetTableDescriptionsStage,
    GetTableMetadataStage,
    ValidateSqlStage,
    execute_validated_sql_message,
    get_table_descriptions_message,
    get_table_metadata_message,
    show_progress_message,
    validate_sql_message,
)


class SubmitModelResponseLayoutArgs(BaseModel):
    """Strict tool-call schema for final SQL answer layouts."""

    model_config = ConfigDict(extra="forbid")

    layout: AgentCommentaryResponse


def _get_runtime_thread_id(tool_runtime: ToolRuntime) -> str:
    return str(((tool_runtime.config.get("configurable") or {}).get("thread_id") or ""))


def _is_structured_sql_rows(result: object) -> bool:
    return isinstance(result, list) and all(isinstance(row, dict) for row in result)


def _get_explicit_query_row_limit(query: str, dialect: str | None) -> int | None:
    """Return the user-requested SQL row limit when it is explicit in the validated query.

    The table renderer has a safe default cap, but the SQL prompt allows users to request a larger
    ranking result. This preserves that request for backend-owned table rows after SQL validation.
    For example, sqlglot normalizes T-SQL `SELECT TOP 75 * FROM dbo.customer_orders` into a limit expression
    that lets us render 75 rows instead of truncating the response at the default cap.
    """

    try:
        ast = sqlglot.parse_one(query, read=dialect or None)
    except Exception:
        return None

    limit = ast.args.get("limit")
    if isinstance(limit, exp.Limit):
        expression = limit.expression
    elif isinstance(limit, exp.Fetch):
        expression = limit.args.get("count")
    else:
        return None

    if not isinstance(expression, exp.Literal) or expression.is_string:
        return None
    try:
        row_limit = int(expression.this)
    except (TypeError, ValueError):
        return None
    return row_limit if row_limit > 0 else None


@tool
def ask_user(question: str) -> str:
    """Ask one Russian question when missing information would materially change the SQL or result."""
    return question


@tool
def get_table_descriptions() -> str:
    """List allowlisted tables with their source, SQL dialect, and business summary."""
    writer = get_stream_writer()
    show_progress_message(
        writer=writer,
        stage=get_table_descriptions_message(GetTableDescriptionsStage.START),
    )
    descriptions = load_table_descriptions()
    logger.debug(format_log_event("tool.metadata", "descriptions_ready", tables=len(descriptions)))
    return json.dumps(descriptions, ensure_ascii=True, indent=2)


@tool
def get_table_metadata(table: str) -> str:
    """Return columns and query policies for one table from `get_table_descriptions`."""
    writer = get_stream_writer()
    show_progress_message(
        writer=writer,
        stage=get_table_metadata_message(GetTableMetadataStage.START),
    )
    logger.debug(format_log_event("tool.metadata", "started", table=table))
    metadata = load_table_metadata().get(table)
    if not metadata:
        show_progress_message(
            writer=writer,
            stage=get_table_metadata_message(GetTableMetadataStage.NOT_FOUND),
        )
        logger.debug(format_log_event("tool.metadata", "rejected", table=table, reason="not_allowed"))
        return "Table not found or not allowed."
    show_progress_message(
        writer=writer,
        stage=get_table_metadata_message(GetTableMetadataStage.FOUND),
    )
    logger.debug(format_log_event("tool.metadata", "succeeded", table=table))
    return json.dumps(metadata, ensure_ascii=True, indent=2)


@tool(args_schema=SubmitModelResponseLayoutArgs, return_direct=True)
def submit_model_response_layout(
    layout: AgentCommentaryResponse | dict[str, object], tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command | dict[str, object]:
    """Submit the final layout after SQL execution.

    Include Russian commentary and one `data_table_placeholder` per result. Never include SQL rows; the backend
    inserts them.
    """

    logger.debug(
        format_log_event(
            "tool.ui",
            "layout_submitted",
            layout_type=type(layout).__name__,
            blocks=_count_layout_blocks(layout),
        )
    )
    try:
        validated_layout = AgentCommentaryResponse.model_validate(layout)
    except ValidationError as exc:
        logger.warning(
            format_log_event("tool.ui", "layout_rejected", error=build_log_preview(exc))
        )
        return {
            "type": "reject",
            "error_code": "INVALID_MODEL_RESPONSE_LAYOUT",
            "reason": str(exc),
        }

    logger.debug(
        format_log_event("tool.ui", "layout_accepted", blocks=len(validated_layout.blocks))
    )
    return Command(
        update={
            "model_response_layout": validated_layout.model_dump(exclude_none=True),
            "messages": [
                ToolMessage(
                    content="Final SQL layout submitted.",
                    tool_call_id=tool_call_id,
                    name="submit_model_response_layout",
                )
            ],
        }
    )


def _count_layout_blocks(layout: object) -> int | None:
    if isinstance(layout, AgentCommentaryResponse):
        return len(layout.blocks)
    if isinstance(layout, dict) and isinstance(layout.get("blocks"), list):
        return len(layout["blocks"])
    return None


@tool
def validate_sql(query: str, tool_runtime: ToolRuntime) -> Command | dict[str, object]:
    """Validate one read-only query against source, dialect, table, and metadata policies.

    Returns a one-time `validated_id` on success, or `error_code` and `reason` for correction.
    """

    import ai.ai_utils.validate_sql as validate_sql_util

    check_read_only = validate_sql_util.check_query_is_read_only
    evaluate_candidates = (
        validate_sql_util.validate_query_against_source_dialect_candidates
    )
    select_candidate_error = validate_sql_util.select_prioritized_candidate_error
    check_dialect_preparse_guard = validate_sql_util.check_dialect_specific_query_guards
    check_normalize = validate_sql_util.check_and_normalize_table_names
    check_allowlist = validate_sql_util.check_query_uses_only_allowlisted_tables
    check_date_filter = validate_sql_util.check_required_date_filter_for_tables

    def emit_retry_and_return_error(error: dict[str, object]) -> dict[str, object]:
        """Emit RETRY progress/log for a validation rejection and return the original error payload."""
        show_progress_message(
            writer=writer,
            stage=validate_sql_message(ValidateSqlStage.RETRY),
        )
        if error.get("error_code") == "UNBOUND_QUERY_PARAMETER":
            logger.debug(
                format_log_event(
                    "tool.sql",
                    "rejected",
                    thread=build_short_log_id(thread_id),
                    error="UNBOUND_QUERY_PARAMETER",
                    placeholders=",".join(str(placeholder) for placeholder in error.get("placeholders") or []),
                )
            )
            return error
        logger.debug(
            format_log_event(
                "tool.sql",
                "rejected",
                thread=build_short_log_id(thread_id),
                source=error.get("source_id"),
                dialect=error.get("dialect"),
                error=error.get("error_code"),
                reason=build_log_preview(error.get("reason") or ""),
            )
        )
        return error

    writer = get_stream_writer()
    thread_id = _get_runtime_thread_id(tool_runtime)
    show_progress_message(
        writer=writer,
        stage=validate_sql_message(ValidateSqlStage.START),
    )
    logger.debug(
        format_log_event(
            "tool.sql",
            "validation_started",
            thread=build_short_log_id(thread_id),
            query_len=len(query or ""),
        )
    )
    logger.debug(
        "{}\n{}\n-- end validate_sql",
        format_log_event("tool.sql", "validation_query", thread=build_short_log_id(thread_id)),
        query,
    )

    # 1. Read-only guard
    if read_only_check_error := check_read_only(query):
        return emit_retry_and_return_error(read_only_check_error)

    # 2. Candidate source+dialect discovery from allowlisted table config
    candidates = load_allowed_source_candidates()
    if not candidates:
        return emit_retry_and_return_error(
            {
                "type": "reject",
                "error_code": "NO_SOURCE_CANDIDATES",
                "reason": "No allowlisted source+dialect candidates configured.",
            }
        )

    # 3. Per-candidate guard/normalization/allowlist/date-filter checks
    candidate_successes, candidate_errors = evaluate_candidates(
        query=query,
        candidates=candidates,
        load_allowed_tables_for_source=load_allowed_tables_for_source,
        get_short_name_index_for_source=get_short_name_index_for_source,
        check_dialect_preparse_guard=check_dialect_preparse_guard,
        check_normalize=check_normalize,
        check_allowlist=check_allowlist,
        check_date_filter=check_date_filter,
    )

    # 4. Deterministic candidate resolution
    if len(candidate_successes) > 1:
        return emit_retry_and_return_error(
            {
                "type": "reject",
                "error_code": "AMBIGUOUS_SOURCE_OR_DIALECT",
                "reason": "Query is valid for multiple source/dialect candidates. Qualify tables explicitly.",
                "candidates": [
                    {
                        "source_id": candidate_success["source_id"],
                        "dialect": candidate_success["dialect"],
                    }
                    for candidate_success in candidate_successes
                ],
            }
        )

    if not candidate_successes:
        # 4.1 Reject using stable error priority for better model retries
        preferred_error = select_candidate_error(candidate_errors)
        if preferred_error:
            return emit_retry_and_return_error(preferred_error)
        return emit_retry_and_return_error(
            {
                "type": "reject",
                "error_code": "VALIDATION_FAILED_FOR_ALL_CANDIDATES",
                "reason": "Query failed validation for all configured source/dialect candidates.",
            }
        )

    selected = candidate_successes[0]
    canonical_sql = str(selected["canonical_sql"])
    normalized_referenced = selected["tables"]
    selected_source_id = str(selected["source_id"])
    selected_dialect = str(selected["dialect"])

    # 5. Capability token issuance
    validated_id = str(uuid4())
    now = time.time()
    ttl_seconds = 600  # 10 minutes
    state = tool_runtime.state if isinstance(tool_runtime.state, dict) else {}
    validated_queries = dict(state.get("validated_queries", {}))
    validated_queries[validated_id] = {
        "sql": canonical_sql,
        "source_id": selected_source_id,
        "dialect": selected_dialect,
        "tables": [str(table) for table in normalized_referenced],
        "created_at": now,
        "expires_at": now + ttl_seconds,
        "thread_id": thread_id,
        "status": "validated",
    }

    payload = {
        "type": "ok",
        "query": canonical_sql,
        "original_query": query,
        "tables": normalized_referenced,
        "source_id": selected_source_id,
        "dialect": selected_dialect,
        "validated_id": validated_id,
        "validated_at": now,
    }
    logger.debug(
        format_log_event(
            "tool.sql",
            "validation_passed",
            thread=build_short_log_id(thread_id),
            validated_id=build_short_log_id(validated_id),
            tables=",".join(str(table) for table in normalized_referenced),
            canonical_len=len(canonical_sql or query),
        )
    )
    show_progress_message(
        writer=writer,
        stage=validate_sql_message(ValidateSqlStage.OK),
    )
    return Command(
        update={
            "validated_queries": validated_queries,
            "messages": [
                ToolMessage(
                    content=json.dumps(payload, ensure_ascii=False),
                    tool_call_id=tool_runtime.tool_call_id,
                    name="validate_sql",
                )
            ],
        }
    )


@tool
def execute_validated_sql(
    validated_id: str, tool_runtime: ToolRuntime
) -> Command | dict[str, object]:
    """Execute the stored query identified by a one-time `validated_id` from `validate_sql`.

    Raw SQL is not accepted. Correct and revalidate rejected or expired queries.
    """
    writer = get_stream_writer()
    show_progress_message(
        writer=writer,
        stage=execute_validated_sql_message(ExecuteValidatedSqlStage.START),
    )
    thread_id = _get_runtime_thread_id(tool_runtime)
    logger.debug(
        format_log_event(
            "tool.sql",
            "execution_started",
            thread=build_short_log_id(thread_id),
            validated_id=build_short_log_id(validated_id),
        )
    )
    state = tool_runtime.state if isinstance(tool_runtime.state, dict) else {}
    validated_queries = dict(state.get("validated_queries", {}))
    record = validated_queries.get(validated_id)
    if not record:
        show_progress_message(
            writer=writer,
            stage=execute_validated_sql_message(ExecuteValidatedSqlStage.PROBLEM),
        )
        logger.debug(
            format_log_event(
                "tool.sql",
                "rejected",
                thread=build_short_log_id(thread_id),
                error="UNKNOWN_VALIDATED_ID",
            )
        )
        return {
            "type": "reject",
            "error_code": "UNKNOWN_VALIDATED_ID",
            "reason": f"validated_id not found: {validated_id}",
        }

    if record["status"] == "executed":
        show_progress_message(
            writer=writer,
            stage=execute_validated_sql_message(ExecuteValidatedSqlStage.PROBLEM),
        )
        logger.debug(
            format_log_event(
                "tool.sql",
                "rejected",
                thread=build_short_log_id(thread_id),
                validated_id=build_short_log_id(validated_id),
                error="VALIDATED_ID_ALREADY_USED",
            )
        )
        return {
            "type": "reject",
            "error_code": "VALIDATED_ID_ALREADY_USED",
            "reason": f"validated_id already used: {validated_id}",
        }

    if time.time() > float(record.get("expires_at", 0)):
        show_progress_message(
            writer=writer,
            stage=execute_validated_sql_message(ExecuteValidatedSqlStage.PROBLEM),
        )
        logger.debug(
            format_log_event(
                "tool.sql",
                "rejected",
                thread=build_short_log_id(thread_id),
                validated_id=build_short_log_id(validated_id),
                error="VALIDATED_ID_EXPIRED",
            )
        )
        return {
            "type": "reject",
            "error_code": "VALIDATED_ID_EXPIRED",
            "reason": f"validated_id expired: {validated_id}",
        }

    if record["thread_id"] != thread_id:
        show_progress_message(
            writer=writer,
            stage=execute_validated_sql_message(ExecuteValidatedSqlStage.PROBLEM),
        )
        logger.debug(
            format_log_event(
                "tool.sql",
                "rejected",
                thread=build_short_log_id(thread_id),
                validated_id=build_short_log_id(validated_id),
                error="THREAD_MISMATCH",
                record_thread=build_short_log_id(record.get("thread_id", "")),
            )
        )
        return {
            "type": "reject",
            "error_code": "THREAD_MISMATCH",
            "reason": "validated_id belongs to another thread",
        }

    source_id = str(record.get("source_id", "")).strip()
    if not source_id:
        show_progress_message(
            writer=writer,
            stage=execute_validated_sql_message(ExecuteValidatedSqlStage.PROBLEM),
        )
        logger.debug(
            format_log_event(
                "tool.sql",
                "rejected",
                thread=build_short_log_id(thread_id),
                validated_id=build_short_log_id(validated_id),
                error="VALIDATED_QUERY_SOURCE_MISSING",
            )
        )
        return {
            "type": "reject",
            "error_code": "VALIDATED_QUERY_SOURCE_MISSING",
            "reason": f"validated query does not include source_id: {validated_id}",
        }

    try:
        run_query_tool = get_run_query_tool_for_source(source_id)
    except Exception as exc:
        show_progress_message(
            writer=writer,
            stage=execute_validated_sql_message(ExecuteValidatedSqlStage.DB_ERROR),
        )
        logger.debug(
            format_log_event(
                "tool.sql",
                "rejected",
                thread=build_short_log_id(thread_id),
                validated_id=build_short_log_id(validated_id),
                error="SQL_QUERY_TOOL_INIT_ERROR",
                source=source_id,
                reason=build_log_preview(exc),
            )
        )
        return {
            "type": "reject",
            "error_code": "SQL_QUERY_TOOL_INIT_ERROR",
            "reason": f"failed to initialize run query tool for source '{source_id}': {exc}",
        }

    if run_query_tool is None:
        show_progress_message(
            writer=writer,
            stage=execute_validated_sql_message(ExecuteValidatedSqlStage.DB_CONNECTING),
        )
        logger.debug(
            format_log_event(
                "tool.sql",
                "rejected",
                thread=build_short_log_id(thread_id),
                validated_id=build_short_log_id(validated_id),
                error="SQL_QUERY_TOOL_NOT_CONFIGURED",
                source=source_id,
            )
        )
        return {
            "type": "reject",
            "error_code": "SQL_QUERY_TOOL_NOT_CONFIGURED",
            "reason": f"run query tool is not configured for source: {source_id}",
        }

    logger.debug(
        "{}\n{}\n-- end sql",
        format_log_event(
            "tool.sql",
            "query_ready",
            thread=build_short_log_id(thread_id),
            validated_id=build_short_log_id(validated_id),
        ),
        format_sql_for_log(record["sql"], str(record.get("dialect", "")) or None),
    )
    show_progress_message(
        writer=writer,
        stage=execute_validated_sql_message(ExecuteValidatedSqlStage.DB_WAITING),
    )
    try:
        result = run_query_tool.run_structured(record["sql"])
    except Exception as exc:
        show_progress_message(
            writer=writer,
            stage=execute_validated_sql_message(ExecuteValidatedSqlStage.DB_ERROR),
        )
        logger.debug(
            format_log_event(
                "tool.sql",
                "rejected",
                thread=build_short_log_id(thread_id),
                validated_id=build_short_log_id(validated_id),
                error="SQL_EXECUTION_ERROR",
                reason=build_log_preview(exc),
            )
        )
        return {
            "type": "reject",
            "error_code": "SQL_EXECUTION_ERROR",
            "reason": str(exc),
        }

    sql_result_table_blocks = list(state.get("sql_result_table_blocks") or [])
    if _is_structured_sql_rows(result):
        table_block = build_data_table_block(
            block_id=f"sql-result-{len(sql_result_table_blocks) + 1}",
            rows=result,
            row_limit=(
                _get_explicit_query_row_limit(record["sql"], str(record.get("dialect", "")) or None)
                or DEFAULT_SQL_RESULT_ROW_LIMIT
            ),
        )
        table_block = table_block.model_copy(
            update={
                "details": DataTableDetails(
                    facts=DataTableFacts(
                        source_id=source_id,
                        dialect=str(record.get("dialect", "")),
                        validated_id=validated_id,
                        tables=[str(table) for table in record.get("tables") or []],
                        raw_sql=format_sql_for_log(record["sql"], str(record.get("dialect", "")) or None),
                    )
                )
            }
        )
        table_block_payload = table_block.model_dump(exclude_none=True)
        sql_result_table_blocks.append(table_block_payload)
        result_content = json.dumps(table_block_payload, ensure_ascii=False, default=str)
    elif isinstance(result, str):
        result_content = result
    else:
        result_content = json.dumps(result, ensure_ascii=False, default=str)

    validated_queries[validated_id] = {**record, "status": "executed"}
    logger.debug(
        format_log_event(
            "tool.sql",
            "executed",
            thread=build_short_log_id(thread_id),
            validated_id=build_short_log_id(validated_id),
            result_type=type(result).__name__,
            result_len=len(result_content),
        )
    )
    show_progress_message(
        writer=writer,
        stage=execute_validated_sql_message(ExecuteValidatedSqlStage.FINAL_ANALYSIS),
    )

    return Command(
        update={
            "validated_queries": validated_queries,
            **(
                {"sql_result_table_blocks": sql_result_table_blocks}
                if _is_structured_sql_rows(result)
                else {}
            ),
            "messages": [
                ToolMessage(
                    content=result_content,
                    tool_call_id=tool_runtime.tool_call_id,
                    name="execute_validated_sql",
                )
            ],
        }
    )
