"""User-facing progress messages for tool execution stages.

Stage semantics (runtime meaning):

- GetTableDescriptionsStage.START:
  Called at the start of `get_table_descriptions`, before reading the allowlisted
  table catalog from JSON.

- GetTableMetadataStage.START:
  Called at the start of `get_table_metadata`, before lookup for the requested
  table card in metadata JSON.
- GetTableMetadataStage.FOUND:
  Called after metadata lookup succeeds and before returning that metadata.
- GetTableMetadataStage.NOT_FOUND:
  Called when requested table is missing or not allowlisted.

- ValidateSqlStage.START:
  Called at the start of `validate_sql`, before read-only guard and dialect checks.
- ValidateSqlStage.RETRY:
  Called on any validation reject path (readonly reject, placeholder reject, or
  table-resolution error) to indicate rewrite/retry is needed.
- ValidateSqlStage.OK:
  Called after validation succeeds and validated token payload is ready.

- ExecuteValidatedSqlStage.START:
  Called at the start of `execute_validated_sql`, before validated_id checks.
- ExecuteValidatedSqlStage.PROBLEM:
  Called when validated_id checks fail (unknown, already used, expired, or wrong
  thread).
- ExecuteValidatedSqlStage.DB_CONNECTING:
  Called when execution cannot proceed because query tool is not configured.
- ExecuteValidatedSqlStage.DB_WAITING:
  Called immediately before invoking the DB query tool (waiting for DB response).
- ExecuteValidatedSqlStage.DB_ERROR:
  Called if DB invocation raises an exception.
- ExecuteValidatedSqlStage.FINAL_ANALYSIS:
  Called after successful DB result is received and before returning ToolMessage.

"""

from __future__ import annotations

import random
from enum import Enum
from typing import Callable, Mapping, TypeVar

StageEnum = TypeVar("StageEnum", bound=Enum)  # generic typing so editors don't complain

DEFAULT_PROGRESS_FALLBACK_MESSAGE = "is thinking..."


class GetTableDescriptionsStage(str, Enum):
    """Stages for table-description discovery progress."""

    START = "start"


class GetTableMetadataStage(str, Enum):
    """Stages for detailed table-metadata lookup progress."""

    START = "start"
    FOUND = "found"
    NOT_FOUND = "not_found"


class ValidateSqlStage(str, Enum):
    """Stages for SQL validation progress."""

    START = "start"
    RETRY = "retry"
    OK = "ok"


class ExecuteValidatedSqlStage(str, Enum):
    """Stages for validated SQL execution progress."""

    START = "start"
    PROBLEM = "problem"
    DB_CONNECTING = "db_connecting"
    DB_WAITING = "db_waiting"
    DB_ERROR = "db_error"
    FINAL_ANALYSIS = "final_analysis"


# Extend the tuples of any stage. Message for corresponding stage will be picked at random at runtime.
_GET_TABLE_DESCRIPTIONS_MESSAGES: dict[GetTableDescriptionsStage, tuple[str, ...]] = {
    GetTableDescriptionsStage.START: (
        "is checking available tables",
        "is reviewing the list of available tables",
        "is browsing available tables",
        "is selecting relevant tables",
        "is confirming which tables are available",
        "is building the list of available tables",
        "is searching for relevant tables",
        "is analyzing available tables",
        "is identifying the required tables",
        "is matching the request to available tables",
        "is opening the table catalog",
        "is reviewing allowlisted tables",
        "is searching for tables that fit the task",
        "is locating the relevant data",
        "is checking the available data map",
        "is building a shortlist of tables",
        "is reviewing available data sources",
        "is selecting candidate tables",
        "is comparing the request with table descriptions",
        "is preparing tables for detailed review",
    ),
}

_GET_TABLE_METADATA_MESSAGES: dict[GetTableMetadataStage, tuple[str, ...]] = {
    GetTableMetadataStage.START: (
        "is examining the table structure",
        "is reviewing table fields and relationships",
        "is checking the table columns",
        "is reading the table description",
        "is reviewing the table contents",
        "is analyzing the table schema",
        "is checking the table metadata",
        "is reviewing the table card",
        "is verifying the table structure",
        "is identifying the required table fields",
        "is reading the table details",
        "is opening the selected table schema",
        "is reviewing field types and purposes",
        "is checking which columns are available",
        "is comparing the table with its description",
        "is examining the available table fields",
        "is checking the table query rules",
        "is searching metadata for the required columns",
        "is reviewing the table constraints",
        "is opening the technical table card",
    ),
    GetTableMetadataStage.FOUND: (
        "is loading the table description",
        "is finding the table description",
        "is confirming the table structure",
        "is collecting the table details",
        "is preparing the table description",
        "is confirming the table parameters",
        "is recording the table metadata",
        "is identifying the required table fields",
        "is verifying the table description",
        "is matching the table to the request",
        "is finding the table card",
        "is loading the selected table structure",
        "is finding the required fields in the description",
        "is confirming the metadata is available",
        "is reviewing the table contents",
        "is matching the table fields to the task",
        "is preparing the table for SQL",
        "is finding the technical table details",
        "is confirming the table schema",
        "is preparing metadata for query construction",
    ),
    GetTableMetadataStage.NOT_FOUND: (
        "is searching for another matching table",
        "is reviewing the requested table name",
        "is trying an alternative table match",
        "is rephrasing the table lookup",
        "is checking additional table details",
        "is rechecking the requested table",
        "is searching for another relevant table",
        "is reviewing the table selection",
        "is checking the request against the catalog",
        "is refining the table criteria",
        "is checking the allowlisted table names",
        "is reviewing the available table catalog",
        "is checking the table name",
        "is refining the metadata lookup",
        "is searching other allowed data sources",
        "is checking alternative table cards",
        "is matching the table name against the catalog",
        "is searching available table descriptions",
        "is finding a safe allowlisted alternative",
        "is trying another metadata lookup",
    ),
}

_VALIDATE_SQL_MESSAGES: dict[ValidateSqlStage, tuple[str, ...]] = {
    ValidateSqlStage.START: (
        "is checking the SQL query",
        "is comparing the SQL query with the rules",
        "is analyzing the SQL query",
        "is checking the SQL query for correctness",
        "is reviewing the SQL query details",
        "is reading the SQL query",
        "is comparing the SQL query with its constraints",
        "is running SQL validation",
        "is evaluating the SQL query",
        "is checking the SQL query before execution",
        "is checking that the SQL query is read-only",
        "is comparing the SQL with allowlisted tables",
        "is checking the SQL execution route",
        "is reviewing SQL against source rules",
        "is checking SQL operations",
        "is identifying tables and conditions in the SQL query",
        "is checking SQL before issuing an execution token",
        "is comparing SQL with available data sources",
        "is checking the SQL dialect and tables",
        "is preparing SQL for safety validation",
    ),
    ValidateSqlStage.RETRY: (
        "is correcting the SQL and trying again",
        "is adjusting the SQL and retrying",
        "is reworking the SQL query for another attempt",
        "is refining the SQL query and checking it again",
        "is rebuilding the SQL query and retrying",
        "is correcting the SQL syntax and retrying",
        "is rechecking the SQL for another attempt",
        "is adapting the SQL query and trying again",
        "is updating the SQL query and repeating validation",
        "is adjusting the SQL and running validation again",
        "is rewriting SQL to follow access rules",
        "is finding a more precise SQL query",
        "is refining the SQL tables and conditions",
        "is rebuilding the query with clearer conditions",
        "is preparing a revised SQL query",
        "is selecting the correct route for the SQL query",
        "is fixing the SQL validation issue",
        "is refining SQL for validation",
        "is rechecking the constraints and updating SQL",
        "is preparing a safe SQL query",
    ),
    ValidateSqlStage.OK: (
        "is validating the SQL query",
        "is confirming the SQL query is correct",
        "is completing SQL validation",
        "is confirming the SQL query is valid",
        "is preparing the SQL query for execution",
        "is confirming the SQL query follows the rules",
        "is successfully validating the SQL query",
        "is approving the SQL query",
        "is confirming the SQL query is ready to run",
        "is finishing the SQL query checks",
        "is confirming the safe SQL execution route",
        "is validating SQL and preparing it for execution",
        "is confirming SQL uses allowlisted tables",
        "is authorizing the SQL query for execution",
        "is confirming the dialect and data source",
        "is recording the validated SQL query",
        "is preparing the SQL execution token",
        "is checking SQL for read-only access and allowed tables",
        "is confirming SQL follows the source rules",
        "is preparing to run the SQL query",
    ),
}

_EXECUTE_VALIDATED_SQL_MESSAGES: dict[ExecuteValidatedSqlStage, tuple[str, ...]] = {
    ExecuteValidatedSqlStage.START: (
        "is executing the validated SQL query",
        "is running the validated SQL query",
        "is sending the validated SQL query to the database",
        "is starting SQL query execution",
        "is beginning to execute the SQL query",
        "is moving to SQL query execution",
        "is executing SQL under the validated rules",
        "is applying the validated SQL query",
        "is running the SQL query after validation",
        "is processing the SQL query for execution",
        "is preparing validated SQL for execution",
        "is checking the SQL execution token",
        "is moving from validation to SQL execution",
        "is submitting the validated query for execution",
        "is checking authorization to run the SQL query",
        "is opening the SQL execution stage",
        "is preparing the route to the selected database",
        "is loading validated SQL from the request state",
        "is checking the SQL execution token lifetime",
        "is starting execution on the validated route",
    ),
    ExecuteValidatedSqlStage.PROBLEM: (
        "is refining the SQL query",
        "is reviewing the SQL execution issue",
        "is correcting the SQL query parameters",
        "is revisiting the SQL query",
        "is adapting the SQL query to the constraints",
        "is adjusting the SQL query before retrying",
        "is refining the SQL query wording",
        "is rebuilding the SQL query",
        "is adjusting the SQL query for another attempt",
        "is aligning the SQL query with execution requirements",
        "is checking the validated SQL query data",
        "is preparing a new SQL execution token",
        "is checking the current execution authorization",
        "is matching the SQL query to the current conversation",
        "is checking whether the SQL can run again",
        "is refreshing the SQL validation state",
        "is reviewing the validated SQL query state",
        "is identifying the SQL execution issue",
        "is checking constraints before retrying",
        "is preparing a new safe execution route",
    ),
    ExecuteValidatedSqlStage.DB_CONNECTING: (
        "is waiting for the database connection",
        "is checking the database connection",
        "is establishing a database connection",
        "is preparing the database connection",
        "is starting to connect to the database",
        "is moving to database connection",
        "is checking database access",
        "is checking database connection readiness",
        "is setting up the database connection",
        "is connecting to the database",
        "is checking the selected data source",
        "is finding the configured database query tool",
        "is preparing the SQL execution channel",
        "is matching the connection to the selected source",
        "is preparing database access",
        "is checking the database query configuration",
        "is waiting for the SQL query tool",
        "is checking database query readiness",
        "is checking the data connection route",
        "is preparing the connection for query execution",
    ),
    ExecuteValidatedSqlStage.DB_WAITING: (
        "is waiting for the database response",
        "is waiting for the database result",
        "is receiving the database response",
        "is processing the database response",
        "is waiting for the query result",
        "is reading data from the database",
        "is tracking query execution in the database",
        "is receiving the database result",
        "is checking the database response",
        "is collecting the query result from the database",
        "is waiting for the database to run the validated query",
        "is waiting while the database calculates the result",
        "is waiting for data from the selected source",
        "is keeping the database connection open for the result",
        "is waiting while the database prepares result rows",
        "is waiting for database-side query completion",
        "is receiving data from the validated SQL query",
        "is waiting for the data source to return the result",
        "is collecting the response while the query runs",
        "is waiting for the current database query to finish",
    ),
    ExecuteValidatedSqlStage.DB_ERROR: (
        "is correcting the SQL and trying again",
        "is handling the database error and preparing a retry",
        "is refining SQL after the database error",
        "is correcting the query after the database error",
        "is rechecking SQL against the database error",
        "is preparing execution after correcting the error",
        "is analyzing the database error and adjusting SQL",
        "is updating SQL for another execution attempt",
        "is preparing the corrected query",
        "is reducing the query error risk before retrying",
        "is reviewing the database error response",
        "is checking what prevented query execution",
        "is refining SQL using the database message",
        "is identifying the query execution failure",
        "is rechecking the query after the database rejection",
        "is preparing a correction for the execution error",
        "is analyzing the database-side query failure",
        "is handling the execution issue and preparing a retry",
        "is comparing SQL with the database error",
        "is finding a safe way to retry the query",
    ),
    ExecuteValidatedSqlStage.FINAL_ANALYSIS: (
        "is analyzing the query result",
        "is checking the final query data",
        "is matching the result to the request",
        "is preparing a summary of the query result",
        "is evaluating the returned data",
        "is refining the conclusion from the query result",
        "is building the final result analysis",
        "is summarizing the query result",
        "is checking the completeness of the query result",
        "is confirming the query result",
        "is preparing the final answer",
        "is reviewing the returned rows",
        "is turning the query result into a clear conclusion",
        "is checking that the data answers the question",
        "is summarizing the database result",
        "is comparing the result with the original task",
        "is preparing an explanation of the returned data",
        "is identifying the key points in the query result",
        "is checking the result table before answering",
        "is building the answer from the returned data",
        "is turning the query result into a final conclusion",
    ),
}

def show_progress_message(*, writer: Callable[[str], None], stage: str) -> None:
    """Emit a user-facing progress message via LangGraph stream writer."""
    text = stage.strip() if isinstance(stage, str) else ""
    writer(text or DEFAULT_PROGRESS_FALLBACK_MESSAGE)


def _pick_stage_message(
    stage_messages: Mapping[StageEnum, tuple[str, ...]], stage: StageEnum
) -> str:
    """Pick one stage phrase or fallback when stage mapping is missing/invalid."""
    options = stage_messages.get(stage)
    if not options:
        return DEFAULT_PROGRESS_FALLBACK_MESSAGE

    normalized = tuple(
        phrase.strip()
        for phrase in options
        if isinstance(phrase, str) and phrase.strip()
    )
    if not normalized:
        return DEFAULT_PROGRESS_FALLBACK_MESSAGE
    return random.choice(normalized)


def get_table_descriptions_message(stage: GetTableDescriptionsStage) -> str:
    """Return one random phrase for a table-description stage."""
    return _pick_stage_message(_GET_TABLE_DESCRIPTIONS_MESSAGES, stage)


def get_table_metadata_message(stage: GetTableMetadataStage) -> str:
    """Return one random phrase for a table-metadata stage."""
    return _pick_stage_message(_GET_TABLE_METADATA_MESSAGES, stage)


def validate_sql_message(stage: ValidateSqlStage) -> str:
    """Return one random phrase for a SQL-validation stage."""
    return _pick_stage_message(_VALIDATE_SQL_MESSAGES, stage)


def execute_validated_sql_message(stage: ExecuteValidatedSqlStage) -> str:
    """Return one random phrase for a validated-SQL execution stage."""
    return _pick_stage_message(_EXECUTE_VALIDATED_SQL_MESSAGES, stage)
