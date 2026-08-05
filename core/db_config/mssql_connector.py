"""MS SQL SQLDatabase connector helpers."""

from __future__ import annotations

import os

from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine
from sqlalchemy.engine import URL

from core.db_config.clickhouse_connector import read_required_env
from core.db_config.source_registry import SQLSourceConfig


def create_mssql_sql_database(
    source: SQLSourceConfig, include_tables: list[str]
) -> SQLDatabase:
    """Create SQLDatabase for an MS SQL source and table subset."""
    host = read_required_env(source, "host")
    port_value = read_required_env(source, "port")
    user = read_required_env(source, "user")
    password = read_required_env(source, "password")
    database = read_required_env(source, "database")
    odbc_driver = os.getenv("MSSQL_ODBC_DRIVER", "ODBC Driver 18 for SQL Server")

    url = URL.create(
        drivername="mssql+pyodbc",
        username=user,
        password=password,
        host=host,
        port=int(port_value),
        database=database,
        query={"driver": odbc_driver, "TrustServerCertificate": "yes"},
    )
    engine = create_engine(url)

    return SQLDatabase(
        engine,
        include_tables=include_tables,
        sample_rows_in_table_info=0,
    )
