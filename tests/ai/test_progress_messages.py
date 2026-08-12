"""Progress catalog tests (copy-agnostic).

Flow:
- Patch `random.choice` and assert each selector reads only the bucket for the requested stage.
- Return a sentinel value to prove selector output comes from that stage bucket.
- Validate catalog hygiene (non-empty, trimmed strings) without pinning exact wording.
"""

import pytest

from ai.ai_utils import progress_messages as pm


def test_show_progress_message_forwards_text_to_writer() -> None:
    """`show_progress_message` should pass text through without modification."""
    captured: list[str] = []

    def _writer(text: str) -> None:
        captured.append(text)

    pm.show_progress_message(writer=_writer, stage="stage-text")

    assert captured == ["stage-text"]


def _assert_stage_selector(monkeypatch, selector, stage, expected_bucket) -> None:
    """Assert selector reads exactly one bucket: the one for the requested stage."""
    selected_options: list[tuple[str, ...]] = []
    sentinel = "selected-message"

    def _fake_choice(options: tuple[str, ...]) -> str:
        selected_options.append(options)
        return sentinel

    monkeypatch.setattr(pm.random, "choice", _fake_choice)

    result = selector(stage)

    assert result == sentinel
    assert selected_options == [expected_bucket]


@pytest.mark.parametrize(
    ("stage", "expected_bucket"),
    [
        (
            pm.GetTableDescriptionsStage.START,
            pm._GET_TABLE_DESCRIPTIONS_MESSAGES[pm.GetTableDescriptionsStage.START],
        ),
    ],
)
def test_get_table_descriptions_message_uses_selected_stage_bucket(
    monkeypatch,
    stage,
    expected_bucket,
) -> None:
    """Table-description selector must choose from the requested stage bucket only."""
    _assert_stage_selector(
        monkeypatch,
        pm.get_table_descriptions_message,
        stage,
        expected_bucket,
    )


@pytest.mark.parametrize(
    ("stage", "expected_bucket"),
    [
        (
            pm.GetTableMetadataStage.START,
            pm._GET_TABLE_METADATA_MESSAGES[pm.GetTableMetadataStage.START],
        ),
        (
            pm.GetTableMetadataStage.FOUND,
            pm._GET_TABLE_METADATA_MESSAGES[pm.GetTableMetadataStage.FOUND],
        ),
        (
            pm.GetTableMetadataStage.NOT_FOUND,
            pm._GET_TABLE_METADATA_MESSAGES[pm.GetTableMetadataStage.NOT_FOUND],
        ),
    ],
)
def test_get_table_metadata_message_uses_selected_stage_bucket(
    monkeypatch,
    stage,
    expected_bucket,
) -> None:
    """Metadata selector must not cross-pick phrases from other metadata stages."""
    _assert_stage_selector(
        monkeypatch, pm.get_table_metadata_message, stage, expected_bucket
    )


@pytest.mark.parametrize(
    ("stage", "expected_bucket"),
    [
        (
            pm.ValidateSqlStage.START,
            pm._VALIDATE_SQL_MESSAGES[pm.ValidateSqlStage.START],
        ),
        (
            pm.ValidateSqlStage.RETRY,
            pm._VALIDATE_SQL_MESSAGES[pm.ValidateSqlStage.RETRY],
        ),
        (pm.ValidateSqlStage.OK, pm._VALIDATE_SQL_MESSAGES[pm.ValidateSqlStage.OK]),
    ],
)
def test_validate_sql_message_uses_selected_stage_bucket(
    monkeypatch,
    stage,
    expected_bucket,
) -> None:
    """Validate-SQL selector must use the exact phrase pool for the requested stage."""
    _assert_stage_selector(monkeypatch, pm.validate_sql_message, stage, expected_bucket)


@pytest.mark.parametrize(
    ("stage", "expected_bucket"),
    [
        (
            pm.ExecuteValidatedSqlStage.START,
            pm._EXECUTE_VALIDATED_SQL_MESSAGES[pm.ExecuteValidatedSqlStage.START],
        ),
        (
            pm.ExecuteValidatedSqlStage.PROBLEM,
            pm._EXECUTE_VALIDATED_SQL_MESSAGES[pm.ExecuteValidatedSqlStage.PROBLEM],
        ),
        (
            pm.ExecuteValidatedSqlStage.DB_CONNECTING,
            pm._EXECUTE_VALIDATED_SQL_MESSAGES[
                pm.ExecuteValidatedSqlStage.DB_CONNECTING
            ],
        ),
        (
            pm.ExecuteValidatedSqlStage.DB_WAITING,
            pm._EXECUTE_VALIDATED_SQL_MESSAGES[pm.ExecuteValidatedSqlStage.DB_WAITING],
        ),
        (
            pm.ExecuteValidatedSqlStage.DB_ERROR,
            pm._EXECUTE_VALIDATED_SQL_MESSAGES[pm.ExecuteValidatedSqlStage.DB_ERROR],
        ),
        (
            pm.ExecuteValidatedSqlStage.FINAL_ANALYSIS,
            pm._EXECUTE_VALIDATED_SQL_MESSAGES[
                pm.ExecuteValidatedSqlStage.FINAL_ANALYSIS
            ],
        ),
    ],
)
def test_execute_validated_sql_message_uses_selected_stage_bucket(
    monkeypatch,
    stage,
    expected_bucket,
) -> None:
    """Execute-SQL selector must use only the pool for the requested execution stage."""
    _assert_stage_selector(
        monkeypatch,
        pm.execute_validated_sql_message,
        stage,
        expected_bucket,
    )


@pytest.mark.parametrize(
    "catalog",
    [
        pm._GET_TABLE_DESCRIPTIONS_MESSAGES,
        pm._GET_TABLE_METADATA_MESSAGES,
        pm._VALIDATE_SQL_MESSAGES,
        pm._EXECUTE_VALIDATED_SQL_MESSAGES,
    ],
)
def test_message_catalog_contains_only_non_empty_strings(catalog) -> None:
    """All message pools must contain clean, non-empty display strings."""
    for bucket in catalog.values():
        assert bucket
        for phrase in bucket:
            assert isinstance(phrase, str)
            assert phrase.strip() == phrase
            assert phrase


@pytest.mark.parametrize(
    "catalog",
    [
        {},
        {pm.ValidateSqlStage.START: ()},
        {pm.ValidateSqlStage.START: ("", "   ")},
    ],
)
def test_validate_sql_message_falls_back_for_invalid_catalog(
    monkeypatch,
    catalog,
) -> None:
    """Missing, empty, and blank-only stage buckets should use the fallback."""
    monkeypatch.setattr(pm, "_VALIDATE_SQL_MESSAGES", catalog)

    result = pm.validate_sql_message(pm.ValidateSqlStage.START)

    assert result == pm.DEFAULT_PROGRESS_FALLBACK_MESSAGE


def test_show_progress_message_falls_back_for_blank_stage() -> None:
    """Blank emitted text should be normalized to default fallback."""
    captured: list[str] = []

    def _writer(text: str) -> None:
        captured.append(text)

    pm.show_progress_message(writer=_writer, stage="   ")

    assert captured == [pm.DEFAULT_PROGRESS_FALLBACK_MESSAGE]
