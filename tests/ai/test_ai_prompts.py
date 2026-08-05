import importlib

from ai import ai_prompts
import ai.ai_utils.sql_tools as sql_tools


def test_system_prompt_uses_source_matched_dialect_without_generic_schema_tools() -> None:
    prompt = ai_prompts.SYSTEM_PROMPT_RU_SARCASTIC

    assert "exactly one source and its dialect" in prompt
    assert "fully qualified table names" in prompt
    assert "sql_db_list_tables" not in prompt
    assert "sql_db_schema" not in prompt
    assert not hasattr(ai_prompts, "SYSTEM_PROMPT_RU_SARCASTIC_CLI_DEMO")
    assert not hasattr(ai_prompts, "SYSTEM_PROMPT_CHECK_QUERY")


def test_ai_prompts_imports_with_mixed_allowlisted_dialects(monkeypatch) -> None:
    monkeypatch.setattr(
        sql_tools,
        "get_configured_sql_dialect",
        lambda: (_ for _ in ()).throw(
            AssertionError("should not be called by ai_prompts")
        ),
    )

    reloaded = importlib.reload(ai_prompts)

    assert "exactly one source and its dialect" in reloaded.SYSTEM_PROMPT_RU_SARCASTIC


def test_prompts_do_not_add_generic_brevity_instructions() -> None:
    prompts = [
        ai_prompts.SYSTEM_PROMPT_RU_SARCASTIC,
        ai_prompts.SYSTEM_PROMPT_SMALLTALK_META,
        ai_prompts.SYSTEM_PROMPT_THEME_CHANGE,
    ]

    for prompt in prompts:
        lowered_prompt = prompt.lower()
        assert "concise" not in lowered_prompt
        assert "keep the answer short" not in lowered_prompt
        assert "be brief" not in lowered_prompt


def test_smalltalk_and_theme_prompts_keep_their_product_scope() -> None:
    assert "Reply in Russian" in ai_prompts.SYSTEM_PROMPT_SMALLTALK_META
    assert "SQL questions about sales" in ai_prompts.SYSTEM_PROMPT_SMALLTALK_META
    assert "Do not answer substantial unrelated topics" in ai_prompts.SYSTEM_PROMPT_SMALLTALK_META
    assert "Reply in Russian" in ai_prompts.SYSTEM_PROMPT_THEME_CHANGE
    assert "theme-switching tool" in ai_prompts.SYSTEM_PROMPT_THEME_CHANGE


def test_sql_prompt_preserves_clarification_and_business_defaults() -> None:
    prompt = ai_prompts.SYSTEM_PROMPT_RU_SARCASTIC

    assert "call `ask_user`" in prompt
    assert "materially change the source, query, or result meaning" in prompt
    assert "reasonable business default" in prompt
    assert "recent" in prompt
    assert "актуальные" in prompt
    assert "current calendar year" in prompt
    assert "one neutral Russian question" in " ".join(prompt.split())


def test_sql_prompt_preserves_guarded_execution_workflow() -> None:
    prompt = ai_prompts.SYSTEM_PROMPT_RU_SARCASTIC

    assert "Call `get_table_descriptions`" in prompt
    assert "call `get_table_metadata` for every selected" in prompt
    assert "including required date filters" in prompt
    assert "Call `validate_sql(query)`" in prompt
    assert "execute only with `execute_validated_sql(validated_id)`" in prompt
    assert "Never execute raw SQL" in prompt
    assert "Stop after five" in prompt


def test_sql_prompt_preserves_domain_boundary() -> None:
    prompt = ai_prompts.SYSTEM_PROMPT_RU_SARCASTIC

    assert "For substantial unrelated requests" in prompt
    assert "business data" in prompt
    assert "Greetings and assistant-meta questions are allowed" in prompt


def test_sql_prompt_delegates_final_layout_details_to_tool_schema() -> None:
    prompt = ai_prompts.SYSTEM_PROMPT_RU_SARCASTIC

    assert "submit_model_response_layout(layout)" in prompt
    assert "exactly once" in prompt
    assert "SQL rows are backend-owned" in prompt
    assert "do not reproduce them in commentary or Markdown tables" in prompt
    assert "Use the final-layout tool schema" in prompt
    assert "`commentary` blocks" not in prompt
    assert "`data_table_placeholder` blocks" not in prompt
