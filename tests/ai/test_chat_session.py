from __future__ import annotations

from pathlib import Path

from ai.ai_utils import chat_session


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _read_ai_main_template() -> str:
    template_path = PROJECT_ROOT / "ai" / "templates" / "ai" / "ai_main.html"
    return template_path.read_text(encoding="utf-8")


def test_normalize_chat_session_key_keeps_valid_key() -> None:
    assert chat_session.normalize_chat_session_key("chat_ABC-1234") == "chat_ABC-1234"


def test_normalize_chat_session_key_replaces_blank_or_invalid_values(
    monkeypatch,
) -> None:
    monkeypatch.setattr(chat_session, "new_chat_session_key", lambda: "chat-generated")

    assert chat_session.normalize_chat_session_key("") == "chat-generated"
    assert chat_session.normalize_chat_session_key("   ") == "chat-generated"
    assert (
        chat_session.normalize_chat_session_key("bad key with spaces")
        == "chat-generated"
    )


def test_build_thread_id_is_stable_for_same_user_session_and_client_key() -> None:
    thread_id = chat_session.build_thread_id(
        user_id=7, session_key="session-1", client_key="chat-1"
    )

    assert thread_id == chat_session.build_thread_id(
        user_id=7, session_key="session-1", client_key="chat-1"
    )


def test_build_thread_id_separates_users_sessions_and_client_keys() -> None:
    base = chat_session.build_thread_id(
        user_id=7, session_key="session-1", client_key="chat-1"
    )

    assert base != chat_session.build_thread_id(
        user_id=8, session_key="session-1", client_key="chat-1"
    )
    assert base != chat_session.build_thread_id(
        user_id=7, session_key="session-2", client_key="chat-1"
    )
    assert base != chat_session.build_thread_id(
        user_id=7, session_key="session-1", client_key="chat-2"
    )


def test_ai_main_template_persists_chat_session_key_in_session_storage() -> None:
    content = _read_ai_main_template()

    assert "data-signals:chat-session-key__ifmissing=" in content
    assert "sessionStorage.getItem('chatSessionKey')" in content
    assert "window.aiChatSessionUi.newChatSessionKey()" in content
    assert (
        "if ($chatSessionKey) sessionStorage.setItem('chatSessionKey', $chatSessionKey);"
        in content
    )
    assert "активной вкладки браузера" in content


def test_ai_main_template_persists_transcript_and_usage_per_chat_session_key() -> None:
    content = _read_ai_main_template()

    assert (
        "const getMessagesStorageKey = (chatSessionKey) => `chat-ui:${chatSessionKey}:messages`;"
        in content
    )
    assert (
        "const getUsageStorageKey = (chatSessionKey) => `chat-ui:${chatSessionKey}:usage`;"
        in content
    )
    assert "const loadUsageState = (chatSessionKey) => {" in content
    assert "window.aiChatSessionUi.loadUsageState($chatSessionKey)" in content
    assert "return {};" in content
    assert (
        "sessionStorage.setItem(`chat-ui:${$chatSessionKey}:usage`, JSON.stringify({"
        in content
    )
    assert (
        'persistTranscript(sessionStorage.getItem(CHAT_SESSION_STORAGE_KEY) || "")'
        in content
    )


def test_ai_main_template_rehydrates_transcript_without_duplicate_dom_messages() -> (
    None
):
    content = _read_ai_main_template()

    assert 'template id="chat-default-content-template"' in content
    assert (
        'root.innerHTML = `${getDefaultChatMarkup()}${persistedMessages.map(renderPersistedMessage).join("")}`;'
        in content
    )
    assert "transcriptPersistencePaused = true;" in content


def test_ai_main_template_loads_markdown_dependencies_with_guarded_setup() -> None:
    content = _read_ai_main_template()

    assert "marked.min.js" in content
    assert "purify.min.js" in content
    assert "globalThis.marked" in content
    assert "globalThis.DOMPurify" in content
    assert "setOptions" in content
    assert "markdownParser.parse" in content
    assert "sanitizer.sanitize" in content
    assert "gfm: true" in content
    assert "breaks: true" in content
    assert "return false;" in content or "return null;" in content


def test_ai_main_template_renders_only_assistant_markdown_with_sanitizer() -> None:
    content = _read_ai_main_template()

    assert '[data-chat-role="assistant"]:not([data-chat-blocks]) .chat-bubble' in content
    assert '[data-chat-role="user"] .chat-bubble' not in content
    assert "markdownParser.parse" in content
    assert "sanitize" in content
    assert "markdown-content" in content
    assert "innerHTML" in content


def test_ai_main_template_debounces_markdown_rendering_before_persisting() -> None:
    content = _read_ai_main_template()

    assert "MARKDOWN_RENDER_DEBOUNCE_MS" in content
    assert "window.setTimeout" in content
    assert "renderAssistantMarkdown();" in content
    assert content.index("renderAssistantMarkdown();") < content.index(
        'persistTranscript(sessionStorage.getItem(CHAT_SESSION_STORAGE_KEY) || "")'
    )


def test_ai_main_template_preserves_raw_markdown_in_existing_session_transcript() -> None:
    content = _read_ai_main_template()

    assert "chatRawText" in content
    assert "role === \"assistant\"" in content
    assert "getMessagesStorageKey" in content
    assert "JSON.stringify(messages)" in content
    assert "markdown:${chatSessionKey}" not in content
    assert "chat-ui:${chatSessionKey}:markdown" not in content


def test_ai_main_template_persists_structured_assistant_blocks() -> None:
    content = _read_ai_main_template()

    assert "node.dataset.chatBlocks" in content
    assert "JSON.parse(node.dataset.chatBlocks)" in content
    assert "[{ role, blocks }]" in content
    assert "wrapper.dataset.chatBlocks = stringifyBlocks(messageBlocks);" in content
    assert "appendStructuredBlocks(bubble, messageBlocks);" in content


def test_ai_main_template_restores_commentary_markdown_and_data_tables() -> None:
    content = _read_ai_main_template()

    assert "const createCommentaryBlock = (block) => {" in content
    assert "const createDataTableBlock = (block) => {" in content
    assert 'element.dataset.blockType = "commentary";' in content
    assert 'element.dataset.blockType = "data_table";' in content
    assert 'root.querySelectorAll(\'[data-block-type="commentary"]\')' in content


def test_ai_main_template_has_data_table_copy_controls() -> None:
    content = _read_ai_main_template()

    assert "const createTableCopyButton = () => createCopyButton" in content
    assert "const createCopyButton = ({ className, datasetKey, defaultLabel }) => {" in content
    assert 'button.dataset[datasetKey] = "true";' in content
    assert 'datasetKey: "tableCopyButton"' in content
    assert "const copyDataTableToClipboard = async (button) => {" in content
    assert "const fallbackCopyText = (text) => {" in content
    assert "window.isSecureContext && navigator.clipboard" in content
    assert "navigator.clipboard.writeText(text);" in content
    assert "await writeClipboardText(tsv)" in content
    assert 'document.execCommand("copy")' in content
    assert "normalizeTableCopyCell" in content
    assert "data-table-copy-button" in content


def test_ai_main_template_restores_data_table_details_and_sql_copy() -> None:
    content = _read_ai_main_template()

    assert "const getTableDetails = (block, renderedRowCount, rowCount) => {" in content
    assert "RESERVED_ANSWER_DETAIL_NOTE_LABELS" in content
    assert "const createDataTableDetailsBlock = (details) => {" in content
    assert "Показать SQL" in content
    assert "data-sql-copy-button" in content
    assert "const copySqlToClipboard = async (button) => {" in content
    assert "copySqlToClipboard(sqlButton);" in content
    assert "code.textContent = details.rawSql;" in content


def test_ai_main_template_renders_restored_assistant_markdown_after_rebuild() -> None:
    content = _read_ai_main_template()

    restore_index = content.index("const restoreTranscript =")
    assert "renderPersistedMessage" in content[restore_index:]
    assert "renderAssistantMarkdown();" in content[restore_index:]
    assert "transcriptPersistencePaused = true;" in content[restore_index:]
    assert "transcriptPersistencePaused = false;" in content[restore_index:]


def test_ai_main_template_has_scoped_compact_markdown_styles() -> None:
    content = _read_ai_main_template()

    assert "#chat-app .markdown-content {" in content
    assert "overflow-x: auto;" in content
    assert "#chat-app .markdown-content table {" in content


def test_ai_main_template_has_new_chat_reset_action() -> None:
    navbar_template_path = (
        Path(__file__).resolve().parents[2]
        / "core"
        / "templates"
        / "core"
        / "navbar.html"
    )
    ai_content = _read_ai_main_template()
    navbar_content = navbar_template_path.read_text(encoding="utf-8")

    assert "dispatchEvent(new CustomEvent('ai-new-chat-requested'" not in ai_content
    assert "Новый чат" in navbar_content
    assert (
        "chatApp.dispatchEvent(new CustomEvent('ai-new-chat-requested', { bubbles: true }));"
        in navbar_content
    )
    assert (
        "sessionStorage.removeItem(getMessagesStorageKey(previousChatSessionKey));"
        in ai_content
    )
    assert (
        "sessionStorage.removeItem(getUsageStorageKey(previousChatSessionKey));"
        in ai_content
    )


def test_navbar_new_chat_button_is_only_rendered_for_ai_main() -> None:
    template_path = (
        Path(__file__).resolve().parents[2]
        / "core"
        / "templates"
        / "core"
        / "navbar.html"
    )
    content = template_path.read_text(encoding="utf-8")

    assert "request.resolver_match.url_name == 'ai_main'" in content
    assert "document.getElementById('chat-app')" in content
    assert (
        "chatApp.dispatchEvent(new CustomEvent('ai-new-chat-requested', { bubbles: true }));"
        in content
    )


def test_ai_main_template_resets_visible_usage_immediately_on_navbar_new_chat_event() -> (
    None
):
    content = _read_ai_main_template()

    assert "data-on:ai-new-chat-requested=" in content
    assert "$sessionTotalTokens = 0;" in content
    assert "$sessionTokenBadgeText = '';" in content


def test_base_template_keeps_mobile_sidebar_width_compact() -> None:
    template_path = (
        Path(__file__).resolve().parents[2]
        / "core"
        / "templates"
        / "core"
        / "base.html"
    )
    content = template_path.read_text(encoding="utf-8")

    assert "w-[min(14.5rem,calc(100vw-4rem))] bg-base-100 shadow-2xl" in content
    assert "max-w-[calc(100vw-4rem)]" not in content


def test_base_template_shows_current_ui_version() -> None:
    template_path = (
        Path(__file__).resolve().parents[2]
        / "core"
        / "templates"
        / "core"
        / "base.html"
    )
    content = template_path.read_text(encoding="utf-8")

    assert "v0.9.35" in content


def test_navbar_template_uses_narrower_sidebar_widths() -> None:
    template_path = (
        Path(__file__).resolve().parents[2]
        / "core"
        / "templates"
        / "core"
        / "navbar.html"
    )
    content = template_path.read_text(encoding="utf-8")

    assert "w-[min(14.5rem,calc(100vw-4rem))]" in content
    assert "w-[14.5rem]" in content
