import json

import pytest

from ai import ai_tools_extra


def _call_switch_color_theme(theme: str = "") -> str:
    return getattr(ai_tools_extra.switch_color_theme, "func")(theme)


@pytest.fixture
def emitted_events(monkeypatch) -> list[object]:
    events: list[object] = []

    def writer(event: object) -> None:
        events.append(event)

    monkeypatch.setattr(ai_tools_extra, "get_stream_writer", lambda: writer)
    return events


def test_switch_color_theme_picks_random_theme_when_input_is_empty(
    monkeypatch, emitted_events: list[object]
) -> None:
    monkeypatch.setattr(ai_tools_extra.random, "choice", lambda _themes: "cupcake")

    result = _call_switch_color_theme("")
    payload = json.loads(result)

    assert payload["status"] == "ok"
    assert payload["theme"] == "cupcake"
    assert payload["source"] == "random"
    assert emitted_events == [{"kind": "theme", "theme": "cupcake"}]


@pytest.mark.parametrize(
    "theme",
    [
        "switch theme",
        "switch color theme",
        "switch color scheme",
        "random",
        "random theme",
        "смени тему",
        "переключи тему",
        "смени цветовую тему",
        "смени цветовую схему",
        "поставь рандомую тему",
        "поставь случайную тему",
        "поставь случайную цветовую тему",
        "поставь случайную цветовую схему",
    ],
)
def test_switch_color_theme_recognizes_english_and_russian_random_aliases(
    monkeypatch,
    emitted_events: list[object],
    theme: str,
) -> None:
    monkeypatch.setattr(ai_tools_extra.random, "choice", lambda _themes: "nord")

    result = _call_switch_color_theme(theme)

    assert json.loads(result) == {"status": "ok", "theme": "nord", "source": "random"}
    assert emitted_events == [{"kind": "theme", "theme": "nord"}]


def test_switch_color_theme_resolves_theme_by_1_based_index(
    emitted_events: list[object],
) -> None:
    result = _call_switch_color_theme("3")
    payload = json.loads(result)

    assert payload["status"] == "ok"
    assert payload["theme"] == ai_tools_extra.DAISYUI_THEMES[2]
    assert payload["source"] == "index"
    assert emitted_events == [
        {"kind": "theme", "theme": ai_tools_extra.DAISYUI_THEMES[2]}
    ]


@pytest.mark.parametrize("theme", ["0", str(len(ai_tools_extra.DAISYUI_THEMES) + 1)])
def test_switch_color_theme_rejects_out_of_range_index(
    theme: str,
    emitted_events: list[object],
) -> None:
    result = _call_switch_color_theme(theme)
    payload = json.loads(result)

    assert payload["status"] == "reject"
    assert payload["error_code"] == "THEME_INDEX_OUT_OF_RANGE"
    assert emitted_events == []


def test_switch_color_theme_resolves_theme_by_name_case_insensitive(
    emitted_events: list[object],
) -> None:
    result = _call_switch_color_theme("CuPcAkE")
    payload = json.loads(result)

    assert payload["status"] == "ok"
    assert payload["theme"] == "cupcake"
    assert payload["source"] == "name"
    assert emitted_events == [{"kind": "theme", "theme": "cupcake"}]


def test_switch_color_theme_rejects_unknown_theme_name(
    emitted_events: list[object],
) -> None:
    result = _call_switch_color_theme("does-not-exist")
    payload = json.loads(result)

    assert payload["status"] == "reject"
    assert payload["error_code"] == "UNKNOWN_THEME"
    assert emitted_events == []
