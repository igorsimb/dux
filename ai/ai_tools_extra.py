import json
import random

from langchain.tools import tool
from langgraph.config import get_stream_writer

DAISYUI_THEMES = (
    "light",
    "dark",
    "cupcake",
    "bumblebee",
    "emerald",
    "corporate",
    "synthwave",
    "retro",
    "cyberpunk",
    "valentine",
    "halloween",
    "garden",
    "forest",
    "aqua",
    "lofi",
    "pastel",
    "fantasy",
    "wireframe",
    "black",
    "luxury",
    "dracula",
    "cmyk",
    "autumn",
    "business",
    "acid",
    "lemonade",
    "night",
    "coffee",
    "winter",
    "dim",
    "nord",
    "sunset",
    "caramellatte",
    "abyss",
    "silk",
)

_RANDOM_THEME_HINTS = {
    "",
    "switch theme",
    "switch color theme",
    "switch color scheme",
    "random",
    "random theme",
    # Russian aliases intentionally preserve the existing bilingual input contract.
    "смени тему",
    "переключи тему",
    "смени цветовую тему",
    "смени цветовую схему",
    "поставь рандомую тему",
    "поставь случайную тему",
    "поставь случайную цветовую тему",
    "поставь случайную цветовую схему",
}


@tool
def switch_color_theme(theme: str = "") -> str:
    """Switch the UI theme by DaisyUI name or 1-based index; use an empty value for a random theme."""
    normalized = (theme or "").strip().lower()
    resolved_theme: str
    source: str

    if normalized in _RANDOM_THEME_HINTS:
        resolved_theme = random.choice(DAISYUI_THEMES)
        source = "random"
    elif normalized.isdigit():
        one_based_index = int(normalized)
        if one_based_index < 1 or one_based_index > len(DAISYUI_THEMES):
            return json.dumps(
                {
                    "status": "reject",
                    "error_code": "THEME_INDEX_OUT_OF_RANGE",
                    "reason": f"Theme index must be between 1 and {len(DAISYUI_THEMES)}.",
                },
                ensure_ascii=False,
            )
        resolved_theme = DAISYUI_THEMES[one_based_index - 1]
        source = "index"
    else:
        if normalized not in DAISYUI_THEMES:
            return json.dumps(
                {
                    "status": "reject",
                    "error_code": "UNKNOWN_THEME",
                    "reason": "Unknown DaisyUI theme name.",
                },
                ensure_ascii=False,
            )
        resolved_theme = normalized
        source = "name"

    writer = get_stream_writer()
    writer({"kind": "theme", "theme": resolved_theme})
    return json.dumps(
        {"status": "ok", "theme": resolved_theme, "source": source},
        ensure_ascii=False,
    )
