from __future__ import annotations

import re

INTENT_SMALLTALK_META = "smalltalk_meta"
INTENT_THEME_CHANGE = "theme_change"
INTENT_SQL_AGENT = "sql_agent"

_SMALLTALK_PHRASES = {
    "hi",
    "sup",
    "hey",
    "hello",
    "hello there",
    "what can you do",
    "добрый день",
    "здарова",
    "здрасти",
    "здрасьте",
    "здравствуйте",
    "как дела",
    "привет",
    "что ты можешь",
    "что ты умеешь",
    "что ты умеешь делать",
    "что умеешь делать",
}

_THEME_PHRASES = {
    "измени тему",
    "измени цвет",
    "переключи тему",
    "поменяй тему",
    "поменяй цвет",
    "поставь другой цвет",
    "поставь случайную тему",
    "сделай тему потемнее",
    "сделай тему посветлее",
    "сделай тему темнее",
    "смени тему",
    "смени цвет",
    "смени цвет темы",
}

_BUSINESS_INDICATORS = {
    "customer",
    "customers",
    "data",
    "inventory",
    "order",
    "orders",
    "price",
    "prices",
    "product",
    "products",
    "revenue",
    "sales",
    "аналит",
    "выручк",
    "заказ",
    "заказы",
    "клиент",
    "клиента",
    "клиентов",
    "минимал",
    "остатки",
    "покажи",
    "продаж",
    "расследуй",
    "исследуй",
    "сравни",
    "товар",
    "товаров",
    "топ",
    "цена",
    "цены",
}


def classify_intent(message: str) -> str:
    normalized_message = _normalize_message(message)
    if _has_business_indicator(normalized_message):
        return INTENT_SQL_AGENT
    if _is_theme_change_request(normalized_message):
        return INTENT_THEME_CHANGE
    if _matches_phrase(normalized_message, _SMALLTALK_PHRASES):
        return INTENT_SMALLTALK_META
    return INTENT_SQL_AGENT


def _normalize_message(message: str) -> str:
    lowered_message = message.casefold()
    cleaned_message = re.sub(r"[^\w\s]", " ", lowered_message)
    collapsed_message = re.sub(r"\s+", " ", cleaned_message)
    return collapsed_message.strip()


def _has_business_indicator(message: str) -> bool:
    """Return True when the turn looks like a data request.

    The explicit follow-up pattern catches short context-dependent turns such as:
    - "а по клиентам?"
    - "по заказам"

    The router does not read conversation history, so these fragments need to stay on
    the SQL path instead of being treated as lightweight chat.
    """
    if re.search(r"(?:^|\s)(?:а\s+)?по\s+\w+", message):
        return True
    return any(_matches_word_or_stem(message, indicator) for indicator in _BUSINESS_INDICATORS)


def _is_theme_change_request(message: str) -> bool:
    """Return True when the turn asks to change the UI theme.

    The regex branches cover phrasing that is hard to represent as fixed phrases:
    - "смени пожалуйста тему" allows one polite/filler word between action and "тему"
    - "switch the theme to nord" and "change the theme" match documented English examples
    - "make the theme lighter" and "make the theme darker" match English brightness requests
    """
    if _matches_phrase(message, _THEME_PHRASES):
        return True
    if re.search(r"(?:^|\s)(смени|поменяй|измени|переключи)\s+\w+\s+тему(?:\s|$)", message):
        return True
    if re.search(r"(?:^|\s)(switch|change)\s+the\s+theme(?:\s|$)", message):
        return True
    if re.search(r"(?:^|\s)make\s+the\s+theme\s+(lighter|darker)(?:\s|$)", message):
        return True
    return "смени тему на " in message


def _matches_phrase(message: str, phrases: set[str]) -> bool:
    return any(_matches_exact_phrase(message, phrase) for phrase in phrases)


def _matches_exact_phrase(message: str, phrase: str) -> bool:
    if " " in phrase:
        return phrase in message
    return bool(re.search(rf"(?:^|\s){re.escape(phrase)}(?:\s|$)", message))


def _matches_word_or_stem(message: str, indicator: str) -> bool:
    return bool(re.search(rf"(?:^|\s){re.escape(indicator)}\w*(?:\s|$)", message))


__all__ = [
    "INTENT_SMALLTALK_META",
    "INTENT_THEME_CHANGE",
    "INTENT_SQL_AGENT",
    "classify_intent",
]
