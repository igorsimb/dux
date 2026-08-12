import pytest

from ai.ai_utils.intent_router import (
    INTENT_SQL_AGENT,
    INTENT_SMALLTALK_META,
    INTENT_THEME_CHANGE,
    classify_intent,
)


@pytest.mark.parametrize(
    "message",
    [
        "здарова",
        "Привет",
        "добрый день",
        "hello there",
        "  ПРИВЕТ!!!  ",
        "здрасьте)",
        "как дела?",
    ],
)
def test_classify_intent_routes_recognized_english_and_russian_greetings_to_smalltalk_meta(
    message: str,
) -> None:
    assert classify_intent(message) == INTENT_SMALLTALK_META


@pytest.mark.parametrize(
    "message",
    [
        "что ты умеешь?",
        "что ты можешь",
        "расскажи, что ты умеешь делать",
        "what can you do?",
    ],
)
def test_classify_intent_routes_recognized_english_and_russian_meta_requests_to_smalltalk_meta(
    message: str,
) -> None:
    assert classify_intent(message) == INTENT_SMALLTALK_META


@pytest.mark.parametrize(
    "message",
    [
        "переключи тему",
        "смени тему на nord",
        "сделай тему темнее",
        "поставь случайную тему",
        "смени цвет",
        "поставь другой цвет",
    ],
)
def test_classify_intent_routes_recognized_russian_theme_requests_to_theme_change(message: str) -> None:
    assert classify_intent(message) == INTENT_THEME_CHANGE


def test_classify_intent_routes_mixed_greeting_and_data_request_to_sql_agent() -> None:
    assert (
        classify_intent("привет, покажи топ 10 клиентов по выручке")
        == INTENT_SQL_AGENT
    )


@pytest.mark.parametrize(
    "message",
    [
        "у какого заказа минимальная сумма",
        "сравни клиентов по выручке за 14 дней",
        "покажи топ 10 клиентов по числу заказов",
        "остатки по товару 12345",
        "продажи по категории электроника",
        "какие заказы есть по региону север",
    ],
)
def test_classify_intent_routes_recognized_russian_business_requests_to_sql_agent(message: str) -> None:
    assert classify_intent(message) == INTENT_SQL_AGENT


@pytest.mark.parametrize(
    "message",
    [
        "",
        "   ",
        "go ahead",
        "help",
        "yes",
        "okay",
        "sure",
        "hmm",
    ],
)
def test_classify_intent_falls_back_to_sql_agent_for_ambiguous_requests(message: str) -> None:
    assert classify_intent(message) == INTENT_SQL_AGENT


def test_classify_intent_prefers_business_indicators_over_greeting_indicators() -> None:
    assert classify_intent("здарова, есть остатки по товару 12345?") == INTENT_SQL_AGENT


@pytest.mark.parametrize(
    "message",
    [
        "привет, покажи топ 10 клиентов по выручке",
        "добрый день, сравни клиентов по числу заказов",
        "как дела, есть остатки по товару 12345?",
        "hello, покажи продажи по категориям",
    ],
)
def test_classify_intent_prefers_business_indicators_over_smalltalk_in_mixed_requests(message: str) -> None:
    assert classify_intent(message) == INTENT_SQL_AGENT


def test_classify_intent_prefers_business_indicators_over_theme_words_in_analytical_requests() -> (
    None
):
    assert classify_intent("сравни цветовые группы товаров по категориям") == INTENT_SQL_AGENT
    assert classify_intent("покажи товары синего цвета по категориям") == INTENT_SQL_AGENT
    assert classify_intent("какой цвет чаще встречается у товаров") == INTENT_SQL_AGENT


@pytest.mark.parametrize(
    "message",
    [
        "а по клиентам?",
        "а за январь?",
        "клиенты?",
        "заказы",
        "по товару 12345?",
        "только остатки",
        "только продажи",
        "а клиенты?",
        "по категории электроника",
        "минимальная?",
    ],
)
def test_classify_intent_routes_short_data_followups_to_sql_agent(message: str) -> None:
    assert classify_intent(message) == INTENT_SQL_AGENT


@pytest.mark.parametrize(
    "message",
    [
        "привет, а по клиентам?",
        "здравствуйте, а по заказам?",
    ],
)
def test_classify_intent_should_route_greeting_prefixed_data_followups_to_sql_agent(message: str) -> None:
    assert classify_intent(message) == INTENT_SQL_AGENT


def test_classify_intent_should_route_polite_theme_request_to_theme_change() -> None:
    assert classify_intent("  СМЕНИ, пожалуйста, ТЕМУ на cupcake  ") == INTENT_THEME_CHANGE


@pytest.mark.parametrize(
    "message",
    [
        "customer orders for January",
        "customers by revenue",
        "sales data for January",
    ],
)
def test_classify_intent_should_not_match_short_smalltalk_inside_business_words(message: str) -> None:
    assert classify_intent(message) == INTENT_SQL_AGENT


@pytest.mark.parametrize(
    "message",
    [
        "switch the theme to nord",
        "change the theme",
        "make the theme lighter",
    ],
)
def test_classify_intent_should_route_documented_english_theme_requests_to_theme_change(message: str) -> None:
    assert classify_intent(message) == INTENT_THEME_CHANGE
