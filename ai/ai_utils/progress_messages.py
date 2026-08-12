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

DEFAULT_PROGRESS_FALLBACK_MESSAGE = "думает..."


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
        "проверяет доступные таблицы",
        "сверяется со списком доступных таблиц",
        "просматривает доступные таблицы",
        "подбирает подходящие таблицы",
        "уточняет, какие таблицы доступны",
        "собирает список доступных таблиц",
        "ищет релевантные таблицы",
        "анализирует доступные таблицы",
        "определяет нужные таблицы",
        "сопоставляет запрос с доступными таблицами",
        "открывает каталог таблиц",
        "перебирает разрешенные таблицы",
        "ищет таблицы под задачу",
        "смотрит, где могут быть нужные данные",
        "проверяет карту доступных данных",
        "собирает варианты таблиц для запроса",
        "ориентируется в доступных источниках",
        "выбирает таблицы-кандидаты",
        "сравнивает запрос с описаниями таблиц",
        "готовит список таблиц для дальнейшей проверки",
    ),
}

_GET_TABLE_METADATA_MESSAGES: dict[GetTableMetadataStage, tuple[str, ...]] = {
    GetTableMetadataStage.START: (
        "изучает структуру таблицы",
        "разбирает поля и связи таблицы",
        "проверяет колонки таблицы",
        "смотрит описание таблицы",
        "уточняет состав таблицы",
        "анализирует схему таблицы",
        "проверяет метаданные таблицы",
        "просматривает карточку таблицы",
        "сверяет структуру таблицы",
        "определяет нужные поля таблицы",
        "читает подробности по таблице",
        "раскрывает схему выбранной таблицы",
        "смотрит типы и назначение полей",
        "проверяет, какие колонки можно использовать",
        "сверяет таблицу с ее описанием",
        "изучает доступные поля таблицы",
        "проверяет правила работы с таблицей",
        "ищет нужные колонки в метаданных",
        "уточняет ограничения для таблицы",
        "просматривает техническую карточку таблицы",
    ),
    GetTableMetadataStage.FOUND: (
        "получил описание таблицы",
        "нашел описание таблицы",
        "подтвердил структуру таблицы",
        "собрал данные по таблице",
        "подготовил описание таблицы",
        "уточнил параметры таблицы",
        "зафиксировал метаданные таблицы",
        "выделил нужные поля таблицы",
        "проверил описание таблицы",
        "сопоставил таблицу с запросом",
        "нашел карточку таблицы",
        "получил структуру выбранной таблицы",
        "нашел нужные поля в описании таблицы",
        "подтвердил доступность метаданных",
        "разобрал состав таблицы",
        "сверил поля таблицы с задачей",
        "готовит таблицу к использованию в SQL",
        "нашел технические детали таблицы",
        "уточнил схему таблицы",
        "подготовил метаданные для построения запроса",
    ),
    GetTableMetadataStage.NOT_FOUND: (
        "не нашел подходящую таблицу, уточняет запрос",
        "не видит нужную таблицу, уточняет формулировку",
        "не нашел таблицу по запросу, пробует уточнить",
        "не нашел совпадение по таблице, переформулирует",
        "не находит подходящую таблицу, уточняет детали",
        "не определил нужную таблицу, проверяет запрос",
        "не видит релевантную таблицу, ищет другой вариант",
        "не нашел таблицу в списке, уточняет задачу",
        "не обнаружил подходящую таблицу, перепроверяет запрос",
        "не сопоставил таблицу с запросом, уточняет критерии",
        "не нашел разрешенную таблицу с таким именем",
        "не видит таблицу в доступном каталоге",
        "не нашел карточку таблицы, проверяет название",
        "не может открыть метаданные таблицы, уточняет выбор",
        "не нашел таблицу среди разрешенных источников",
        "не видит подходящую карточку таблицы",
        "не сопоставил название таблицы с каталогом",
        "не нашел доступное описание таблицы",
        "не видит таблицу в allowlist, ищет безопасный вариант",
        "не нашел метаданные, проверяет другую формулировку",
    ),
}

_VALIDATE_SQL_MESSAGES: dict[ValidateSqlStage, tuple[str, ...]] = {
    ValidateSqlStage.START: (
        "проверяет SQL-запрос",
        "сверяет SQL-запрос с правилами",
        "анализирует SQL-запрос",
        "проверяет корректность SQL-запроса",
        "уточняет детали SQL-запроса",
        "просматривает SQL-запрос",
        "сопоставляет SQL-запрос с ограничениями",
        "проводит проверку SQL-запроса",
        "оценивает SQL-запрос",
        "проверяет SQL-запрос перед запуском",
        "проверяет, что SQL безопасен для чтения",
        "сверяет SQL с разрешенными таблицами",
        "проверяет маршрут выполнения SQL",
        "разбирает SQL по правилам источника",
        "проверяет SQL на допустимые операции",
        "ищет таблицы и условия в SQL-запросе",
        "проверяет SQL перед выдачей токена выполнения",
        "сверяет SQL с доступными источниками данных",
        "проверяет диалект и таблицы SQL-запроса",
        "готовит SQL к безопасной проверке",
    ),
    ValidateSqlStage.RETRY: (
        "исправляет SQL и пробует снова",
        "корректирует SQL и повторяет попытку",
        "дорабатывает SQL-запрос и запускает повторно",
        "уточняет SQL-запрос и проверяет снова",
        "пересобирает SQL-запрос и пробует еще раз",
        "исправляет формулировку SQL и повторяет",
        "перепроверяет SQL и делает новую попытку",
        "адаптирует SQL-запрос и пробует снова",
        "обновляет SQL-запрос и повторяет проверку",
        "подправляет SQL и запускает повторную проверку",
        "переписывает SQL с учетом правил доступа",
        "ищет более точный вариант SQL-запроса",
        "уточняет таблицы и условия в SQL",
        "пересобирает запрос без спорных частей",
        "проверка не прошла, готовит новую версию SQL",
        "подбирает корректный маршрут для SQL-запроса",
        "устраняет проблему в SQL и проверяет заново",
        "уточняет SQL, чтобы пройти валидацию",
        "перепроверяет ограничения и меняет SQL",
        "готовит безопасную версию SQL-запроса",
    ),
    ValidateSqlStage.OK: (
        "проверил SQL-запрос",
        "подтвердил корректность SQL-запроса",
        "завершил проверку SQL-запроса",
        "убедился, что SQL-запрос правильный",
        "подготовил SQL-запрос к выполнению",
        "согласовал SQL-запрос с правилами",
        "успешно проверил SQL-запрос",
        "утвердил SQL-запрос",
        "убедился, SQL-запрос готов к запуску",
        "завершил проверку SQL-запроса",
        "подтвердил безопасный маршрут SQL-запроса",
        "проверил SQL и подготовил его к выполнению",
        "убедился, что SQL использует разрешенные таблицы",
        "выдал SQL-запросу допуск к выполнению",
        "подтвердил диалект и источник данных",
        "зафиксировал проверенную версию SQL-запроса",
        "подготовил токен для выполнения SQL",
        "проверил SQL на чтение и доступные таблицы",
        "согласовал SQL с правилами источника",
        "готов перейти к запуску SQL-запроса",
    ),
}

_EXECUTE_VALIDATED_SQL_MESSAGES: dict[ExecuteValidatedSqlStage, tuple[str, ...]] = {
    ExecuteValidatedSqlStage.START: (
        "выполняет проверенный SQL-запрос",
        "запускает проверенный SQL-запрос",
        "отправляет проверенный SQL-запрос в базу",
        "инициирует выполнение SQL-запроса",
        "начинает выполнение SQL-запроса",
        "переходит к выполнению SQL-запроса",
        "выполняет SQL-запрос по подтвержденным правилам",
        "применяет проверенный SQL-запрос",
        "запускает SQL-запрос после проверки",
        "обрабатывает SQL-запрос для выполнения",
        "готовит проверенный SQL к запуску",
        "проверяет токен выполнения SQL-запроса",
        "переходит от проверки к выполнению SQL",
        "поднимает проверенный запрос на выполнение",
        "сверяет разрешение на запуск SQL-запроса",
        "открывает этап выполнения SQL-запроса",
        "готовит маршрут к нужной базе данных",
        "берет проверенный SQL из состояния запроса",
        "проверяет срок действия SQL-допуска",
        "запускает выполнение по подтвержденному маршруту",
    ),
    ExecuteValidatedSqlStage.PROBLEM: (
        "уточняет SQL-запрос",
        "проверяет, почему SQL-запрос не подходит",
        "исправляет параметры SQL-запроса",
        "пересматривает SQL-запрос",
        "адаптирует SQL-запрос под ограничения",
        "корректирует SQL-запрос перед повтором",
        "уточняет формулировку SQL-запроса",
        "перестраивает SQL-запрос",
        "подправляет SQL-запрос для повторной попытки",
        "согласовывает SQL-запрос с условиями выполнения",
        "проверяет данные проверенного SQL-запроса",
        "не может использовать этот SQL-допуск, уточняет запрос",
        "проверяет актуальность разрешения на выполнение",
        "сверяет SQL-запрос с текущим диалогом",
        "проверяет, можно ли повторно запустить этот SQL",
        "не запускает SQL без действующего подтверждения",
        "уточняет состояние проверенного SQL-запроса",
        "ищет причину, почему выполнение нельзя продолжить",
        "проверяет ограничения перед повторной попыткой",
        "готовит новый безопасный путь выполнения",
    ),
    ExecuteValidatedSqlStage.DB_CONNECTING: (
        "ожидает подключение к базе",
        "проверяет подключение к базе",
        "устанавливает соединение с базой",
        "подготавливает подключение к базе",
        "инициирует подключение к базе",
        "переходит к подключению к базе",
        "сверяет доступ к базе",
        "проверяет готовность подключения к базе",
        "налаживает соединение с базой",
        "подключается к базе данных",
        "проверяет, доступен ли нужный источник данных",
        "ищет настроенный инструмент запроса к базе",
        "готовит канал выполнения SQL-запроса",
        "сверяет подключение с выбранным источником",
        "подготавливает доступ к базе данных",
        "проверяет конфигурацию запроса к базе",
        "ожидает готовность SQL-инструмента",
        "смотрит, можно ли отправить запрос в базу",
        "проверяет маршрут подключения к данным",
        "готовит соединение для выполнения запроса",
    ),
    ExecuteValidatedSqlStage.DB_WAITING: (
        "ждет ответ от базы",
        "ожидает результат от базы",
        "получает ответ от базы",
        "обрабатывает ответ базы",
        "дожидается результата запроса",
        "считывает данные из базы",
        "отслеживает выполнение запроса в базе",
        "принимает результат от базы",
        "проверяет ответ базы",
        "собирает результат запроса из базы",
        "ждет, пока база выполнит проверенный запрос",
        "база считает результат, это может занять немного времени",
        "ожидает данные от источника, запрос уже отправлен",
        "держит соединение с базой и ждет результат",
        "база обрабатывает SQL-запрос и готовит строки ответа",
        "ждет завершения запроса на стороне базы данных",
        "получает данные из базы по подтвержденному SQL",
        "ожидает, когда источник данных вернет результат",
        "запрос выполняется в базе, собирает ответ",
        "ждет ответ базы и не запускает лишних запросов",
    ),
    ExecuteValidatedSqlStage.DB_ERROR: (
        "исправляет SQL и пробует снова",
        "обрабатывает ошибку базы и повторяет попытку",
        "уточняет SQL после ошибки базы",
        "корректирует запрос после ошибки базы",
        "перепроверяет SQL из-за ошибки базы",
        "повторяет выполнение после исправления ошибки",
        "анализирует ошибку базы и корректирует SQL",
        "обновляет SQL и запускает повторно",
        "перезапускает запрос после исправления",
        "снижает риск ошибки и повторяет запрос",
        "разбирает ответ базы с ошибкой",
        "проверяет, что помешало базе выполнить запрос",
        "уточняет SQL по сообщению от базы",
        "ищет причину сбоя при выполнении запроса",
        "перепроверяет запрос после отказа базы",
        "готовит корректировку после ошибки выполнения",
        "анализирует сбой запроса на стороне базы",
        "обрабатывает проблему выполнения и готовит повтор",
        "сверяет SQL с ошибкой, полученной от базы",
        "ищет безопасный способ повторить запрос",
    ),
    ExecuteValidatedSqlStage.FINAL_ANALYSIS: (
        "анализирует результат запроса",
        "проверяет итоговые данные запроса",
        "сопоставляет результат с запросом",
        "готовит итог по результату запроса",
        "оценивает полученные данные",
        "уточняет вывод по результатам запроса",
        "формирует финальный анализ результата",
        "обобщает результат запроса",
        "проверяет полноту результата запроса",
        "подтверждает итог запроса",
        "готовит финальный ответ",
        "разбирает полученные строки ответа",
        "переводит результат запроса в понятный вывод",
        "проверяет, что данные отвечают на вопрос",
        "собирает итог по данным из базы",
        "сверяет результат с исходной задачей",
        "готовит объяснение найденных данных",
        "выделяет главное в результате запроса",
        "проверяет таблицу результата перед ответом",
        "формирует ответ на основе полученных данных",
        "сводит результат запроса в финальный вывод",
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
