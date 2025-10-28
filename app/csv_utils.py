from __future__ import annotations

import csv
import io
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from .constants import SEPARATOR_CANDIDATES, SUPPORTED_ENCODINGS


def _decode_csv_content(content: bytes) -> str:
    last_error: Optional[Exception] = None
    for encoding in SUPPORTED_ENCODINGS:
        try:
            return content.decode(encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
    raise ValueError("Не удалось декодировать CSV. Используйте UTF-8 или CP1251.") from last_error


def normalize_csv_text(text: str) -> str:
    text = text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    semicolon_count = text.count(";")
    comma_count = text.count(",")
    tab_count = text.count("\t")
    pipe_count = text.count("|")

    most_common = max(semicolon_count, comma_count, tab_count, pipe_count)
    if most_common == 0:
        return text

    if semicolon_count == most_common and semicolon_count >= 2:
        return text.replace("\t", ";")

    if tab_count == most_common and tab_count >= 2:
        return text.replace(";", "\t")

    return text


def read_uploaded_dataframe(content: bytes) -> pd.DataFrame:
    decoded = _decode_csv_content(content)
    normalized = normalize_csv_text(decoded)
    header_line = normalized.splitlines()[0] if normalized else ""

    attempts: List[Tuple[str, Dict[str, Any]]] = []
    seen: Set[Tuple[Optional[str], Optional[str]]] = set()

    def add_attempt(label: str, *, sep: Optional[str] = None, engine: Optional[str] = None) -> None:
        key = (sep, engine)
        if key in seen:
            return
        seen.add(key)
        kwargs: Dict[str, Any] = {}
        if sep is not None:
            kwargs["sep"] = sep
        if engine is not None:
            kwargs["engine"] = engine
        attempts.append((label, kwargs))

    try:
        dialect = csv.Sniffer().sniff(normalized[:4096], delimiters=[",", ";", "\t", "|"])
        add_attempt(f"sniffer({dialect.delimiter!r})", sep=dialect.delimiter)
    except csv.Error:
        add_attempt("auto", sep=None, engine="python")

    for sep in (",", ";", "\t", "|"):
        add_attempt(f"sep={sep!r}", sep=sep)
    add_attempt("auto-python", sep=None, engine="python")
    add_attempt("default", sep=None, engine=None)

    errors: List[str] = []
    for label, kwargs in attempts:
        try:
            dataframe = pd.read_csv(io.StringIO(normalized), **kwargs)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{label}: {exc}")
            continue

        if dataframe.empty and dataframe.columns.size == 0:
            errors.append(f"{label}: пустой результат")
            continue

        sep = kwargs.get("sep")
        if sep and dataframe.columns.size == 1 and header_line.count(sep) >= 1:
            errors.append(f"{label}: предполагаемый разделитель {sep!r} не сработал (получена 1 колонка)")
            continue

        if not sep and dataframe.columns.size == 1:
            if any(symbol in header_line for symbol in (",", ";", "\t", "|")):
                errors.append(f"{label}: auto не смог разделить столбцы")
                continue

        return dataframe

    message = "Не удалось прочитать CSV. Проверьте разделители (, ; табуляция) и структуру файла."
    if errors:
        details = "; ".join(errors[:3])
        message = f"{message} Подробности: {details}"
    raise ValueError(message)
