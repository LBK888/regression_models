# -*- coding: utf-8 -*-
"""Translation for the user interface.

The source code holds English text, which doubles as the lookup key. A language
catalogue maps those English strings to the translated text, so English needs no
catalogue at all and an untranslated string degrades to readable English rather
than to a missing-key placeholder.

    from i18n import tr

    label = QLabel(tr("Model library"))
    status = tr("Found {count} model files", count=12)

Placeholders use :meth:`str.format` names, never positions, so a translation can
reorder them.

The selected language is stored in ``settings.json`` beside the program, and the
process-wide default is English.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from project_paths import SETTINGS_PATH


DEFAULT_LANGUAGE = "en"

#: Selectable languages, in the order the picker shows them.
LANGUAGES: tuple[tuple[str, str], ...] = (
    ("en", "English"),
    ("zh_TW", "繁體中文"),
)

LANGUAGE_NAMES = dict(LANGUAGES)

_catalog: dict[str, str] = {}
_language = DEFAULT_LANGUAGE
_listeners: list[Callable[[str], None]] = []


def _load_catalog(language: str) -> dict[str, str]:
    """Import one language module; an English or unknown code needs no catalogue."""

    if language == "en":
        return {}
    try:
        module = __import__(f"locales.{language}", fromlist=["MESSAGES"])
    except ImportError:
        return {}
    messages = getattr(module, "MESSAGES", {})
    return {str(key): str(value) for key, value in messages.items() if value}


def available_languages() -> tuple[tuple[str, str], ...]:
    return LANGUAGES


def current_language() -> str:
    return _language


def language_name(language: str | None = None) -> str:
    return LANGUAGE_NAMES.get(language or _language, _language)


def set_language(language: str, *, persist: bool = True) -> None:
    """Switch the active language and notify every registered listener."""

    global _language, _catalog
    if language not in LANGUAGE_NAMES:
        language = DEFAULT_LANGUAGE
    if language == _language and _catalog or (language == _language == DEFAULT_LANGUAGE):
        if persist:
            save_language(language)
        return
    _language = language
    _catalog = _load_catalog(language)
    if persist:
        save_language(language)
    for listener in tuple(_listeners):
        listener(language)


def on_language_changed(listener: Callable[[str], None]) -> None:
    """Register a callback invoked after the language changes."""

    if listener not in _listeners:
        _listeners.append(listener)


def tr(text: str, /, **placeholders: Any) -> str:
    """Translate ``text`` into the active language and fill in placeholders.

    ``text`` is the English source string. An entry missing from the catalogue
    falls through to ``text`` itself, so the interface stays usable while a
    translation is incomplete.
    """

    translated = _catalog.get(text, text)
    if not placeholders:
        return translated
    try:
        return translated.format(**placeholders)
    except (KeyError, IndexError, ValueError):
        # A malformed translation must not take the window down; fall back to
        # the English source, and to the raw string if that is broken too.
        try:
            return text.format(**placeholders)
        except (KeyError, IndexError, ValueError):
            return text


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _read_settings() -> dict[str, Any]:
    path = Path(SETTINGS_PATH)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def load_language() -> str:
    """The stored language, or English when nothing valid is stored."""

    stored = str(_read_settings().get("language") or "")
    return stored if stored in LANGUAGE_NAMES else DEFAULT_LANGUAGE


def save_language(language: str) -> None:
    settings = _read_settings()
    settings["language"] = language
    path = Path(SETTINGS_PATH)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        # A read-only install folder should not block using the program.
        pass


def initialize() -> str:
    """Apply the stored language at start-up. Returns the active language."""

    set_language(load_language(), persist=False)
    return _language
