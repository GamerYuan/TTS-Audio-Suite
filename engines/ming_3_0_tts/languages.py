"""Language helpers for the official Ming 3.0 TTS runtime.

Ming accepts either short codes or full names as ``language_id`` and writes the
resolved name into the model prompt. The suite keeps readable display names in
the UI and maps them back to the official names here.
"""

from __future__ import annotations

from typing import Optional

from utils.models.language_mapper import resolve_language_alias

LANGUAGE_OPTIONS = (
    "Auto",
    "Chinese",
    "English",
    "German",
    "Spanish",
    "French",
    "Korean",
    "Japanese",
)

# The five languages Ming-3.0 adds on top of Chinese/English. The official
# MULTILINGUAL task requires an explicit language_id for these.
MULTILINGUAL_LANGUAGE_OPTIONS = (
    "German",
    "Spanish",
    "French",
    "Korean",
    "Japanese",
)

_CODE_TO_NAME = {
    "zh": "Chinese",
    "zh-cn": "Chinese",
    "zh-tw": "Chinese",
    "zho": "Chinese",
    "cmn": "Chinese",
    "chinese": "Chinese",
    "mandarin": "Chinese",
    "en": "English",
    "eng": "English",
    "english": "English",
    "de": "German",
    "deu": "German",
    "ger": "German",
    "german": "German",
    "es": "Spanish",
    "spa": "Spanish",
    "spanish": "Spanish",
    "fr": "French",
    "fra": "French",
    "fre": "French",
    "french": "French",
    "ko": "Korean",
    "kor": "Korean",
    "korean": "Korean",
    "ja": "Japanese",
    "jpn": "Japanese",
    "japanese": "Japanese",
    # Cantonese is a Chinese dialect for language tagging. The DIALECT task
    # controls the actual dialect and takes no language_id.
    "yue": "Chinese",
    "zh-yue": "Chinese",
    "cantonese": "Chinese",
}

_LANGUAGE_TO_PARSER_CODE = {
    "Chinese": "zh",
    "English": "en",
    "German": "de",
    "Spanish": "es",
    "French": "fr",
    "Korean": "ko",
    "Japanese": "ja",
}

_AUTO_VALUES = {"", "auto", "auto-detect", "auto_detect", "none", "null", "default"}


def normalize_ming_language(language: Optional[str]) -> Optional[str]:
    """Return the official language name, or ``None`` for auto/no language."""
    if language is None:
        return None

    raw = str(language).strip()
    if raw.lower() in _AUTO_VALUES:
        return None

    direct = _CODE_TO_NAME.get(raw.lower())
    if direct:
        return direct

    resolved = resolve_language_alias(raw)
    return _CODE_TO_NAME.get(resolved.lower())


def format_ming_language(language: Optional[str]) -> str:
    """Readable display value used by logs and node summaries."""
    normalized = normalize_ming_language(language)
    return normalized or "Auto"


def language_to_parser_code(language: Optional[str]) -> str:
    """Character-parser fallback code for a Ming language display value."""
    normalized = normalize_ming_language(language)
    return _LANGUAGE_TO_PARSER_CODE.get(normalized or "English", "en")


def is_multilingual_language(language: Optional[str]) -> bool:
    return normalize_ming_language(language) in MULTILINGUAL_LANGUAGE_OPTIONS
