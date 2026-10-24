"""Ming native inline paralinguistic tags.

The suite uses square brackets for character tags, which collides with Ming's
literal paralinguistic tags such as ``[Laughter]``. The processor protects the
known official tags before character parsing and restores them immediately
before generation so they reach the official tokenizer verbatim.
"""

from __future__ import annotations

import re
from typing import Dict, Tuple

MING_PARALINGUISTIC_TAGS = (
    "[Breathing]",
    "[Cough]",
    "[Sigh]",
    "[Laughter]",
    "[Surprise-ah]",
    "[Surprise-oh]",
    "[Dissatisfaction-hnn]",
    "[Uhm]",
    "[Confirmation-en]",
    "[Question-ei]",
    "[Question-ah]",
    "[Question-en]",
    "[Question-oh]",
    "[Question-huh]",
)

_TAG_PATTERN = re.compile("|".join(re.escape(tag) for tag in MING_PARALINGUISTIC_TAGS))
_PLACEHOLDER_PATTERN = re.compile("\ue000(\\d+)\ue001")


def protect_native_tags(text: str) -> Tuple[str, Dict[str, str]]:
    """Replace official tags with parser-safe private-use placeholders."""
    if not text:
        return text, {}

    replacements: Dict[str, str] = {}

    def _replace(match: re.Match) -> str:
        placeholder = f"\ue000{len(replacements)}\ue001"
        replacements[placeholder] = match.group(0)
        return placeholder

    return _TAG_PATTERN.sub(_replace, text), replacements


def restore_native_tags(text: str, replacements: Dict[str, str]) -> str:
    """Restore placeholders created by :func:`protect_native_tags`."""
    if not text or not replacements:
        return text
    return _PLACEHOLDER_PATTERN.sub(
        lambda match: replacements.get(match.group(0), match.group(0)),
        text,
    )
