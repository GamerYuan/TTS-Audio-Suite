"""Task and control contract for the official Ming 3.0 TTS runtime.

Ming exposes one task per call. Structured tasks build their own instruction
from controls; FREEFORM uses a natural-language instruction and cannot be
combined with controls. Keep the official task names and control values in one
place so the node UI, adapter, logs and tests cannot drift.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from engines.ming_3_0_tts.languages import (
    MULTILINGUAL_LANGUAGE_OPTIONS,
    normalize_ming_language,
)

TASK_TTS = "TTS (voice cloning)"
TASK_MULTILINGUAL = "Multilingual TTS"
TASK_BASIC = "Basic attributes"
TASK_EMOTION = "Emotion"
TASK_DIALECT = "Dialect"
TASK_IP = "Built-in voice (IP)"
TASK_VOICE_DESIGN = "Voice design"
TASK_FREEFORM = "Freeform instruction"

TASK_MODE_OPTIONS = (
    TASK_TTS,
    TASK_MULTILINGUAL,
    TASK_BASIC,
    TASK_EMOTION,
    TASK_DIALECT,
    TASK_IP,
    TASK_VOICE_DESIGN,
    TASK_FREEFORM,
)

# Official task names accepted by tts_tasks.build_task_generation_kwargs.
OFFICIAL_TASK_BY_MODE = {
    TASK_TTS: "TTS",
    TASK_MULTILINGUAL: "MULTILINGUAL",
    TASK_BASIC: "BASIC",
    TASK_EMOTION: "EMOTION",
    TASK_DIALECT: "DIALECT",
    TASK_IP: "IP",
    TASK_VOICE_DESIGN: "VOICE_DESIGN",
    TASK_FREEFORM: "FREEFORM",
}

# Values are the exact model inputs from tts_tasks.py.
BASIC_SPEED_VALUES = ("慢速", "中速", "快速")
BASIC_PITCH_VALUES = ("低", "中", "高")
BASIC_VOLUME_VALUES = ("低", "中", "高")
EMOTION_VALUES = ("高兴", "悲伤", "愤怒", "恐惧", "厌恶", "惊讶", "中性")
DIALECT_VALUES = ("cantonese", "chuanyu dialect")

DEFAULT_SPEED = "中速"
DEFAULT_PITCH = "中"
DEFAULT_VOLUME = "中"
DEFAULT_EMOTION = "中性"
DEFAULT_DIALECT = "cantonese"

# Only these tasks accept reference audio according to the official table.
# FREEFORM accepts an optional reference, but does not accept its transcript.
_TASKS_WITH_REFERENCE = {
    TASK_TTS,
    TASK_MULTILINGUAL,
    TASK_BASIC,
    TASK_EMOTION,
    TASK_DIALECT,
    TASK_FREEFORM,
}
_TASKS_REQUIRING_REFERENCE = {TASK_BASIC, TASK_EMOTION, TASK_DIALECT}
_TASKS_WITH_LANGUAGE = {TASK_TTS, TASK_MULTILINGUAL}
_TASKS_WITH_TRANSCRIPT = {TASK_TTS, TASK_MULTILINGUAL}
# Voice Design and Freeform split text with the official semantic chunker and
# reuse the first generated fragment as the reference for later fragments.
# Pre-splitting them would change that native behavior.
_NATIVE_CHUNKING_TASKS = {TASK_VOICE_DESIGN, TASK_FREEFORM}


def resolve_official_task(task_mode: str) -> str:
    """Map a node mode label to the official task name."""
    try:
        return OFFICIAL_TASK_BY_MODE[str(task_mode)]
    except KeyError as error:
        raise ValueError(
            f"Unknown Ming 3.0 TTS task mode {task_mode!r}. "
            f"Expected one of: {', '.join(TASK_MODE_OPTIONS)}"
        ) from error


def accepts_language(task_mode: str) -> bool:
    return task_mode in _TASKS_WITH_LANGUAGE


def accepts_reference_transcript(task_mode: str) -> bool:
    return task_mode in _TASKS_WITH_TRANSCRIPT


def uses_reference_audio(task_mode: str) -> bool:
    return task_mode in _TASKS_WITH_REFERENCE


def requires_reference_audio(task_mode: str) -> bool:
    return task_mode in _TASKS_REQUIRING_REFERENCE


def uses_native_chunking(task_mode: str) -> bool:
    return task_mode in _NATIVE_CHUNKING_TASKS


def is_structured_task(task_mode: str) -> bool:
    return task_mode in {
        TASK_BASIC,
        TASK_EMOTION,
        TASK_DIALECT,
        TASK_IP,
        TASK_VOICE_DESIGN,
    }


def validate_task_language(task_mode: str, language: Optional[str]) -> Optional[str]:
    """Validate and normalize a language for the selected task.

    Returns the official language name, or ``None`` when the task does not use
    a language_id. Raises for an unsupported Multilingual selection rather than
    silently generating a different task.
    """
    normalized = normalize_ming_language(language)
    if task_mode == TASK_MULTILINGUAL:
        if normalized not in MULTILINGUAL_LANGUAGE_OPTIONS:
            raise ValueError(
                "Ming Multilingual TTS requires an explicit language: "
                + ", ".join(MULTILINGUAL_LANGUAGE_OPTIONS)
                + ". Use TTS (voice cloning) for Chinese or English."
            )
        return normalized
    if task_mode == TASK_TTS:
        return normalized
    # DIALECT takes no language_id. Structured and Freeform tasks in the
    # official table do not use target_language either.
    return None


def resolve_controls(task_mode: str, config: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """Build the exact official controls mapping for the selected task."""
    if task_mode == TASK_BASIC:
        speed = str(config.get("speed", DEFAULT_SPEED))
        pitch = str(config.get("pitch", DEFAULT_PITCH))
        volume = str(config.get("volume", DEFAULT_VOLUME))
        for name, value, allowed in (
            ("speed", speed, BASIC_SPEED_VALUES),
            ("pitch", pitch, BASIC_PITCH_VALUES),
            ("volume", volume, BASIC_VOLUME_VALUES),
        ):
            if value not in allowed:
                raise ValueError(
                    f"Ming Basic attribute {name}={value!r} is not official. "
                    f"Expected one of: {', '.join(allowed)}"
                )
        return {"speed": speed, "pitch": pitch, "volume": volume}

    if task_mode == TASK_EMOTION:
        emotion = str(config.get("emotion", DEFAULT_EMOTION))
        if emotion not in EMOTION_VALUES:
            raise ValueError(
                f"Ming emotion={emotion!r} is not official. "
                f"Expected one of: {', '.join(EMOTION_VALUES)}"
            )
        return {"emotion": emotion}

    if task_mode == TASK_DIALECT:
        dialect = str(config.get("dialect", DEFAULT_DIALECT))
        if dialect not in DIALECT_VALUES:
            raise ValueError(
                f"Ming dialect={dialect!r} is not official. "
                f"Expected one of: {', '.join(DIALECT_VALUES)}"
            )
        return {"dialect": dialect}

    if task_mode == TASK_IP:
        ip_voice = str(config.get("ip_voice_id") or "").strip()
        if not ip_voice:
            raise ValueError("Ming Built-in voice (IP) requires a catalog voice.")
        return {"ip": ip_voice}

    if task_mode == TASK_VOICE_DESIGN:
        style = str(config.get("voice_style") or "").strip()
        if not style:
            raise ValueError(
                "Ming Voice design requires a non-empty voice description. "
                "Describe gender, age, timbre, pace and delivery."
            )
        return {"style": style}

    return None


def resolve_instruction(task_mode: str, config: Dict[str, Any]) -> Optional[str]:
    if task_mode != TASK_FREEFORM:
        return None
    instruction = str(config.get("instruction") or "").strip()
    if not instruction:
        raise ValueError(
            "Ming Freeform instruction requires a non-empty natural-language "
            "delivery description. Do not put it in the spoken text."
        )
    return instruction


def task_summary(task_mode: str, config: Dict[str, Any], language: Optional[str] = None) -> str:
    """Compact one-line task summary for the Settings log."""
    parts = [f"task={task_mode}"]
    if language:
        parts.append(f"language={language}")
    if task_mode == TASK_BASIC:
        parts.append(
            "controls="
            f"speed:{config.get('speed', DEFAULT_SPEED)},"
            f"pitch:{config.get('pitch', DEFAULT_PITCH)},"
            f"volume:{config.get('volume', DEFAULT_VOLUME)}"
        )
    elif task_mode == TASK_EMOTION:
        parts.append(f"emotion={config.get('emotion', DEFAULT_EMOTION)}")
    elif task_mode == TASK_DIALECT:
        parts.append(f"dialect={config.get('dialect', DEFAULT_DIALECT)}")
    elif task_mode == TASK_IP:
        parts.append(f"ip={config.get('ip_voice_id', '')}")
    elif task_mode == TASK_VOICE_DESIGN:
        parts.append("style=voice description")
    elif task_mode == TASK_FREEFORM:
        parts.append("instruction=freeform")
    return " | ".join(parts)
