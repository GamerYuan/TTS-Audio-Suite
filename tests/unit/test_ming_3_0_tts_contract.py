"""Contract tests for the Ming 3.0 TTS integration."""

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from engines.ming_3_0_tts import tasks
from engines.ming_3_0_tts.languages import (
    format_ming_language,
    language_to_parser_code,
    normalize_ming_language,
)
from engines.ming_3_0_tts.native_tags import (
    MING_PARALINGUISTIC_TAGS,
    protect_native_tags,
    restore_native_tags,
)
from engines.ming_3_0_tts.voices import (
    CHINESE_EMOTION_VARIANTS,
    CHINESE_VOICES_WITHOUT_EMOTION,
    ENGLISH_IP_VOICES,
    IP_VOICE_BASES,
    IP_VOICE_ID_COUNT,
    emotion_labels_for_voice,
    resolve_ip_voice,
)
from utils.text.segment_parameters import ParameterValidator


def test_language_normalization_uses_official_names():
    assert normalize_ming_language("Japanese") == "Japanese"
    assert normalize_ming_language("ja") == "Japanese"
    assert normalize_ming_language("japanese") == "Japanese"
    assert normalize_ming_language("Auto") is None
    assert normalize_ming_language("") is None
    assert format_ming_language("ko") == "Korean"
    assert language_to_parser_code("German") == "de"
    assert language_to_parser_code(None) == "en"


def test_task_modes_map_to_official_tasks_and_language_contract():
    assert tasks.resolve_official_task(tasks.TASK_TTS) == "TTS"
    assert tasks.resolve_official_task(tasks.TASK_MULTILINGUAL) == "MULTILINGUAL"
    assert tasks.resolve_official_task(tasks.TASK_VOICE_DESIGN) == "VOICE_DESIGN"
    assert tasks.resolve_official_task(tasks.TASK_FREEFORM) == "FREEFORM"

    assert tasks.accepts_language(tasks.TASK_TTS)
    assert tasks.accepts_language(tasks.TASK_MULTILINGUAL)
    assert not tasks.accepts_language(tasks.TASK_DIALECT)
    assert not tasks.accepts_language(tasks.TASK_BASIC)
    assert tasks.requires_reference_audio(tasks.TASK_BASIC)
    assert tasks.requires_reference_audio(tasks.TASK_EMOTION)
    assert tasks.requires_reference_audio(tasks.TASK_DIALECT)
    assert not tasks.requires_reference_audio(tasks.TASK_TTS)
    assert tasks.uses_native_chunking(tasks.TASK_VOICE_DESIGN)
    assert tasks.uses_native_chunking(tasks.TASK_FREEFORM)
    assert not tasks.uses_native_chunking(tasks.TASK_TTS)


def test_task_language_validation_rejects_multilingual_auto():
    with pytest.raises(ValueError, match="Multilingual TTS requires"):
        tasks.validate_task_language(tasks.TASK_MULTILINGUAL, "Auto")
    assert tasks.validate_task_language(tasks.TASK_MULTILINGUAL, "Japanese") == "Japanese"
    assert tasks.validate_task_language(tasks.TASK_DIALECT, "Japanese") is None


def test_control_values_are_official_and_validated():
    assert tasks.resolve_controls(
        tasks.TASK_BASIC, {"speed": "快速", "pitch": "低", "volume": "高"}
    ) == {"speed": "快速", "pitch": "低", "volume": "高"}
    assert tasks.resolve_controls(tasks.TASK_EMOTION, {"emotion": "惊讶"}) == {
        "emotion": "惊讶"
    }
    assert tasks.resolve_controls(tasks.TASK_DIALECT, {"dialect": "cantonese"}) == {
        "dialect": "cantonese"
    }
    assert tasks.resolve_controls(
        tasks.TASK_IP, {"ip_voice_id": "US_Stella"}
    ) == {"ip": "US_Stella"}
    assert tasks.resolve_controls(
        tasks.TASK_VOICE_DESIGN, {"voice_style": "warm narrator"}
    ) == {"style": "warm narrator"}
    assert tasks.resolve_instruction(
        tasks.TASK_FREEFORM, {"instruction": "Speak slowly."}
    ) == "Speak slowly."

    with pytest.raises(ValueError):
        tasks.resolve_controls(tasks.TASK_BASIC, {"speed": "fast"})
    with pytest.raises(ValueError):
        tasks.resolve_controls(tasks.TASK_EMOTION, {"emotion": "joyful"})
    with pytest.raises(ValueError):
        tasks.resolve_controls(tasks.TASK_VOICE_DESIGN, {"voice_style": ""})
    with pytest.raises(ValueError):
        tasks.resolve_instruction(tasks.TASK_FREEFORM, {"instruction": ""})


def test_ip_catalog_is_complete_and_uses_exact_identifiers():
    assert len(CHINESE_VOICES_WITHOUT_EMOTION) == 50
    assert len(CHINESE_EMOTION_VARIANTS) == 62
    assert sum(len(v) for v in CHINESE_EMOTION_VARIANTS.values()) == 429
    assert ENGLISH_IP_VOICES == (
        "US_Emma",
        "US_James",
        "US_Stella",
        "UK_George",
        "UK_Charlotte",
    )
    assert len(IP_VOICE_BASES) == 117
    assert IP_VOICE_ID_COUNT == 484

    assert resolve_ip_voice("US_James", "None") == "US_James"
    assert resolve_ip_voice("US_James", "") == "US_James"
    assert resolve_ip_voice("灵信远", "None") == "灵信远_中性"
    assert resolve_ip_voice("灵信远", "高兴") == "灵信远_高兴"
    assert "高兴" in emotion_labels_for_voice("灵信远")

    with pytest.raises(ValueError, match="no emotion variants"):
        resolve_ip_voice("US_James", "高兴")
    with pytest.raises(ValueError, match="does not have"):
        # 灵晓声 has no Fearful variant in the official catalog.
        resolve_ip_voice("灵晓声", "恐惧")
    with pytest.raises(ValueError, match="not in the official catalog"):
        resolve_ip_voice("NotAVoice")
    with pytest.raises(ValueError, match="Unknown Ming IP emotion"):
        resolve_ip_voice("灵信远", "Excited")


def test_native_tags_survive_character_parser_protection():
    text = "[Alice] Hello [Laughter] world [Cough] [pause:1s]"
    protected, replacements = protect_native_tags(text)
    assert "[Laughter]" not in protected
    assert "[Cough]" not in protected
    assert "[Alice]" in protected
    assert "[pause:1s]" in protected
    assert restore_native_tags(protected, replacements) == text
    assert len(MING_PARALINGUISTIC_TAGS) == 14
    assert "[Laughter]" in MING_PARALINGUISTIC_TAGS
    assert "[Question-huh]" in MING_PARALINGUISTIC_TAGS


def test_ming_parameters_are_registered_for_segment_tags():
    for parameter in ("seed", "cfg", "sigma", "temperature", "max_length", "max_decode_steps"):
        assert ParameterValidator.is_parameter_supported(parameter, "ming_3_0_tts")
    assert ParameterValidator.normalize_parameter_name("decode_steps") == "max_decode_steps"
    assert ParameterValidator.normalize_parameter_name("semantic_length") == "max_length"

    valid, message, value = ParameterValidator.validate_parameter_value("sigma", "0.3")
    assert valid, message
    assert value == 0.3
    valid, message, value = ParameterValidator.validate_parameter_value("max_length", "32")
    assert valid, message
    assert value == 32
