"""Engine node tests for Ming 3.0 TTS."""

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

NODE_PATH = REPO_ROOT / "nodes" / "engines" / "ming_3_0_tts_engine_node.py"
SPEC = importlib.util.spec_from_file_location("ming_3_0_tts_engine_node_test_module", NODE_PATH)
NODE_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(NODE_MODULE)
Ming30TTSEngineNode = NODE_MODULE.Ming30TTSEngineNode

from engines.ming_3_0_tts import tasks
from engines.ming_3_0_tts.ming_3_0_tts_downloader import Ming30TTSDownloader


@pytest.fixture(autouse=True)
def no_download_or_source_scan(monkeypatch):
    monkeypatch.setattr(Ming30TTSDownloader, "list_local_models", lambda self: [])
    monkeypatch.setattr(Ming30TTSDownloader, "resolve_upstream_source", lambda self, path: None)


def make_config(node, **overrides):
    kwargs = dict(
        model="Ming-3.0-tts-0.5B",
        device="auto",
        task_mode=tasks.TASK_TTS,
        language="English",
        max_length=24,
        max_decode_steps=1000,
        cfg=2.0,
        sigma=0.25,
        temperature=0.0,
    )
    kwargs.update(overrides)
    (engine_data,) = node.create_engine_config(**kwargs)
    return engine_data["config"]


def test_input_types_expose_official_contract():
    inputs = Ming30TTSEngineNode.INPUT_TYPES()
    assert inputs["required"]["model"][0][0] == "Ming-3.0-tts-0.5B"
    assert inputs["required"]["device"][0] == ["auto", "cuda"]
    assert inputs["required"]["task_mode"][0] == list(tasks.TASK_MODE_OPTIONS)
    assert inputs["required"]["language"][0][0] == "Auto"
    assert inputs["optional"]["ip_voice"][1]["default"] == "US_James"
    assert "cache_prompts" in inputs["optional"]
    assert "upstream_path" in inputs["optional"]
    assert inputs["required"]["max_length"][1]["default"] == 24


def test_local_models_are_explicit_options(monkeypatch):
    monkeypatch.setattr(
        Ming30TTSDownloader, "list_local_models", lambda self: ["Ming-3.0-tts-0.5B"]
    )
    options = Ming30TTSEngineNode._model_options()
    assert options[0] == "Ming-3.0-tts-0.5B"
    assert "local:Ming-3.0-tts-0.5B" in options


def test_tts_config_is_passed_through():
    node = Ming30TTSEngineNode()
    config = make_config(node, language="Japanese", upstream_path="auto")
    assert config["engine_type"] == "ming_3_0_tts"
    assert config["task_mode"] == tasks.TASK_TTS
    assert config["language"] == "Japanese"
    assert config["cfg"] == 2.0
    assert config["sigma"] == 0.25
    assert config["text_normalization"] is False
    assert config["cache_prompts"] is True


def test_multilingual_requires_one_of_five_languages():
    node = Ming30TTSEngineNode()
    with pytest.raises(ValueError, match="requires an explicit language"):
        make_config(node, task_mode=tasks.TASK_MULTILINGUAL, language="Auto")
    config = make_config(node, task_mode=tasks.TASK_MULTILINGUAL, language="Korean")
    assert config["language"] == "Korean"


def test_ip_resolution_uses_exact_catalog_identifier():
    node = Ming30TTSEngineNode()
    config = make_config(
        node,
        task_mode=tasks.TASK_IP,
        ip_voice="灵信远",
        ip_emotion="高兴",
    )
    assert config["ip_voice_id"] == "灵信远_高兴"

    with pytest.raises(ValueError, match="no emotion variants"):
        make_config(node, task_mode=tasks.TASK_IP, ip_voice="US_James", ip_emotion="高兴")


def test_voice_design_and_freeform_require_text():
    node = Ming30TTSEngineNode()
    with pytest.raises(ValueError, match="Voice design requires"):
        make_config(node, task_mode=tasks.TASK_VOICE_DESIGN, voice_style="")
    with pytest.raises(ValueError, match="Freeform instruction requires"):
        make_config(node, task_mode=tasks.TASK_FREEFORM, instruction="")

    style_config = make_config(
        node, task_mode=tasks.TASK_VOICE_DESIGN, voice_style="warm narrator"
    )
    assert style_config["voice_style"] == "warm narrator"
    freeform_config = make_config(
        node, task_mode=tasks.TASK_FREEFORM, instruction="Speak slowly."
    )
    assert freeform_config["instruction"] == "Speak slowly."


def test_dialect_ignores_language_value():
    node = Ming30TTSEngineNode()
    config = make_config(node, task_mode=tasks.TASK_DIALECT, language="Japanese")
    assert config["language"] == "Auto"
    assert config["dialect"] == "cantonese"


def test_invalid_device_and_task_are_rejected():
    node = Ming30TTSEngineNode()
    with pytest.raises(ValueError, match="Unsupported Ming 3.0 TTS device"):
        make_config(node, device="cpu")
    with pytest.raises(ValueError, match="Unknown Ming 3.0 TTS task"):
        make_config(node, task_mode="Not a task")
