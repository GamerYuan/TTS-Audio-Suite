"""Adapter tests for Ming 3.0 TTS, using a fake official runtime."""

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from engines.adapters.ming_3_0_tts_adapter import Ming30TTSEngineAdapter
from engines.ming_3_0_tts import tasks
from utils.audio.cache import get_audio_cache


class FakeEngine:
    def __init__(self, audio=None):
        self.calls = []
        self.sample_rate = 44100
        self.audio = audio if audio is not None else torch.zeros(1, 4410)

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return {"audio": self.audio.clone(), "sample_rate": self.sample_rate}


def make_adapter(monkeypatch, config, fake_engine):
    adapter = Ming30TTSEngineAdapter(config)
    monkeypatch.setattr(adapter, "_ensure_model_loaded", lambda: None)
    monkeypatch.setattr(adapter, "_get_engine", lambda: fake_engine)
    return adapter


def base_config(**overrides):
    config = {
        "engine_type": "ming_3_0_tts",
        "model": "Ming-3.0-tts-0.5B",
        "device": "cuda",
        "task_mode": tasks.TASK_TTS,
        "language": "English",
        "speed": tasks.DEFAULT_SPEED,
        "pitch": tasks.DEFAULT_PITCH,
        "volume": tasks.DEFAULT_VOLUME,
        "emotion": tasks.DEFAULT_EMOTION,
        "dialect": tasks.DEFAULT_DIALECT,
        "ip_voice": "US_James",
        "ip_emotion": "None",
        "ip_voice_id": "US_James",
        "voice_style": "",
        "instruction": "",
        "max_length": 24,
        "max_decode_steps": 1000,
        "cfg": 2.0,
        "sigma": 0.25,
        "temperature": 0.0,
        "text_normalization": False,
        "cache_prompts": True,
        "upstream_path": "auto",
    }
    config.update(overrides)
    return config


@pytest.fixture(autouse=True)
def clear_audio_cache():
    get_audio_cache().clear_cache()
    yield
    get_audio_cache().clear_cache()


def test_tts_with_reference_and_transcript(monkeypatch):
    fake = FakeEngine()
    adapter = make_adapter(monkeypatch, base_config(), fake)
    voice_ref = {
        "audio": {"waveform": torch.randn(1, 1, 16000), "sample_rate": 16000},
        "reference_text": "Hello there.",
    }

    audio = adapter.generate_single("Hello world", voice_ref, seed=7, character_name="Alice")

    assert audio.shape == (1, 4410)
    call = fake.calls[0]
    assert call["task"] == "TTS"
    assert call["language_id"] == "English"
    assert call["controls"] is None
    assert call["instruction"] is None
    assert call["prompt_text"] == "Hello there."
    assert call["prompt_wav_path"] and Path(call["prompt_wav_path"]).is_file()
    assert call["use_spk_emb"] is True
    assert call["use_zero_spk_emb"] is False
    assert call["seed"] == 7


def test_tts_without_reference_uses_upstream_default_speaker_behavior(monkeypatch):
    fake = FakeEngine()
    adapter = make_adapter(monkeypatch, base_config(), fake)

    adapter.generate_single("Hello world", None, seed=0, character_name="narrator")

    call = fake.calls[0]
    assert call["prompt_wav_path"] is None
    assert call["prompt_text"] is None
    assert call["use_spk_emb"] is False
    assert call["use_zero_spk_emb"] is False


def test_tts_with_audio_but_no_transcript_uses_speaker_embedding_only(monkeypatch):
    fake = FakeEngine()
    adapter = make_adapter(monkeypatch, base_config(), fake)
    voice_ref = {"audio": {"waveform": torch.randn(1, 1, 16000), "sample_rate": 16000}}

    adapter.generate_single("Hello world", voice_ref, seed=1)

    call = fake.calls[0]
    assert call["prompt_wav_path"]
    assert call["prompt_text"] is None
    assert call["use_spk_emb"] is True


def test_basic_controls_require_reference_and_ignore_transcript(monkeypatch):
    fake = FakeEngine()
    adapter = make_adapter(
        monkeypatch,
        base_config(task_mode=tasks.TASK_BASIC, speed="快速", pitch="低", volume="高"),
        fake,
    )
    voice_ref = {
        "audio": {"waveform": torch.randn(1, 1, 16000), "sample_rate": 16000},
        "reference_text": "should not be sent",
    }
    adapter.generate_single("test", voice_ref)
    call = fake.calls[0]
    assert call["task"] == "BASIC"
    assert call["controls"] == {"speed": "快速", "pitch": "低", "volume": "高"}
    assert call["language_id"] is None
    assert call["prompt_text"] is None
    assert call["use_spk_emb"] is True

    missing = make_adapter(monkeypatch, base_config(task_mode=tasks.TASK_BASIC), FakeEngine())
    with pytest.raises(ValueError, match="requires a reference voice"):
        missing.generate_single("test", None)


def test_dialect_task_never_sends_language(monkeypatch):
    fake = FakeEngine()
    adapter = make_adapter(
        monkeypatch,
        base_config(task_mode=tasks.TASK_DIALECT, language="Japanese", dialect="cantonese"),
        fake,
    )
    voice_ref = {"audio": {"waveform": torch.randn(1, 1, 16000), "sample_rate": 16000}}
    adapter.generate_single("test", voice_ref)
    call = fake.calls[0]
    assert call["task"] == "DIALECT"
    assert call["language_id"] is None
    assert call["controls"] == {"dialect": "cantonese"}


def test_ip_and_voice_design_do_not_use_reference_audio(monkeypatch):
    fake = FakeEngine()
    adapter = make_adapter(
        monkeypatch,
        base_config(task_mode=tasks.TASK_IP, ip_voice_id="灵信远_高兴"),
        fake,
    )
    voice_ref = {"audio": {"waveform": torch.randn(1, 1, 16000), "sample_rate": 16000}}
    adapter.generate_single("test", voice_ref)
    call = fake.calls[0]
    assert call["task"] == "IP"
    assert call["controls"] == {"ip": "灵信远_高兴"}
    assert call["prompt_wav_path"] is None
    assert call["use_spk_emb"] is False

    design = make_adapter(
        monkeypatch,
        base_config(task_mode=tasks.TASK_VOICE_DESIGN, voice_style="warm narrator"),
        FakeEngine(),
    )
    design.generate_single("test", voice_ref)
    design_call = design._get_engine().calls[0]
    assert design_call["task"] == "VOICE_DESIGN"
    assert design_call["controls"] == {"style": "warm narrator"}
    assert design_call["prompt_wav_path"] is None
    assert design_call["prompt_text"] is None


def test_freeform_instruction_cannot_combine_controls(monkeypatch):
    fake = FakeEngine()
    adapter = make_adapter(
        monkeypatch,
        base_config(task_mode=tasks.TASK_FREEFORM, instruction="Speak slowly and warmly."),
        fake,
    )
    adapter.generate_single("test", None)
    call = fake.calls[0]
    assert call["task"] == "FREEFORM"
    assert call["instruction"] == "Speak slowly and warmly."
    assert call["controls"] is None
    assert call["language_id"] is None

    empty = make_adapter(monkeypatch, base_config(task_mode=tasks.TASK_FREEFORM), FakeEngine())
    with pytest.raises(ValueError, match="Freeform instruction"):
        empty.generate_single("test", None)


def test_multilingual_requires_language(monkeypatch):
    fake = FakeEngine()
    adapter = make_adapter(monkeypatch, base_config(task_mode=tasks.TASK_MULTILINGUAL), fake)
    with pytest.raises(ValueError, match="requires an explicit language"):
        adapter.generate_single("test", None)

    valid = make_adapter(
        monkeypatch,
        base_config(task_mode=tasks.TASK_MULTILINGUAL, language="Japanese"),
        FakeEngine(),
    )
    valid.generate_single("test", None)
    assert valid._get_engine().calls[0]["language_id"] == "Japanese"


def test_cache_reuses_identical_request(monkeypatch):
    fake = FakeEngine()
    adapter = make_adapter(monkeypatch, base_config(), fake)
    voice_ref = {"audio": {"waveform": torch.randn(1, 1, 16000), "sample_rate": 16000}}

    first = adapter.generate_single("same text", voice_ref, seed=11, character_name="A")
    second = adapter.generate_single("same text", voice_ref, seed=11, character_name="A")

    assert len(fake.calls) == 1
    assert torch.equal(first, second)


def test_cache_key_changes_with_seed_and_controls(monkeypatch):
    fake = FakeEngine()
    adapter = make_adapter(monkeypatch, base_config(), fake)
    voice_ref = {"audio": {"waveform": torch.randn(1, 1, 16000), "sample_rate": 16000}}

    adapter.generate_single("same text", voice_ref, seed=1, character_name="A")
    adapter.generate_single("same text", voice_ref, seed=2, character_name="A")
    assert len(fake.calls) == 2


def test_stable_prompt_paths_are_content_addressed(monkeypatch):
    adapter = make_adapter(monkeypatch, base_config(), FakeEngine())
    waveform = torch.randn(1, 1, 8000)
    first = adapter._stable_prompt_path_from_tensor(waveform, 16000)
    second = adapter._stable_prompt_path_from_tensor(waveform.clone(), 16000)
    changed = adapter._stable_prompt_path_from_tensor(waveform + 0.1, 16000)

    assert first == second
    assert first != changed
    assert Path(first).is_file()
    assert Path(changed).is_file()
    import soundfile as sf

    data, sample_rate = sf.read(first)
    assert sample_rate == 16000
    assert data.shape[0] == 8000


def test_audio_hash_component_is_stable_for_reference(monkeypatch):
    adapter = make_adapter(monkeypatch, base_config(), FakeEngine())
    waveform = torch.randn(1, 1, 4000)
    reference = {"audio": {"waveform": waveform, "sample_rate": 16000}}
    first = adapter._resolve_voice_reference(reference, tasks.TASK_TTS)
    second = adapter._resolve_voice_reference(reference, tasks.TASK_TTS)
    changed = adapter._resolve_voice_reference(
        {"audio": {"waveform": waveform + 0.1, "sample_rate": 16000}}, tasks.TASK_TTS
    )
    assert first["audio_component"] == second["audio_component"]
    assert first["audio_component"] != changed["audio_component"]
    assert first["prompt_wav_path"] == second["prompt_wav_path"]
    assert first["prompt_wav_path"] != changed["prompt_wav_path"]
