"""Processor-level tests for Ming 3.0 TTS."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from engines.ming_3_0_tts import tasks
from utils.text.character_parser import character_parser

PROCESSOR_PATH = REPO_ROOT / "nodes" / "ming_3_0_tts" / "ming_3_0_tts_processor.py"
spec = importlib.util.spec_from_file_location("ming_processor_test_module", PROCESSOR_PATH)
processor_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(processor_module)
Ming30TTSProcessor = processor_module.Ming30TTSProcessor

SRT_PATH = REPO_ROOT / "nodes" / "ming_3_0_tts" / "ming_3_0_tts_srt_processor.py"
srt_spec = importlib.util.spec_from_file_location("ming_srt_processor_test_module", SRT_PATH)
srt_module = importlib.util.module_from_spec(srt_spec)
srt_spec.loader.exec_module(srt_module)
Ming30TTSSRTProcessor = srt_module.Ming30TTSSRTProcessor


class FakeAdapter:
    SAMPLE_RATE = 44100

    def __init__(self):
        self.config = {}
        self.calls = []

    @property
    def sample_rate(self):
        return self.SAMPLE_RATE

    def update_config(self, config):
        self.config = config

    def generate_single(self, **kwargs):
        self.calls.append(kwargs)
        return torch.zeros(1, 4410)

    @staticmethod
    def describe_config(config):
        return f"task={config.get('task_mode')}"


def test_parameter_log_uses_real_ming_fields():
    log = Ming30TTSProcessor._format_parameter_log(
        {"seed": 5, "cfg": 2.5, "sigma": 0.3, "max_length": 32, "temperature": 0.1},
        {
            "cfg": 2.5,
            "sigma": 0.3,
            "max_length": 32,
            "temperature": 0.1,
            "language": "Japanese",
        },
        current_seed=5,
    )
    assert "seed=5" in log
    assert "cfg=2.5" in log
    assert "sigma=0.3" in log
    assert "max_length=32" in log
    assert "temperature=0.1" in log


def test_segment_language_switching_respects_task_contract():
    character_parser.default_language = "en"
    config = {"language": "English"}
    seg = SimpleNamespace(language="Japanese", explicit_language=True)
    assert Ming30TTSProcessor._should_apply_segment_language(
        seg, config, tasks.TASK_TTS
    )
    assert not Ming30TTSProcessor._should_apply_segment_language(
        seg, config, tasks.TASK_DIALECT
    )

    multilingual = {"language": "Japanese"}
    chinese_segment = SimpleNamespace(language="Chinese", explicit_language=True)
    assert not Ming30TTSProcessor._should_apply_segment_language(
        chinese_segment, multilingual, tasks.TASK_MULTILINGUAL
    )


def test_interrupt_check_raises(monkeypatch):
    import comfy.model_management as model_management

    processor = object.__new__(Ming30TTSProcessor)
    monkeypatch.setattr(model_management, "interrupt_processing", True, raising=False)
    with pytest.raises(InterruptedError, match="interrupted"):
        processor._check_interrupt("interrupted")


def test_combine_audio_segments_uses_shared_reporting():
    adapter = FakeAdapter()
    processor = Ming30TTSProcessor(adapter, {"task_mode": tasks.TASK_TTS})
    segments = [
        {"waveform": torch.zeros(1, 4410), "sample_rate": 44100, "text": "a"},
        {"waveform": torch.zeros(1, 4410), "sample_rate": 44100, "text": "b"},
    ]
    audio, chunk_info = processor.combine_audio_segments(
        segments, method="auto", silence_ms=0, original_text="a b", return_info=True
    )
    assert audio.shape[-1] > 0
    assert chunk_info


@pytest.mark.parametrize("method", ["auto", "silence", "crossfade"])
def test_explicit_pause_preserves_exact_samples(monkeypatch, method):
    monkeypatch.setattr(processor_module.model_management, "interrupt_processing", False)
    adapter = FakeAdapter()
    adapter.generate_single = lambda **kwargs: torch.ones(1, 4410)
    processor = Ming30TTSProcessor(adapter, {"task_mode": tasks.TASK_TTS})
    monkeypatch.setattr(processor, "_setup_character_parser", lambda text: tasks.TASK_TTS)
    monkeypatch.setattr(processor_module, "get_character_mapping", lambda *args, **kwargs: {})
    monkeypatch.setattr(character_parser, "parse_text_segments", lambda text: [
        SimpleNamespace(text=text, character="narrator", parameters={},
                        language="en", explicit_language=False)
    ])
    segments = processor.process_text(
        "Hello [pause:1s] world", {}, seed=1, enable_chunking=False,
        apply_edit_postprocessing=False,
    )
    assert segments[1]["explicit_pause"] is True
    audio, info = processor.combine_audio_segments(
        segments, method=method, silence_ms=250, return_info=True,
    )
    assert audio.shape[-1] == 4410 + 44100 + 4410
    assert torch.all(audio[..., :4410] == 1)
    assert torch.all(audio[..., 4410:48510] == 0)
    assert torch.all(audio[..., 48510:] == 1)
    assert info["junction_methods"] == ["concatenate", "concatenate"]


@pytest.mark.parametrize("method", ["auto", "silence", "crossfade"])
def test_pause_boundaries_retain_speech_run_combination(method):
    processor = Ming30TTSProcessor(FakeAdapter(), {"task_mode": tasks.TASK_TTS})
    speech = [
        {"waveform": torch.ones(1, 8820), "text": "a"},
        {"waveform": torch.ones(1, 8820), "text": "b"},
    ]
    pause = {"waveform": torch.zeros(1, 44100), "text": "[pause:1s]",
             "explicit_pause": True}
    expected_speech = processor.combine_audio_segments(speech, method=method, silence_ms=250)
    result, info = processor.combine_audio_segments(
        speech + [pause, speech[0]], method=method, silence_ms=250, return_info=True,
    )
    assert torch.equal(result, torch.cat([
        expected_speech, pause["waveform"], speech[0]["waveform"],
    ], dim=-1))
    assert info["timing_unit"] == "groups"
    assert info["total_chunks"] == 3
    assert info["original_chunk_count"] == 4
    assert [group["original_chunk_count"] for group in info["group_details"]] == [2, 1, 1]
    speech_info = info["group_details"][0]["combination_info"]
    assert speech_info["total_chunks"] == 2
    assert speech_info["method_used"] == ("crossfade" if method == "auto" else method)
    assert info["group_details"][1]["explicit_pause"] is True
    report = processor_module.ChunkTimingHelper.enhance_generation_info("Generated", info)
    assert "pause-safe groups" in report
    assert f"Speech group (2 chunks, {speech_info['method_used']})" in report


def test_stretch_to_fit_exports_fitted_subtitle_windows():
    processor = object.__new__(Ming30TTSSRTProcessor)
    subtitles = [
        SimpleNamespace(start_time=0.0, end_time=1.0, duration=1.0, text="Hello", sequence=1),
        SimpleNamespace(start_time=2.0, end_time=3.0, duration=1.0, text="world", sequence=2),
    ]
    adjustments = [
        {"natural_duration": 2.0, "sequence": 1},
        {"natural_duration": 0.5, "sequence": 2},
    ]
    output = processor._generate_adjusted_srt_string(subtitles, adjustments, "stretch_to_fit")
    assert "00:00:00,000 --> 00:00:01,000" in output
    assert "00:00:02,000 --> 00:00:03,000" in output
    assert adjustments[0]["natural_duration"] == 2.0  # Diagnostics retain natural duration.
    unmodified = processor._generate_adjusted_srt_string(subtitles, adjustments, "pad_with_silence")
    assert "00:00:00,000 --> 00:00:02,000" in unmodified


def test_srt_interrupt_check_raises(monkeypatch):
    import comfy.model_management as model_management

    processor = object.__new__(Ming30TTSSRTProcessor)
    monkeypatch.setattr(model_management, "interrupt_processing", True, raising=False)
    with pytest.raises(InterruptedError, match="SRT"):
        processor._check_interrupt()
