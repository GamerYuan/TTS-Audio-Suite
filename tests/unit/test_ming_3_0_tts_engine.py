"""Lifecycle/import tests for the Ming 3.0 TTS engine wrapper."""

import sys
import textwrap
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from engines.ming_3_0_tts.ming_3_0_tts_downloader import Ming30TTSDownloader
from engines.ming_3_0_tts.ming_3_0_tts_engine import Ming30TTSEngine


FAKE_MING_TTS = textwrap.dedent(
    """
    import torch

    CREATED = []

    class MingTTS:
        def __init__(self, model_path, device="cuda:0", text_normalization=True,
                     cache_prompts=True):
            self.model_path = model_path
            self.device = device
            self.text_normalization = text_normalization
            self.cache_prompts = cache_prompts
            self.sample_rate = 44100
            self.model = torch.nn.Linear(1, 1)
            self.calls = []
            CREATED.append(self)

        def synthesize(self, **kwargs):
            self.calls.append(dict(kwargs, initial_seed=torch.initial_seed()))
            return torch.zeros(1, 4410)
    """
)


def make_source(root: Path) -> Path:
    source = root / "upstream"
    source.mkdir(parents=True)
    (source / "ming_tts.py").write_text(FAKE_MING_TTS, encoding="utf-8")
    for name in (
        "modeling_bailingmm.py",
        "tts_tasks.py",
        "prompt_contract.py",
        "spkemb_extractor.py",
        "configuration_bailingmm.py",
    ):
        (source / name).write_text("# test stub\n", encoding="utf-8")
    for directory, filename in (
        ("frontend", "toolkit.py"),
        ("frontend", "text_segment_cut.py"),
        ("audio_tokenizer", "audio_encoder.py"),
        ("audio_tokenizer", "modeling_audio_vae.py"),
        ("fm", "dit.py"),
    ):
        (source / directory).mkdir(exist_ok=True)
        (source / directory / filename).write_text("# test stub\n", encoding="utf-8")
    return source


def make_checkpoint(root: Path) -> Path:
    checkpoint = root / "Ming-3.0-tts-0.5B"
    checkpoint.mkdir(parents=True)
    for name in Ming30TTSDownloader.REQUIRED_FILES:
        (checkpoint / name).write_bytes(b"x")
    return checkpoint


def make_engine(monkeypatch, tmp_path):
    source = make_source(tmp_path)
    checkpoint = make_checkpoint(tmp_path)
    import engines.ming_3_0_tts.ming_3_0_tts_downloader as ming_downloader

    monkeypatch.setattr(ming_downloader, "allowed_path", lambda path: str(path))
    monkeypatch.setattr(Ming30TTSEngine, "_check_runtime_packages", lambda self: None)
    monkeypatch.setattr(Ming30TTSEngine, "_check_text_normalizer", lambda self: None)
    monkeypatch.setattr(Ming30TTSEngine, "_ensure_runtime_device", lambda self: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    engine = Ming30TTSEngine(
        model_name=Ming30TTSEngine.MODEL_NAME,
        device="cuda",
        upstream_path=str(source),
        text_normalization=False,
        cache_prompts=True,
        model_dir=str(checkpoint),
    )
    return engine


def test_engine_imports_official_source_and_calls_synthesize(monkeypatch, tmp_path):
    engine = make_engine(monkeypatch, tmp_path)
    audio = engine.generate(
        text="Hello world",
        task="TTS",
        language_id="English",
        prompt_wav_path=None,
        prompt_text=None,
        cfg=2.0,
        sigma=0.25,
        temperature=0.0,
        max_length=24,
        max_decode_steps=1000,
        seed=5,
    )

    assert audio["sample_rate"] == 44100
    assert audio["audio"].shape == (1, 4410)
    runtime = engine._runtime
    assert runtime.model_path.endswith("Ming-3.0-tts-0.5B")
    assert runtime.cache_prompts is True
    assert runtime.text_normalization is False
    call = runtime.calls[0]
    assert call["task"] == "TTS"
    assert call["text"] == "Hello world"
    assert call["language_id"] == "English"
    assert call["raw_text"] is False
    assert call["trim_silence"] is True
    assert call["instruction_schema"] is None
    assert call["output_path"] is None


def test_engine_seeds_torch_for_reproducible_generation(monkeypatch, tmp_path):
    engine = make_engine(monkeypatch, tmp_path)
    engine.generate(text="a", task="TTS", seed=1234)
    engine.generate(text="b", task="TTS", seed=1234)
    assert engine._runtime.calls[0]["initial_seed"] == 1234
    assert engine._runtime.calls[1]["initial_seed"] == 1234


def test_reference_seed_is_independent_of_prompt_cache_history(monkeypatch, tmp_path):
    engine = make_engine(monkeypatch, tmp_path)
    engine._ensure_runtime_loaded()
    runtime = engine._runtime
    runtime._prompt_cache = {}

    def stochastic_synthesize(**kwargs):
        path = kwargs["prompt_wav_path"]
        if path not in runtime._prompt_cache:
            runtime._prompt_cache[path] = torch.rand(1)
        return torch.rand(1, 16) + runtime._prompt_cache[path]

    runtime.synthesize = stochastic_synthesize
    request = {"text": "a", "task": "TTS", "prompt_wav_path": "ref.wav"}
    cold = engine.generate(**request, seed=123)["audio"]
    warm = engine.generate(**request, seed=123)["audio"]
    engine.generate(**request, seed=42)
    after_other_seed = engine.generate(**request, seed=123)["audio"]
    runtime._prompt_cache.clear()  # Same cache state as a runtime reload.
    after_clear = engine.generate(**request, seed=123)["audio"]
    assert torch.equal(cold, warm)
    assert torch.equal(cold, after_other_seed)
    assert torch.equal(cold, after_clear)


def test_engine_unload_drops_runtime(monkeypatch, tmp_path):
    engine = make_engine(monkeypatch, tmp_path)
    engine.generate(text="a", task="TTS", seed=1)
    assert engine._runtime is not None
    engine.unload_runtime()
    assert engine._runtime is None
    assert engine.sample_rate == 44100


def test_cpu_offload_preserves_cuda_intent(monkeypatch, tmp_path):
    engine = make_engine(monkeypatch, tmp_path)
    engine.generate(text="a", task="TTS", seed=1)
    engine.to("cpu")
    assert engine.device == "cpu"
    assert engine._intended_device == "cuda"
    assert engine._current_runtime_device().startswith("cpu")


def test_engine_missing_source_is_actionable(monkeypatch, tmp_path):
    checkpoint = make_checkpoint(tmp_path)
    import engines.ming_3_0_tts.ming_3_0_tts_downloader as ming_downloader

    monkeypatch.setattr(ming_downloader, "allowed_path", lambda path: str(path))
    monkeypatch.setattr(Ming30TTSEngine, "_check_runtime_packages", lambda self: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    engine = Ming30TTSEngine(
        device="cuda",
        upstream_path=str(tmp_path / "missing"),
        model_dir=str(checkpoint),
    )
    with pytest.raises(FileNotFoundError, match="upstream source folder does not exist"):
        engine._ensure_runtime_loaded()


def test_engine_rejects_cpu_execution(monkeypatch, tmp_path):
    checkpoint = make_checkpoint(tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    engine = Ming30TTSEngine(device="cpu", model_dir=str(checkpoint))
    with pytest.raises(RuntimeError, match="requires an NVIDIA CUDA GPU"):
        engine._ensure_runtime_loaded()


def test_source_import_restores_shadowed_modules(monkeypatch, tmp_path):
    source = make_source(tmp_path)
    checkpoint = make_checkpoint(tmp_path)
    import engines.ming_3_0_tts.ming_3_0_tts_downloader as ming_downloader

    monkeypatch.setattr(ming_downloader, "allowed_path", lambda path: str(path))
    shadow = type(sys)("frontend")
    shadow.__file__ = str(tmp_path / "other" / "frontend" / "__init__.py")
    sys.modules["frontend"] = shadow
    monkeypatch.setattr(Ming30TTSEngine, "_check_runtime_packages", lambda self: None)
    monkeypatch.setattr(Ming30TTSEngine, "_check_text_normalizer", lambda self: None)
    monkeypatch.setattr(Ming30TTSEngine, "_ensure_runtime_device", lambda self: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    try:
        engine = Ming30TTSEngine(
            device="cuda", upstream_path=str(source), model_dir=str(checkpoint)
        )
        engine._ensure_runtime_loaded()
        assert sys.modules["frontend"] is shadow
    finally:
        sys.modules.pop("ming_tts", None)
        sys.modules.pop("frontend", None)
