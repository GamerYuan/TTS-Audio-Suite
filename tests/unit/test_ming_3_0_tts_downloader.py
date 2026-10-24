"""Downloader/path tests for Ming 3.0 TTS (no network)."""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from engines.ming_3_0_tts.ming_3_0_tts_downloader import (
    Ming30TTSDownloader,
    MODEL_NAME,
)


def make_complete_checkpoint(root: Path, name: str = MODEL_NAME) -> Path:
    checkpoint = root / name
    checkpoint.mkdir(parents=True)
    for required in Ming30TTSDownloader.REQUIRED_FILES:
        path = checkpoint / required
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"data")
    return checkpoint


def test_complete_checkpoint_and_list_local_models(monkeypatch, tmp_path):
    engine_root = tmp_path / "ming_3_0_tts"
    make_complete_checkpoint(engine_root)
    (engine_root / "upstream").mkdir()
    import engines.ming_3_0_tts.ming_3_0_tts_downloader as downloader_module
    import utils.security.path_access as path_access

    monkeypatch.setattr(path_access, "allowed_path", lambda path: str(path))
    monkeypatch.setattr(downloader_module, "get_all_tts_model_paths", lambda model_type="TTS": [str(tmp_path)])
    downloader = Ming30TTSDownloader(base_path=str(tmp_path))

    assert downloader._is_model_complete(str(engine_root / MODEL_NAME))
    assert downloader.list_local_models() == [MODEL_NAME]
    resolved = downloader.resolve_model_path(f"local:{MODEL_NAME}")
    assert Path(resolved) == engine_root / MODEL_NAME


def test_missing_checkpoint_files_are_reported(tmp_path):
    checkpoint = tmp_path / MODEL_NAME
    checkpoint.mkdir()
    (checkpoint / "config.json").write_bytes(b"x")
    downloader = Ming30TTSDownloader(base_path=str(tmp_path))
    assert not downloader._is_model_complete(str(checkpoint), quiet=True)


def test_download_uses_allow_patterns_and_local_dir(monkeypatch, tmp_path):
    calls = {}

    def fake_snapshot_download(**kwargs):
        calls.update(kwargs)
        make_complete_checkpoint(Path(kwargs["local_dir"]).parent)

    import engines.ming_3_0_tts.ming_3_0_tts_downloader as downloader_module

    monkeypatch.setattr(downloader_module, "snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(downloader_module, "allowed_path", lambda path: str(path))
    downloader = Ming30TTSDownloader(base_path=str(tmp_path))

    resolved = downloader.download_model(MODEL_NAME)

    assert calls["repo_id"] == "inclusionAI/Ming-3.0-tts-0.5B"
    assert "model.safetensors" in calls["allow_patterns"]
    assert calls["local_dir_use_symlinks"] is False
    assert Path(resolved).is_dir()
    assert all((Path(resolved) / name).is_file() for name in Ming30TTSDownloader.REQUIRED_FILES)


def test_upstream_source_detection_and_hint(monkeypatch, tmp_path):
    source = tmp_path / "ming_3_0_tts" / "upstream"
    source.mkdir(parents=True)
    for name in Ming30TTSDownloader.UPSTREAM_FILES:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# source\n", encoding="utf-8")

    import engines.ming_3_0_tts.ming_3_0_tts_downloader as downloader_module

    monkeypatch.setattr(downloader_module, "get_all_tts_model_paths", lambda model_type="TTS": [str(tmp_path)])
    downloader = Ming30TTSDownloader(base_path=str(tmp_path))

    assert downloader._is_upstream_source(str(source))
    assert downloader.resolve_upstream_source("auto") == str(source)
    assert "ming_tts.py" in downloader.upstream_setup_hint()
    assert "github.com/inclusionAI/Ming-3.0-tts" in downloader.upstream_setup_hint()


def test_upstream_env_var_override(monkeypatch, tmp_path):
    source = tmp_path / "custom-source"
    source.mkdir()
    for name in Ming30TTSDownloader.UPSTREAM_FILES:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# source\n", encoding="utf-8")

    import engines.ming_3_0_tts.ming_3_0_tts_downloader as downloader_module

    monkeypatch.setattr(downloader_module, "allowed_path", lambda path: str(path))
    monkeypatch.setattr(downloader_module, "get_all_tts_model_paths", lambda model_type="TTS": [str(tmp_path)])
    monkeypatch.setenv("MING_3_0_TTS_SOURCE", str(source))
    downloader = Ming30TTSDownloader(base_path=str(tmp_path))
    assert downloader.resolve_upstream_source("auto") == str(source)


def test_incomplete_explicit_source_raises(monkeypatch, tmp_path):
    source = tmp_path / "incomplete"
    source.mkdir()
    import engines.ming_3_0_tts.ming_3_0_tts_downloader as downloader_module

    monkeypatch.setattr(downloader_module, "allowed_path", lambda path: str(path))
    downloader = Ming30TTSDownloader(base_path=str(tmp_path))
    with pytest.raises(FileNotFoundError, match="incomplete"):
        downloader.resolve_upstream_source(str(source))
