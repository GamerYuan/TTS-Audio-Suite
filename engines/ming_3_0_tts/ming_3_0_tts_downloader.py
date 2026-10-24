"""Downloader and path resolver for Ming 3.0 TTS.

The Hugging Face checkpoint contains weights, tokenizer files and the campplus
speaker encoder, but not the executable upstream Python source. The downloader
only fetches the checkpoint into ComfyUI/models/TTS/ming_3_0_tts/; the official
source checkout is a separate, explicit user step because importing it executes
that source.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import folder_paths
from huggingface_hub import snapshot_download

from utils.hf_download_logging import quiet_hf_download_logs
from utils.models.extra_paths import get_all_tts_model_paths, get_preferred_download_path
from utils.security.path_access import allowed_path, validate_model_paths

MODEL_NAME = "Ming-3.0-tts-0.5B"
ENGINE_FOLDER = "ming_3_0_tts"
UPSTREAM_ENV_VAR = "MING_3_0_TTS_SOURCE"


class Ming30TTSDownloader:
    """Resolve the official checkpoint and the optional upstream source tree."""

    MODELS: Dict[str, Dict[str, str]] = {
        MODEL_NAME: {
            "repo_id": "inclusionAI/Ming-3.0-tts-0.5B",
            "description": "Official Ming 3.0 TTS 0.5B checkpoint (MIT code and weights)",
        },
    }

    REQUIRED_FILES = (
        "added_tokens.json",
        "campplus.onnx",
        "config.json",
        "merges.txt",
        "model.safetensors",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    )

    # Keep the checkpoint download free of report PDFs and README figures.
    ALLOW_PATTERNS = [
        "added_tokens.json",
        "campplus.onnx",
        "config.json",
        "merges.txt",
        "model.safetensors",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    ]
    IGNORE_PATTERNS = ["*.pdf", "figures/*", "README.md", "configuration.json"]

    UPSTREAM_FILES = (
        "ming_tts.py",
        "modeling_bailingmm.py",
        "tts_tasks.py",
        "prompt_contract.py",
        "spkemb_extractor.py",
        "configuration_bailingmm.py",
        "frontend/toolkit.py",
        "frontend/text_segment_cut.py",
        "audio_tokenizer/audio_encoder.py",
        "audio_tokenizer/modeling_audio_vae.py",
        "fm/dit.py",
    )

    def __init__(self, base_path: Optional[str] = None):
        if base_path is None:
            try:
                self.base_path = get_preferred_download_path(
                    model_type="TTS", engine_name=ENGINE_FOLDER
                )
            except Exception:
                self.base_path = os.path.join(folder_paths.models_dir, "TTS", ENGINE_FOLDER)
        else:
            self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)

    # -- checkpoint -----------------------------------------------------
    @validate_model_paths
    def resolve_model_path(self, model_identifier: str) -> str:
        if not model_identifier:
            model_identifier = MODEL_NAME

        if os.path.isabs(model_identifier) and os.path.exists(model_identifier):
            return model_identifier
        if os.path.isdir(model_identifier):
            return os.path.abspath(model_identifier)

        if str(model_identifier).startswith("local:"):
            local_name = str(model_identifier)[6:]
            resolved = self._find_local_model(local_name)
            if resolved:
                print(f"📁 Using local Ming 3.0 TTS model: {resolved}")
                return resolved
            raise FileNotFoundError(
                f"Local Ming 3.0 TTS model not found: {local_name}. "
                f"Searched configured TTS folders under {ENGINE_FOLDER}/."
            )

        if model_identifier in self.MODELS:
            return self.get_model_path(model_identifier)

        raise FileNotFoundError(
            f"Unknown Ming 3.0 TTS model identifier: {model_identifier}. "
            f"Available: {', '.join(self.MODELS)}"
        )

    def get_model_path(self, model_name: str = MODEL_NAME) -> str:
        # Reuse any complete installation in configured TTS folders, including
        # a manual direct folder, before downloading another copy.
        local = self._find_local_model(model_name)
        if local:
            return local
        return self.download_model(model_name, force=False)

    def download_model(self, model_name: str = MODEL_NAME, force: bool = False) -> str:
        if model_name not in self.MODELS:
            raise ValueError(f"Unknown Ming 3.0 TTS model: {model_name}")

        model_info = self.MODELS[model_name]
        repo_id = model_info["repo_id"]
        model_dir = os.path.join(self.base_path, model_name)

        print(f"\n{'=' * 60}")
        print("📦 Ming 3.0 TTS Model Download")
        print(f"{'=' * 60}")
        print(f"Model: {model_name}")
        print(f"Description: {model_info['description']}")
        print(f"Repository: {repo_id}")
        print(f"Target: {model_dir}")
        print(f"{'=' * 60}\n")

        try:
            with quiet_hf_download_logs():
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=allowed_path(model_dir),
                    allow_patterns=list(self.ALLOW_PATTERNS),
                    ignore_patterns=list(self.IGNORE_PATTERNS),
                    local_dir_use_symlinks=False,
                    resume_download=True,
                    force_download=force,
                )
        except Exception as error:
            raise RuntimeError(
                f"Failed to download {model_name} from {repo_id}: {error}"
            ) from error

        if not self._is_model_complete(model_dir):
            raise RuntimeError(f"Downloaded Ming 3.0 TTS model is incomplete: {model_dir}")

        print(f"✅ Ming 3.0 TTS model ready: {model_dir}")
        return model_dir

    def _is_model_complete(self, model_dir: str, quiet: bool = False) -> bool:
        if not os.path.isdir(model_dir):
            return False

        missing = [
            relative
            for relative in self.REQUIRED_FILES
            if not os.path.isfile(os.path.join(model_dir, relative))
            or os.path.getsize(os.path.join(model_dir, relative)) <= 0
        ]
        if missing:
            if not quiet:
                print(f"❌ Ming 3.0 TTS model incomplete. Missing {len(missing)} file(s):")
                for relative in missing:
                    print(f"   - {relative}")
            return False
        return True

    def _candidate_model_dirs(self, name: str) -> List[str]:
        candidates: List[str] = []
        base_folders = list(get_all_tts_model_paths("TTS"))
        if self.base_path not in base_folders:
            base_folders.insert(0, self.base_path)
        for base in base_folders:
            candidates.extend(
                [
                    os.path.join(base, ENGINE_FOLDER, name),
                    os.path.join(base, name),
                    os.path.join(base, ENGINE_FOLDER, "upstream", name),
                ]
            )
        # Case-insensitive fallback for the common 0.5b/0.5B spelling.
        deduped = []
        seen = set()
        for candidate in candidates:
            key = os.path.normcase(os.path.abspath(candidate))
            if key not in seen:
                seen.add(key)
                deduped.append(candidate)
        return deduped

    def _find_local_model(self, name: str) -> Optional[str]:
        for candidate in self._candidate_model_dirs(name):
            if os.path.isdir(candidate) and self._is_model_complete(candidate, quiet=True):
                return candidate
        return None

    def list_local_models(self) -> List[str]:
        """Return complete checkpoint folder names under configured TTS paths."""
        found: List[str] = []
        seen = set()
        for base in get_all_tts_model_paths("TTS"):
            roots = [os.path.join(base, ENGINE_FOLDER), base]
            for root in roots:
                if not os.path.isdir(root):
                    continue
                try:
                    names = sorted(os.listdir(root))
                except OSError:
                    continue
                for name in names:
                    candidate = os.path.join(root, name)
                    if not os.path.isdir(candidate) or not self._is_model_complete(candidate, quiet=True):
                        continue
                    key = name.lower()
                    if key not in seen:
                        seen.add(key)
                        found.append(name)
        return found

    # -- upstream source ------------------------------------------------
    def resolve_upstream_source(self, path_value: Optional[str] = "auto") -> Optional[str]:
        """Resolve the official GitHub source tree without importing it.

        ``path_value`` may be ``auto``/empty or an explicit folder inside
        ComfyUI's permitted data folders. The environment variable
        ``MING_3_0_TTS_SOURCE`` also works for advanced setups.
        """
        explicit = str(path_value or "").strip()
        if explicit and explicit.lower() != "auto":
            resolved = os.path.abspath(os.path.expanduser(allowed_path(explicit)))
            if not os.path.isdir(resolved):
                raise FileNotFoundError(
                    f"Ming 3.0 TTS upstream source folder does not exist: {resolved}\n"
                    + self.upstream_setup_hint()
                )
            if not self._is_upstream_source(resolved):
                missing = self._missing_upstream_files(resolved)
                raise FileNotFoundError(
                    f"Ming 3.0 TTS upstream source is incomplete at {resolved}. "
                    f"Missing: {', '.join(missing)}\n" + self.upstream_setup_hint()
                )
            return resolved

        env_path = os.environ.get(UPSTREAM_ENV_VAR, "").strip()
        if env_path:
            return self.resolve_upstream_source(env_path)

        for candidate in self.upstream_candidates():
            if self._is_upstream_source(candidate):
                return candidate
        return None

    def upstream_candidates(self) -> List[str]:
        candidates: List[str] = []
        base_folders = list(get_all_tts_model_paths("TTS"))
        if self.base_path not in base_folders:
            base_folders.insert(0, self.base_path)
        for base in base_folders:
            engine_root = os.path.join(base, ENGINE_FOLDER)
            candidates.extend(
                [
                    os.path.join(engine_root, "upstream"),
                    os.path.join(engine_root, "upstream", "Ming-3.0-tts"),
                    os.path.join(engine_root, "upstream", "Ming-3.0-tts-main"),
                    os.path.join(engine_root, "upstream", "Ming-3.0-tts-0.5B"),
                    os.path.join(engine_root, "source"),
                    os.path.join(engine_root, "Ming-3.0-tts"),
                ]
            )
        # Keep an optional development checkout inside the suite, but never
        # require it. Normal installations use models/TTS.
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        candidates.append(os.path.join(repo_root, "engines", ENGINE_FOLDER, "upstream"))

        deduped: List[str] = []
        seen = set()
        for candidate in candidates:
            key = os.path.normcase(os.path.abspath(candidate))
            if key not in seen:
                seen.add(key)
                deduped.append(candidate)
        return deduped

    def _missing_upstream_files(self, source_dir: str) -> List[str]:
        return [
            relative
            for relative in self.UPSTREAM_FILES
            if not os.path.isfile(os.path.join(source_dir, relative))
        ]

    def _is_upstream_source(self, source_dir: str) -> bool:
        return os.path.isdir(source_dir) and not self._missing_upstream_files(source_dir)

    def upstream_setup_hint(self) -> str:
        target = os.path.join(self.base_path, "upstream")
        return (
            "Ming 3.0 TTS needs the official GitHub source in addition to the Hugging Face "
            "checkpoint (the checkpoint alone is not executable).\n"
            f"Place it at: {target}\n"
            "  Option A: git clone https://github.com/inclusionAI/Ming-3.0-tts "
            f"\"{target}\"\n"
            "  Option B: download the repository ZIP and extract it so the folder contains "
            "ming_tts.py, modeling_bailingmm.py, tts_tasks.py and the frontend/ directory.\n"
            "The suite does not download or install that source automatically because "
            "importing it executes it. MIT-licensed upstream code and weights."
        )
