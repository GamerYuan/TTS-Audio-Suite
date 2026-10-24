"""Ming 3.0 TTS engine wrapper.

Wraps the official ``ming_tts.MingTTS`` runtime behind ComfyUI lifecycle hooks.
The upstream source is imported lazily from an explicit checkout; its pinned
requirements (Transformers 4.46.3, torch 2.6) are intentionally not installed
or downgraded by this module.
"""

from __future__ import annotations

import gc
import importlib
import importlib.util
import os
import sys
import threading
from contextlib import contextmanager
from typing import Any, Dict, Optional

import torch

current_dir = os.path.dirname(__file__)
engines_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(engines_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from engines.ming_3_0_tts.ming_3_0_tts_downloader import Ming30TTSDownloader
from utils.device import resolve_torch_device

_IMPORT_LOCK = threading.RLock()

# Top-level modules provided by the official GitHub checkout. They use
# absolute imports, so they must be visible on sys.path during import.
_SOURCE_TOP_LEVELS = (
    "ming_tts",
    "modeling_bailingmm",
    "tts_tasks",
    "prompt_contract",
    "configuration_bailingmm",
    "spkemb_extractor",
    "frontend",
    "audio_tokenizer",
    "fm",
)

_REQUIRED_RUNTIME_PACKAGES = (
    "transformers",
    "torchaudio",
    "x_transformers",
    "diffusers",
    "onnxruntime",
    "inflect",
    "einops",
    "soundfile",
    "safetensors",
)


class Ming30TTSEngine:
    """Thin wrapper around the official MingTTS runtime."""

    SAMPLE_RATE = 44100
    MODEL_NAME = "Ming-3.0-tts-0.5B"

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: str = "auto",
        upstream_path: str = "auto",
        text_normalization: bool = False,
        cache_prompts: bool = True,
        model_dir: Optional[str] = None,
    ):
        self.model_name = model_name or self.MODEL_NAME
        self.device = resolve_torch_device(device)
        self._intended_device = self.device
        self.upstream_path = str(upstream_path or "auto")
        self.text_normalization = bool(text_normalization)
        self.cache_prompts = bool(cache_prompts)
        self.model_dir = model_dir or Ming30TTSDownloader().resolve_model_path(self.model_name)
        self.sample_rate = self.SAMPLE_RATE
        self._runtime = None
        self._runtime_class = None
        self._runtime_source: Optional[str] = None

    # -- import helpers -------------------------------------------------
    @staticmethod
    def _module_under_source(module: Any, source_dir: str) -> bool:
        module_file = getattr(module, "__file__", None)
        if not module_file:
            return False
        try:
            module_path = os.path.abspath(module_file)
            return os.path.commonpath([module_path, source_dir]) == source_dir
        except (TypeError, ValueError, OSError):
            return False

    @staticmethod
    def _top_level_name(module_name: str) -> str:
        return str(module_name).split(".", 1)[0]

    @contextmanager
    def _upstream_import_context(self, source_dir: str):
        """Temporarily expose the upstream checkout on sys.path.

        Any same-named modules from other packages are set aside and restored
        afterwards so importing Ming cannot poison another custom node.
        """
        source_dir = os.path.abspath(source_dir)
        with _IMPORT_LOCK:
            shadowed = {}
            for module_name, module in list(sys.modules.items()):
                if self._top_level_name(module_name) not in _SOURCE_TOP_LEVELS:
                    continue
                if not self._module_under_source(module, source_dir):
                    shadowed[module_name] = module
                    del sys.modules[module_name]

            sys.path.insert(0, source_dir)
            importlib.invalidate_caches()
            try:
                yield source_dir
            finally:
                try:
                    sys.path.remove(source_dir)
                except ValueError:
                    pass
                for module_name, module in list(sys.modules.items()):
                    if self._top_level_name(module_name) not in _SOURCE_TOP_LEVELS:
                        continue
                    if self._module_under_source(module, source_dir):
                        del sys.modules[module_name]
                for module_name, module in shadowed.items():
                    sys.modules[module_name] = module

    def _check_runtime_packages(self) -> None:
        missing = []
        for package_name in _REQUIRED_RUNTIME_PACKAGES:
            try:
                if importlib.util.find_spec(package_name) is None:
                    missing.append(package_name)
            except (ImportError, ValueError):
                missing.append(package_name)
        if missing:
            raise RuntimeError(
                "Ming 3.0 TTS is missing official runtime Python packages: "
                + ", ".join(sorted(missing))
                + ". The suite installer declares x-transformers, einops and inflect and "
                "installs diffusers/onnxruntime; re-run the suite installer or install only "
                "the missing package in the active ComfyUI environment. This engine does "
                "not install or downgrade torch/transformers."
            )

    def _check_text_normalizer(self) -> None:
        if not self.text_normalization:
            return
        if importlib.util.find_spec("pynini") is None:
            raise RuntimeError(
                "Ming 3.0 TTS text_normalization=True requires pynini, the official "
                "optional text-normalization dependency. Set text_normalization=False "
                "or install pynini separately in the active ComfyUI environment."
            )

    def _import_runtime_class(self, source_dir: str):
        try:
            with self._upstream_import_context(source_dir):
                import ming_tts as upstream_module
        except Exception as error:
            raise RuntimeError(
                "Failed to import the official Ming 3.0 TTS source. "
                f"Source: {source_dir}\n"
                f"Active Python: {sys.executable}\n"
                "The official pins are torch==2.6, torchaudio==2.6, transformers==4.46.3, "
                "x_transformers==2.9.2, diffusers==0.33; this suite runs it in the main "
                "environment without replacing those packages.\n"
                f"Import error: {type(error).__name__}: {error}"
            ) from error

        runtime_class = getattr(upstream_module, "MingTTS", None)
        if runtime_class is None:
            raise RuntimeError(
                f"{os.path.join(source_dir, 'ming_tts.py')} does not expose MingTTS; "
                "the source checkout is not the official Ming 3.0 TTS repository."
            )
        return runtime_class

    def _build_normalizer(self, source_dir: str):
        if not self.text_normalization:
            return None
        try:
            with self._upstream_import_context(source_dir):
                from frontend.talker_tn.talker_tn import TalkerTN

                return TalkerTN()
        except Exception as error:
            raise RuntimeError(
                "Ming 3.0 TTS text normalization is enabled, but the official "
                f"TalkerTN normalizer could not be loaded from {source_dir}. "
                "It requires pynini and the frontend/talker_tn FST files. "
                "Set text_normalization=False to run without it.\n"
                f"Error: {type(error).__name__}: {error}"
            ) from error

    # -- lifecycle ------------------------------------------------------
    def _ensure_runtime_loaded(self) -> None:
        if self._runtime is not None:
            return

        if not str(self._intended_device).startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError(
                "Ming 3.0 TTS requires an NVIDIA CUDA GPU. The official runtime loads "
                "bfloat16 weights and calls torch.autocast(device_type='cuda'); CPU/MPS/XPU "
                "execution is not supported by the upstream source."
            )

        self._check_runtime_packages()
        self._check_text_normalizer()

        source_dir = Ming30TTSDownloader().resolve_upstream_source(self.upstream_path)
        if source_dir is None:
            raise RuntimeError(
                "Ming 3.0 TTS upstream source was not found.\n"
                + Ming30TTSDownloader().upstream_setup_hint()
            )

        runtime_class = self._import_runtime_class(source_dir)
        normalizer = self._build_normalizer(source_dir)
        try:
            runtime = runtime_class(
                str(self.model_dir),
                device=str(self._intended_device),
                text_normalization=self.text_normalization,
                cache_prompts=self.cache_prompts,
            )
        except Exception as error:
            raise RuntimeError(
                "Failed to load the Ming 3.0 TTS checkpoint.\n"
                f"Checkpoint: {self.model_dir}\n"
                f"Source: {source_dir}\n"
                f"Active Python: {sys.executable}\n"
                f"Load error: {type(error).__name__}: {error}"
            ) from error

        if normalizer is not None:
            # Avoid the upstream lazy import after the source path is removed.
            runtime._normalizer = normalizer

        self._runtime = runtime
        self._runtime_class = runtime_class
        self._runtime_source = source_dir
        self.sample_rate = int(getattr(runtime, "sample_rate", self.SAMPLE_RATE))
        self.device = str(self._intended_device)
        self._ensure_runtime_device()

    def _current_runtime_device(self) -> Optional[str]:
        if self._runtime is None:
            return None
        try:
            first_param = next(self._runtime.model.parameters())
            return str(first_param.device)
        except Exception:
            return str(getattr(self._runtime, "device", self.device))

    @staticmethod
    def _devices_equivalent(current_device: Optional[str], target_device: Optional[str]) -> bool:
        if current_device == target_device:
            return True
        if not current_device or not target_device:
            return False
        current = str(current_device)
        target = str(target_device)
        if current.startswith("cuda") and target.startswith("cuda"):
            # resolve_torch_device returns "cuda"; torch reports "cuda:0".
            return current in {"cuda", "cuda:0"} and target in {"cuda", "cuda:0"}
        return False

    def _move_runtime_to(self, target_device: str) -> None:
        if self._runtime is None:
            return
        device_obj = torch.device(target_device)
        self._runtime.model = self._runtime.model.to(device_obj).eval()
        self._runtime.device = target_device

    def _ensure_runtime_device(self, target_device: Optional[str] = None) -> None:
        if self._runtime is None:
            return
        # ComfyUI may offload the model to CPU for Clear VRAM. Generation must
        # restore the intended CUDA device rather than trusting self.device,
        # because the official runtime only executes under CUDA autocast.
        target_device = target_device or self._intended_device
        if not str(target_device).startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError("Ming 3.0 TTS cannot generate without an available CUDA device.")
        current_device = self._current_runtime_device()
        if self._devices_equivalent(current_device, target_device):
            return
        print(f"🔄 Ming 3.0 TTS: moving runtime from {current_device} to {target_device}")
        self._move_runtime_to(target_device)

    def to(self, device):
        """ComfyUI unload/reload hook."""
        self.device = str(device) if isinstance(device, str) else str(torch.device(device))
        # Keep the intended device separate: a CPU offload is temporary and must
        # not permanently turn this CUDA-only engine into a CPU engine.
        if not str(self.device).startswith("cpu"):
            self._intended_device = self.device
        if self._runtime is not None:
            if str(self.device).startswith("cpu"):
                self._move_runtime_to("cpu")
            else:
                self._ensure_runtime_device()
        return self

    def unload_runtime(self) -> None:
        """Fully tear down the runtime so Clear VRAM releases weights and caches."""
        runtime = self._runtime
        if runtime is not None:
            try:
                runtime._prompt_cache.clear()
            except Exception:
                pass
            model = getattr(runtime, "model", None)
            try:
                if model is not None and hasattr(model, "to"):
                    model.to("cpu")
            except Exception:
                pass
            try:
                runtime.model = None
            except Exception:
                pass
            self._runtime = None
            del model
            del runtime

        self._runtime_class = None
        self._runtime_source = None
        self.sample_rate = self.SAMPLE_RATE
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass

    def unload(self) -> None:
        self.unload_runtime()

    # -- generation -----------------------------------------------------
    @staticmethod
    def _seed_runtime(seed: int) -> None:
        seed_value = int(seed or 0)
        torch.manual_seed(seed_value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_value)

    def generate(
        self,
        text: str,
        task: str,
        language_id: Optional[str] = None,
        instruction: Optional[str] = None,
        controls: Optional[Dict[str, Any]] = None,
        prompt_wav_path: Optional[str] = None,
        prompt_text: Optional[str] = None,
        use_spk_emb: bool = False,
        use_zero_spk_emb: bool = False,
        cfg: float = 2.0,
        sigma: float = 0.25,
        temperature: float = 0.0,
        max_length: int = 24,
        max_decode_steps: int = 1000,
        seed: int = 0,
    ) -> Dict[str, Any]:
        self._ensure_runtime_loaded()
        self._ensure_runtime_device()
        # Upstream prompt encoding samples the VAE posterior. Reusing a latent
        # from a previous request skips RNG draws (and may reuse another seed's
        # latent), so cached and uncached calls would otherwise diverge.
        prompt_cache = getattr(self._runtime, "_prompt_cache", None)
        if prompt_cache is not None:
            prompt_cache.clear()
        self._seed_runtime(seed)

        waveform = self._runtime.synthesize(
            task=task,
            text=text,
            language_id=language_id,
            instruction=instruction,
            controls=controls,
            instruction_schema=None,
            prompt_wav_path=prompt_wav_path,
            prompt_text=prompt_text,
            spk_emb_path=None,
            use_spk_emb=bool(use_spk_emb),
            use_zero_spk_emb=bool(use_zero_spk_emb),
            cfg=float(cfg),
            sigma=float(sigma),
            temperature=float(temperature),
            max_length=int(max_length),
            max_decode_steps=int(max_decode_steps),
            trim_silence=True,
            raw_text=False,
            output_path=None,
        )

        if not isinstance(waveform, torch.Tensor):
            waveform = torch.as_tensor(waveform, dtype=torch.float32)
        audio = waveform.detach().float().cpu()
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        elif audio.dim() == 3:
            audio = audio.squeeze(0)
        return {
            "audio": audio,
            "sample_rate": self.sample_rate,
        }
