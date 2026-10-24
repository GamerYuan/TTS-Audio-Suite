"""Adapter from the unified TTS processors to the official Ming 3.0 TTS runtime."""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tempfile
from typing import Any, Dict, Optional, Tuple

import folder_paths
import soundfile as sf
import torch

current_dir = os.path.dirname(__file__)
engines_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(engines_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from engines.ming_3_0_tts import tasks
from engines.ming_3_0_tts.languages import format_ming_language, normalize_ming_language
from utils.audio.audio_hash import generate_stable_audio_component
from utils.audio.cache import get_audio_cache
from utils.models.factory_config import ModelLoadConfig
from utils.voice.reference import effective_voice_audio


class Ming30TTSEngineAdapter:
    """Translate suite processor calls into one official Ming TTS request."""

    SAMPLE_RATE = 44100

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config.copy() if config else {}
        self.audio_cache = get_audio_cache()
        self._last_config: Optional[ModelLoadConfig] = None
        self._load_signature = None
        self._last_reference_note = None

    @property
    def sample_rate(self) -> int:
        # The official checkpoint reports 44100 in config.json. Do not touch the
        # unified model cache just to answer a metadata query.
        return self.SAMPLE_RATE

    def update_config(self, new_config: Dict[str, Any]):
        self.config = new_config.copy() if new_config else {}

    def _build_load_signature(self) -> Tuple[Any, ...]:
        return (
            self.config.get("model", "Ming-3.0-tts-0.5B"),
            self.config.get("device", "auto"),
            self.config.get("upstream_path", "auto"),
            bool(self.config.get("text_normalization", False)),
            bool(self.config.get("cache_prompts", True)),
        )

    def load_model(
        self,
        model: str,
        device: str = "auto",
        upstream_path: str = "auto",
        text_normalization: bool = False,
        cache_prompts: bool = True,
    ):
        from utils.models.unified_model_interface import unified_model_interface

        load_config = ModelLoadConfig(
            engine_name="ming_3_0_tts",
            model_type="tts",
            model_name=model,
            model_path=model,
            device=device,
            additional_params={
                "upstream_path": upstream_path,
                "text_normalization": bool(text_normalization),
                "cache_prompts": bool(cache_prompts),
            },
        )
        self._last_config = load_config
        unified_model_interface.load_model(load_config)

    def _ensure_model_loaded(self):
        signature = self._build_load_signature()
        if signature == self._load_signature and self._last_config is not None:
            return
        self.load_model(
            model=self.config.get("model", "Ming-3.0-tts-0.5B"),
            device=self.config.get("device", "auto"),
            upstream_path=self.config.get("upstream_path", "auto"),
            text_normalization=self.config.get("text_normalization", False),
            cache_prompts=self.config.get("cache_prompts", True),
        )
        self._load_signature = signature

    def _get_engine(self):
        if self._last_config is None:
            self._ensure_model_loaded()
        from utils.models.unified_model_interface import unified_model_interface

        return unified_model_interface.load_model(self._last_config)

    # -- voice reference -------------------------------------------------
    @staticmethod
    def _reference_prompt_text(voice_ref: Optional[Dict[str, Any]]) -> str:
        if not isinstance(voice_ref, dict):
            return ""
        return str(
            voice_ref.get("reference_text")
            or voice_ref.get("prompt_text")
            or voice_ref.get("text")
            or ""
        ).strip()

    def _stable_prompt_path_from_tensor(self, waveform: torch.Tensor, sample_rate: int) -> str:
        audio = waveform.detach().float().cpu()
        if audio.dim() == 3:
            audio = audio.squeeze(0)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        if audio.dim() != 2:
            raise ValueError(f"Ming 3.0 TTS reference audio must be 1D/2D/3D, got {audio.shape}")

        digest = hashlib.sha256()
        digest.update(str(sample_rate).encode("ascii"))
        digest.update(str(tuple(audio.shape)).encode("ascii"))
        digest.update(audio.contiguous().numpy().tobytes())
        prompt_dir = os.path.join(folder_paths.get_temp_directory(), "tts_audio_suite_ming_prompts")
        os.makedirs(prompt_dir, exist_ok=True)
        target = os.path.join(prompt_dir, f"{digest.hexdigest()}.wav")
        if not os.path.isfile(target) or os.path.getsize(target) <= 0:
            handle, temp_target = tempfile.mkstemp(prefix="ming_prompt_", suffix=".wav", dir=prompt_dir)
            os.close(handle)
            try:
                samples = audio.transpose(0, 1).contiguous().numpy()
                sf.write(temp_target, samples, int(sample_rate), subtype="FLOAT")
                os.replace(temp_target, target)
            finally:
                if os.path.exists(temp_target):
                    os.remove(temp_target)
        return target

    def _stable_prompt_path_from_file(self, source_path: str) -> str:
        digest = hashlib.sha256()
        with open(source_path, "rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        suffix = os.path.splitext(source_path)[1].lower()
        if not suffix or len(suffix) > 8 or not suffix[1:].isalnum():
            suffix = ".wav"
        prompt_dir = os.path.join(folder_paths.get_temp_directory(), "tts_audio_suite_ming_prompts")
        os.makedirs(prompt_dir, exist_ok=True)
        target = os.path.join(prompt_dir, f"{digest.hexdigest()}{suffix}")
        if not os.path.isfile(target) or os.path.getsize(target) <= 0:
            handle, temp_target = tempfile.mkstemp(prefix="ming_prompt_", suffix=suffix, dir=prompt_dir)
            os.close(handle)
            try:
                shutil.copyfile(source_path, temp_target)
                os.replace(temp_target, target)
            finally:
                if os.path.exists(temp_target):
                    os.remove(temp_target)
        return target

    def _resolve_voice_reference(
        self,
        voice_ref: Optional[Dict[str, Any]],
        task_mode: str,
    ) -> Dict[str, Any]:
        """Return prompt path, transcript, cache component and speaker flags."""
        if not tasks.uses_reference_audio(task_mode):
            return {
                "prompt_wav_path": None,
                "prompt_text": "",
                "audio_component": "default_voice",
                "has_reference": False,
                "use_spk_emb": False,
                "use_zero_spk_emb": False,
            }

        reference_audio = effective_voice_audio(voice_ref)
        reference_text = self._reference_prompt_text(voice_ref)
        if reference_audio is None:
            if tasks.requires_reference_audio(task_mode):
                raise ValueError(
                    f"Ming {task_mode} requires a reference voice. Connect a narrator or "
                    "character voice with audio; these tasks condition delivery on it."
                )
            return {
                "prompt_wav_path": None,
                "prompt_text": "",
                "audio_component": "default_voice",
                "has_reference": False,
                "use_spk_emb": False,
                "use_zero_spk_emb": False,
            }

        if isinstance(reference_audio, str):
            prompt_path = self._stable_prompt_path_from_file(reference_audio)
            audio_component = generate_stable_audio_component(audio_file_path=reference_audio)
        elif isinstance(reference_audio, dict) and "waveform" in reference_audio:
            prompt_path = self._stable_prompt_path_from_tensor(
                reference_audio["waveform"],
                int(reference_audio.get("sample_rate", self.SAMPLE_RATE)),
            )
            audio_component = generate_stable_audio_component(reference_audio=reference_audio)
        elif torch.is_tensor(reference_audio):
            sample_rate = int(
                voice_ref.get("sample_rate", self.SAMPLE_RATE)
                if isinstance(voice_ref, dict)
                else self.SAMPLE_RATE
            )
            prompt_path = self._stable_prompt_path_from_tensor(reference_audio, sample_rate)
            audio_component = generate_stable_audio_component(
                reference_audio={"waveform": reference_audio, "sample_rate": sample_rate}
            )
        else:
            raise TypeError(
                f"Unsupported Ming 3.0 TTS voice reference type: {type(reference_audio)}"
            )

        # Only TTS/MULTILINGUAL accept a reference transcript. The official
        # control tasks take speaker conditioning only, and FREEFORM does not
        # accept a transcript either.
        prompt_text = reference_text if tasks.accepts_reference_transcript(task_mode) else ""
        return {
            "prompt_wav_path": prompt_path,
            "prompt_text": prompt_text,
            "audio_component": audio_component,
            "has_reference": True,
            "use_spk_emb": True,
            "use_zero_spk_emb": False,
        }

    def _log_reference_mode(
        self,
        task_mode: str,
        reference: Dict[str, Any],
        character_name: Optional[str],
    ) -> None:
        if reference["has_reference"]:
            if tasks.accepts_reference_transcript(task_mode):
                mode = (
                    "acoustic prompt + speaker embedding"
                    if reference["prompt_text"]
                    else "speaker embedding only (no reference transcript)"
                )
            else:
                mode = "speaker embedding only (task ignores reference transcript)"
        elif tasks.uses_reference_audio(task_mode):
            mode = "zero speaker embedding (no reference voice)"
        else:
            mode = "no reference audio (native task)"
        note = f"{task_mode}: {mode}"
        if note != self._last_reference_note:
            print(f"🎙️ Ming 3.0 TTS reference: {note} for '{character_name or 'narrator'}'")
            self._last_reference_note = note

    # -- generation ------------------------------------------------------
    def generate_single(
        self,
        text: str,
        voice_ref: Optional[Dict[str, Any]],
        seed: int = 0,
        enable_audio_cache: bool = True,
        character_name: Optional[str] = None,
    ) -> torch.Tensor:
        stripped = (text or "").strip()
        if not stripped:
            return torch.zeros(1, 0, dtype=torch.float32)

        self._ensure_model_loaded()
        engine = self._get_engine()

        task_mode = str(self.config.get("task_mode", tasks.TASK_TTS))
        official_task = tasks.resolve_official_task(task_mode)
        language_id = tasks.validate_task_language(task_mode, self.config.get("language"))
        controls = tasks.resolve_controls(task_mode, self.config)
        instruction = tasks.resolve_instruction(task_mode, self.config)
        reference = self._resolve_voice_reference(voice_ref, task_mode)
        self._log_reference_mode(task_mode, reference, character_name)

        cfg = float(self.config.get("cfg", 2.0))
        sigma = float(self.config.get("sigma", 0.25))
        temperature = float(self.config.get("temperature", 0.0))
        max_length = int(self.config.get("max_length", 24))
        max_decode_steps = int(self.config.get("max_decode_steps", 1000))
        model_name = self.config.get("model", "Ming-3.0-tts-0.5B")

        cache_key = None
        if enable_audio_cache:
            cache_key = self.audio_cache.generate_cache_key(
                "ming_3_0_tts",
                text=stripped,
                audio_component=reference["audio_component"],
                prompt_text=reference["prompt_text"],
                model=model_name,
                task_mode=task_mode,
                official_task=official_task,
                language=language_id or "none",
                controls=str(sorted((controls or {}).items())),
                instruction=instruction or "",
                cfg=cfg,
                sigma=sigma,
                temperature=temperature,
                max_length=max_length,
                max_decode_steps=max_decode_steps,
                text_normalization=bool(self.config.get("text_normalization", False)),
                cache_prompts=bool(self.config.get("cache_prompts", True)),
                upstream_path=self.config.get("upstream_path", "auto"),
                use_spk_emb=reference["use_spk_emb"],
                use_zero_spk_emb=reference["use_zero_spk_emb"],
                seed=int(seed or 0),
                device=self.config.get("device", "auto"),
                character=character_name or "narrator",
            )
            cached_audio = self.audio_cache.get_cached_audio(cache_key)
            if cached_audio:
                print(
                    f"💾 Using cached Ming 3.0 TTS audio for "
                    f"'{character_name or 'narrator'}': '{stripped[:30]}...'"
                )
                return cached_audio[0]

        result = engine.generate(
            text=stripped,
            task=official_task,
            language_id=language_id,
            instruction=instruction,
            controls=controls,
            prompt_wav_path=reference["prompt_wav_path"],
            prompt_text=reference["prompt_text"] or None,
            use_spk_emb=reference["use_spk_emb"],
            use_zero_spk_emb=reference["use_zero_spk_emb"],
            cfg=cfg,
            sigma=sigma,
            temperature=temperature,
            max_length=max_length,
            max_decode_steps=max_decode_steps,
            seed=int(seed or 0),
        )

        audio_tensor = result["audio"]
        if not isinstance(audio_tensor, torch.Tensor):
            audio_tensor = torch.as_tensor(audio_tensor, dtype=torch.float32)
        audio_tensor = audio_tensor.detach().float().cpu()
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        if enable_audio_cache and cache_key:
            duration = self.audio_cache._calculate_duration(audio_tensor, "ming_3_0_tts")
            self.audio_cache.cache_audio(cache_key, audio_tensor, duration)

        return audio_tensor

    @staticmethod
    def describe_config(config: Dict[str, Any]) -> str:
        """One-line settings summary shared by the adapter and processors."""
        task_mode = str(config.get("task_mode", tasks.TASK_TTS))
        language = format_ming_language(config.get("language")) if tasks.accepts_language(task_mode) else "n/a"
        return (
            f"{tasks.task_summary(task_mode, config, language)} | "
            f"cfg={float(config.get('cfg', 2.0)):.3g} | "
            f"sigma={float(config.get('sigma', 0.25)):.3g} | "
            f"temperature={float(config.get('temperature', 0.0)):.3g} | "
            f"max_length={int(config.get('max_length', 24))} | "
            f"max_decode_steps={int(config.get('max_decode_steps', 1000))} | "
            f"text_normalization={bool(config.get('text_normalization', False))}"
        )

    def normalize_language(self, language: Optional[str]) -> Optional[str]:
        return normalize_ming_language(language)
