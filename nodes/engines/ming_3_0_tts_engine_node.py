"""Ming 3.0 TTS engine configuration node."""

from __future__ import annotations

import importlib.util
import os
import sys

current_dir = os.path.dirname(__file__)
nodes_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(nodes_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

base_node_path = os.path.join(nodes_dir, "base", "base_node.py")
base_spec = importlib.util.spec_from_file_location("base_node_module", base_node_path)
base_module = importlib.util.module_from_spec(base_spec)
sys.modules["base_node_module"] = base_module
base_spec.loader.exec_module(base_module)
BaseTTSNode = base_module.BaseTTSNode

from engines.ming_3_0_tts import tasks
from engines.ming_3_0_tts.languages import (
    LANGUAGE_OPTIONS,
    format_ming_language,
    normalize_ming_language,
)
from engines.ming_3_0_tts.ming_3_0_tts_downloader import Ming30TTSDownloader
from engines.ming_3_0_tts.voices import (
    IP_EMOTION_OPTIONS,
    IP_VOICE_BASES,
    resolve_ip_voice,
)


class Ming30TTSEngineNode(BaseTTSNode):
    """Ming 3.0 TTS engine configuration node."""

    MODEL_VARIANTS = ("Ming-3.0-tts-0.5B",)
    DEVICE_OPTIONS = ["auto", "cuda"]

    @classmethod
    def NAME(cls):
        return "⚙️ Ming 3.0 TTS Engine"

    @classmethod
    def _model_options(cls):
        options = list(cls.MODEL_VARIANTS)
        try:
            local_models = Ming30TTSDownloader().list_local_models()
        except Exception:
            local_models = []
        for name in local_models:
            local_option = f"local:{name}"
            if local_option not in options:
                options.append(local_option)
        return options

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (cls._model_options(), {
                    "default": cls.MODEL_VARIANTS[0],
                    "tooltip": (
                        "Official Ming 3.0 TTS 0.5B checkpoint.\n"
                        "Unprefixed choices auto-download into ComfyUI/models/TTS/ming_3_0_tts/.\n"
                        "local: entries are already present in configured TTS folders.\n"
                        "The checkpoint alone is not executable: the official GitHub source "
                        "must also be available (see upstream_path)."
                    ),
                }),
                "device": (cls.DEVICE_OPTIONS, {
                    "default": "auto",
                    "tooltip": (
                        "Ming 3.0 TTS runs bfloat16 on an NVIDIA CUDA GPU.\n"
                        "The official upstream source calls torch.autocast(device_type='cuda'), "
                        "so CPU/MPS/XPU execution is not supported."
                    ),
                }),
                "task_mode": (list(tasks.TASK_MODE_OPTIONS), {
                    "default": tasks.TASK_TTS,
                    "tooltip": (
                        "One official Ming task per run; controls do not combine across tasks.\n"
                        "• TTS (voice cloning): clone the reference voice; Chinese/English language control\n"
                        "• Multilingual TTS: German, Spanish, French, Korean, Japanese\n"
                        "• Basic attributes: speed, pitch, volume with a reference voice\n"
                        "• Emotion: one of seven official emotions with a reference voice\n"
                        "• Dialect: cantonese or chuanyu dialect with a reference voice\n"
                        "• Built-in voice (IP): exact catalog voice, no reference audio\n"
                        "• Voice design: describe a new voice, no reference audio\n"
                        "• Freeform instruction: natural-language delivery instruction"
                    ),
                }),
                "language": (list(LANGUAGE_OPTIONS), {
                    "default": "Auto",
                    "tooltip": (
                        "Official target language.\n"
                        "Used by TTS (voice cloning) and Multilingual TTS.\n"
                        "Multilingual TTS requires German/Spanish/French/Korean/Japanese.\n"
                        "Dialect, Basic, Emotion, IP, Voice design and Freeform do not use a "
                        "language_id in the official API."
                    ),
                }),
                "max_length": ("INT", {
                    "default": 24,
                    "min": 4,
                    "max": 256,
                    "step": 1,
                    "tooltip": (
                        "Official semantic fragment target length (roughly characters, split at "
                        "punctuation). Default 24. Shorter values increase fragment count; longer "
                        "values keep more text in one fragment. The suite does not bypass this "
                        "native splitter."
                    ),
                }),
                "max_decode_steps": ("INT", {
                    "default": 1000,
                    "min": 1,
                    "max": 2000,
                    "step": 1,
                    "tooltip": (
                        "Official decode-step cap per fragment. The runtime also caps decoding "
                        "to its 512-position KV cache budget."
                    ),
                }),
                "cfg": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.05,
                    "tooltip": "Official classifier-free guidance scale.",
                }),
                "sigma": ("FLOAT", {
                    "default": 0.25,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Official flow-matching sigma.",
                }),
                "temperature": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "Official sampling temperature; 0.0 is the released default.",
                }),
            },
            "optional": {
                "speed": (list(tasks.BASIC_SPEED_VALUES), {
                    "default": tasks.DEFAULT_SPEED,
                    "tooltip": "Basic attributes task: official speed value.",
                }),
                "pitch": (list(tasks.BASIC_PITCH_VALUES), {
                    "default": tasks.DEFAULT_PITCH,
                    "tooltip": "Basic attributes task: official pitch value.",
                }),
                "volume": (list(tasks.BASIC_VOLUME_VALUES), {
                    "default": tasks.DEFAULT_VOLUME,
                    "tooltip": "Basic attributes task: official volume value.",
                }),
                "emotion": (list(tasks.EMOTION_VALUES), {
                    "default": tasks.DEFAULT_EMOTION,
                    "tooltip": "Emotion task: official emotion value.",
                }),
                "dialect": (list(tasks.DIALECT_VALUES), {
                    "default": tasks.DEFAULT_DIALECT,
                    "tooltip": "Dialect task: official dialect value. No language_id is sent.",
                }),
                "ip_voice": (list(IP_VOICE_BASES), {
                    "default": "US_James",
                    "tooltip": (
                        "Built-in voice (IP) task: official catalog voice. The catalog has 50 "
                        "Chinese voices without emotion variants, 62 Chinese bases with 429 "
                        "emotion variants, and five English voices."
                    ),
                }),
                "ip_emotion": (list(IP_EMOTION_OPTIONS), {
                    "default": "None",
                    "tooltip": (
                        "Emotion variant for the selected Chinese IP base voice.\n"
                        "None uses the neutral variant for voices that have variants.\n"
                        "Voices without variants reject a non-None value."
                    ),
                }),
                "voice_style": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": (
                        "Voice design task: describe gender, age, timbre, pace and delivery. "
                        "No reference audio is used. This field is ignored by other tasks."
                    ),
                }),
                "instruction": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": (
                        "Freeform instruction task: natural-language delivery direction. "
                        "Keep the spoken words in the TTS text, not in this field. "
                        "This field is ignored by other tasks and cannot combine with controls."
                    ),
                }),
                "text_normalization": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Official FST text normalization. The official default is True, but it "
                        "requires pynini (optional and awkward on Windows). False keeps the "
                        "pipeline importable without pynini."
                    ),
                }),
                "cache_prompts": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Official prompt-latent cache within a synthesis request. The suite clears "
                        "it before every request so seeded output is independent of earlier "
                        "requests and Clear VRAM. Load-time option."
                    ),
                }),
                "upstream_path": ("STRING", {
                    "default": "auto",
                    "tooltip": (
                        "Official GitHub source checkout (ming_tts.py + modeling_bailingmm.py + "
                        "tts_tasks.py + frontend/).\n"
                        "auto searches models/TTS/ming_3_0_tts/upstream/.\n"
                        "An explicit path must stay inside ComfyUI's data folders.\n"
                        "This is separate from the Hugging Face checkpoint."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("TTS_ENGINE",)
    RETURN_NAMES = ("TTS_engine",)
    FUNCTION = "create_engine_config"
    CATEGORY = "TTS Audio Suite/⚙️ Engines"

    def create_engine_config(
        self,
        model: str,
        device: str,
        task_mode: str,
        language: str,
        max_length: int,
        max_decode_steps: int,
        cfg: float,
        sigma: float,
        temperature: float,
        speed: str = tasks.DEFAULT_SPEED,
        pitch: str = tasks.DEFAULT_PITCH,
        volume: str = tasks.DEFAULT_VOLUME,
        emotion: str = tasks.DEFAULT_EMOTION,
        dialect: str = tasks.DEFAULT_DIALECT,
        ip_voice: str = "US_James",
        ip_emotion: str = "None",
        voice_style: str = "",
        instruction: str = "",
        text_normalization: bool = False,
        cache_prompts: bool = True,
        upstream_path: str = "auto",
    ) -> tuple:
        if device not in self.DEVICE_OPTIONS:
            raise ValueError(
                f"Unsupported Ming 3.0 TTS device {device!r}. "
                f"Expected: {', '.join(self.DEVICE_OPTIONS)}"
            )
        if task_mode not in tasks.TASK_MODE_OPTIONS:
            raise ValueError(
                f"Unknown Ming 3.0 TTS task {task_mode!r}. "
                f"Expected: {', '.join(tasks.TASK_MODE_OPTIONS)}"
            )

        normalized_language = tasks.validate_task_language(task_mode, language)
        tasks.resolve_controls(
            task_mode,
            {
                "speed": speed,
                "pitch": pitch,
                "volume": volume,
                "emotion": emotion,
                "dialect": dialect,
                "ip_voice_id": resolve_ip_voice(ip_voice, ip_emotion)
                if task_mode == tasks.TASK_IP
                else "",
                "voice_style": voice_style,
            },
        )
        tasks.resolve_instruction(task_mode, {"instruction": instruction})
        ip_voice_id = (
            resolve_ip_voice(ip_voice, ip_emotion)
            if task_mode == tasks.TASK_IP
            else ""
        )
        resolved_upstream_path = str(upstream_path or "auto").strip() or "auto"

        config = {
            "engine_type": "ming_3_0_tts",
            "model": model,
            "device": device,
            "task_mode": task_mode,
            "language": normalized_language or "Auto",
            "speed": speed,
            "pitch": pitch,
            "volume": volume,
            "emotion": emotion,
            "dialect": dialect,
            "ip_voice": ip_voice,
            "ip_emotion": ip_emotion,
            "ip_voice_id": ip_voice_id,
            "voice_style": voice_style,
            "instruction": instruction,
            "max_length": int(max_length),
            "max_decode_steps": int(max_decode_steps),
            "cfg": float(cfg),
            "sigma": float(sigma),
            "temperature": float(temperature),
            "text_normalization": bool(text_normalization),
            "cache_prompts": bool(cache_prompts),
            "upstream_path": resolved_upstream_path,
        }

        print(f"⚙️ Ming 3.0 TTS: Configured on {device}")
        print(
            f"   Model: {model} | Task: {task_mode} | "
            f"Language: {format_ming_language(normalized_language)} | Device: {device}"
        )
        print(
            f"   Settings: cfg={float(cfg)}, sigma={float(sigma)}, "
            f"temperature={float(temperature)}, max_length={int(max_length)}, "
            f"max_decode_steps={int(max_decode_steps)}"
        )
        if task_mode in {
            tasks.TASK_BASIC,
            tasks.TASK_EMOTION,
            tasks.TASK_DIALECT,
            tasks.TASK_IP,
            tasks.TASK_VOICE_DESIGN,
            tasks.TASK_FREEFORM,
        }:
            print(f"   {tasks.task_summary(task_mode, config, None)}")
        if task_mode == tasks.TASK_DIALECT and normalize_ming_language(language):
            print("   Language is ignored by the official Dialect task.")
        print(
            f"   Text normalization: {bool(text_normalization)} | "
            f"Prompt cache: {bool(cache_prompts)}"
        )

        try:
            downloader = Ming30TTSDownloader()
            source = downloader.resolve_upstream_source(resolved_upstream_path)
            if source:
                print(f"   Upstream source: {source}")
            else:
                print("   ⚠️ Upstream source not found yet:")
                print("      " + downloader.upstream_setup_hint().replace("\n", "\n      "))
        except Exception as error:
            print(f"   ⚠️ Upstream source check failed: {error}")

        engine_data = {
            "engine_type": "ming_3_0_tts",
            "config": config,
            "capabilities": ["tts"],
        }
        return (engine_data,)
