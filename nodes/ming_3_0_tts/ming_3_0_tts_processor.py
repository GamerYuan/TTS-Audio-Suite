"""Ming 3.0 TTS processor.

Orchestrates character switching, language overrides, pause tags, native Ming
paralinguistic tags, chunking, and generation reporting for the unified TTS
Text node.
"""

from __future__ import annotations

import os
import re
import sys
from typing import Any, Dict, List, Tuple, Union

import comfy.model_management as model_management
import torch

current_dir = os.path.dirname(__file__)
nodes_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(nodes_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from engines.ming_3_0_tts import tasks
from engines.ming_3_0_tts.languages import (
    format_ming_language,
    language_to_parser_code,
    normalize_ming_language,
)
from engines.ming_3_0_tts.native_tags import protect_native_tags, restore_native_tags
from utils.audio.chunk_timing import ChunkTimingHelper
from utils.audio.edit_post_processor import process_segments as apply_edit_post_processing
from utils.text.character_parser import character_parser
from utils.text.pause_processor import PauseTagProcessor
from utils.text.segment_parameters import ParameterValidator, apply_segment_parameters
from utils.text.step_audio_editx_special_tags import get_edit_tags_for_segment
from utils.voice.character_logging import (
    format_resolved_character_block,
    resolved_character_label,
)
from utils.voice.discovery import get_available_characters, get_character_mapping, voice_discovery
from utils.voice.reference import effective_voice_audio


class Ming30TTSProcessor:
    """Internal processor for Ming 3.0 TTS generation."""

    SAMPLE_RATE = 44100

    def __init__(self, adapter, engine_config: Dict[str, Any]):
        self.adapter = adapter
        self.config = engine_config.copy() if engine_config else {}
        self._settings_signature = None

    def update_config(self, new_config: Dict[str, Any]):
        self.config = new_config.copy() if new_config else {}
        self.adapter.update_config(self.config)

    @staticmethod
    def _check_interrupt(message: str) -> None:
        if getattr(model_management, "interrupt_processing", False):
            raise InterruptedError(message)

    @staticmethod
    def _voice_log_note(voice_ref: Dict[str, Any], task_mode: str) -> str:
        if not tasks.uses_reference_audio(task_mode):
            return " [native task: no reference audio]"
        if not isinstance(voice_ref, dict) or effective_voice_audio(voice_ref) is None:
            if tasks.requires_reference_audio(task_mode):
                return " [⚠️ Reference voice required]"
            return " [⚠️ No reference voice - zero speaker]"
        if tasks.accepts_reference_transcript(task_mode):
            reference_text = str(voice_ref.get("reference_text") or "").strip()
            if reference_text:
                return f" [ref text: {len(reference_text)} chars]"
            return " [no ref text: speaker embedding only]"
        return " [speaker embedding only]"

    def _log_generation_text(
        self,
        character_name: str,
        text_content: str,
        language: str,
        voice_ref: Dict[str, Any],
        chunk_count: int,
        parameter_log: str = "",
        show_text_content: bool = True,
    ) -> None:
        task_mode = str(self.config.get("task_mode", tasks.TASK_TTS))
        voice_note = self._voice_log_note(voice_ref, task_mode)
        display_name = resolved_character_label(character_name, voice_ref)
        print(f"🎭 Ming 3.0 TTS - Generating for '{display_name}' (Language: {language}){voice_note}:")
        if parameter_log:
            print(f"🎛️ Ming 3.0 TTS params: {parameter_log}")
        if show_text_content:
            print(format_resolved_character_block(character_name, text_content, voice_ref))
        if chunk_count > 1:
            print(
                f"📝 Chunking {display_name}'s text into {chunk_count} chunks "
                f"(Language: {language}){voice_note}"
            )

    @staticmethod
    def _format_parameter_log(
        filtered_params: Dict[str, Any],
        current_config: Dict[str, Any],
        current_seed: int,
    ) -> str:
        if not filtered_params:
            return ""

        ordered = []
        if "seed" in filtered_params:
            ordered.append(f"seed={current_seed}")
        for key in (
            "cfg",
            "sigma",
            "temperature",
            "max_length",
            "max_decode_steps",
            "text_normalization",
            "language",
        ):
            if key in filtered_params:
                value = current_config.get(key)
                if key == "language":
                    value = format_ming_language(value)
                ordered.append(f"{key}={value}")

        known = {
            "seed",
            "cfg",
            "sigma",
            "temperature",
            "max_length",
            "max_decode_steps",
            "text_normalization",
            "language",
        }
        for key in filtered_params:
            if key not in known:
                ordered.append(f"{key}={current_config.get(key, filtered_params[key])}")
        return ", ".join(ordered)

    def _setup_character_parser(self, text: str) -> str:
        task_mode = str(self.config.get("task_mode", tasks.TASK_TTS))
        configured_language = self.config.get("language", "Auto")
        parser_default = language_to_parser_code(configured_language)

        character_parser.language_resolver.default_language = parser_default
        character_parser.default_language = parser_default

        character_tags = re.findall(r"\[([^\]]+)\]", text or "")
        characters_from_tags = []
        for tag in character_tags:
            if not tag.lower().startswith(("pause:", "wait:", "stop:")):
                characters_from_tags.append(tag.split("|")[0].strip())

        all_available = set()
        available_chars = get_available_characters()
        if available_chars:
            all_available.update(available_chars)
        for alias, target in voice_discovery.get_character_aliases().items():
            all_available.add(alias.lower())
            all_available.add(target.lower())
        for char in characters_from_tags:
            all_available.add(char.lower())
        all_available.add("narrator")
        character_parser.set_available_characters(list(all_available))

        for char, lang in voice_discovery.get_character_language_defaults().items():
            character_parser.set_character_language_default(char, lang)

        character_parser.reset_session_cache()
        return task_mode

    @staticmethod
    def _should_apply_segment_language(seg: Any, base_config: Dict[str, Any], task_mode: str) -> bool:
        if not tasks.accepts_language(task_mode):
            return False
        segment_language = getattr(seg, "language", None)
        if not segment_language:
            return False

        segment_code = normalize_ming_language(segment_language)
        if segment_code is None:
            return False
        if task_mode == tasks.TASK_MULTILINGUAL and segment_code not in (
            "German",
            "Spanish",
            "French",
            "Korean",
            "Japanese",
        ):
            print(
                f"  ⚠️ Ming Multilingual TTS ignores segment language '{segment_language}'. "
                "Only German, Spanish, French, Korean and Japanese are valid for this task."
            )
            return False

        if getattr(seg, "explicit_language", False):
            return True

        global_language = normalize_ming_language(base_config.get("language", "Auto"))
        parser_default = language_to_parser_code(base_config.get("language", "Auto"))
        # Untagged text is resolved to the parser fallback. Do not feed that
        # fallback back into generation, or Auto collapses into a concrete language.
        if normalize_ming_language(segment_language) == normalize_ming_language(parser_default):
            return False
        return segment_code != global_language

    def process_text(
        self,
        text: str,
        voice_mapping: Dict[str, Any],
        seed: int,
        enable_chunking: bool = True,
        max_chars_per_chunk: int = 400,
        chunk_combination_method: str = "auto",
        silence_between_chunks_ms: int = 100,
        enable_audio_cache: bool = True,
        apply_edit_postprocessing: bool = True,
        show_text_logging: bool = True,
    ) -> List[Dict[str, Any]]:
        self._check_interrupt("Ming 3.0 TTS interrupted before setup")

        protected_text, tag_replacements = protect_native_tags(text or "")
        task_mode = self._setup_character_parser(protected_text)

        narrator_info = voice_mapping.get("narrator", {})
        narrator_voice = narrator_info if isinstance(narrator_info, dict) else {"audio": narrator_info}

        base_config = self.config.copy()
        segment_objects = character_parser.parse_text_segments(protected_text)
        if not segment_objects:
            segment_objects = character_parser.parse_text_segments("narrator " + protected_text)

        characters = list({seg.character for seg in segment_objects if seg.character})
        character_mapping = get_character_mapping(characters, engine_type="audio_only")

        settings = self.adapter.describe_config(base_config)
        settings_signature = tuple(sorted(base_config.items(), key=lambda item: item[0]))
        if settings_signature != self._settings_signature:
            print(f"🎛️ Ming 3.0 TTS Settings: {settings}")
            print(f"   Model: {base_config.get('model', 'Ming-3.0-tts-0.5B')} | Device: {base_config.get('device', 'auto')}")
            self._settings_signature = settings_signature

        segment_records: List[Dict[str, Any]] = []
        for seg in segment_objects:
            self._check_interrupt("Ming 3.0 TTS interrupted during character setup")

            segment_text = restore_native_tags(seg.text or "", tag_replacements).strip()
            if not segment_text:
                continue

            current_config = base_config
            current_seed = seed
            filtered_params: Dict[str, Any] = {}
            segment_params = seg.parameters if seg.parameters else {}
            if segment_params:
                filtered_params = ParameterValidator.filter_parameters_for_engine(
                    segment_params, "ming_3_0_tts"
                )
                current_config = apply_segment_parameters(
                    base_config, segment_params, "ming_3_0_tts"
                )
                if "seed" in current_config:
                    current_seed = int(current_config.get("seed", seed))

            if self._should_apply_segment_language(seg, base_config, task_mode):
                current_config = current_config.copy()
                current_config["language"] = seg.language
                print(
                    "  🌍 Ming 3.0 TTS language switched to: "
                    f"{format_ming_language(seg.language)}"
                )

            self.adapter.update_config(current_config)
            segment_language = (
                format_ming_language(current_config.get("language"))
                if tasks.accepts_language(task_mode)
                else "n/a"
            )
            parameter_log = self._format_parameter_log(
                filtered_params, current_config, current_seed
            )

            voice_ref = narrator_voice.copy()
            char_name = seg.character or "narrator"
            if char_name != "narrator" and char_name in voice_mapping:
                char_info = voice_mapping[char_name]
                if isinstance(char_info, dict):
                    voice_ref = char_info.copy()
            elif char_name != "narrator":
                audio_path, ref_text = character_mapping.get(char_name, (None, None))
                if audio_path:
                    voice_ref = {
                        "audio_path": audio_path,
                        "reference_text": ref_text or "",
                    }

            native_chunking = tasks.uses_native_chunking(task_mode)
            if native_chunking and enable_chunking:
                print(
                    f"  🧩 Ming {task_mode} uses official semantic fragmenting; "
                    "suite character chunking is bypassed for this task."
                )

            def _generate_chunks(text_content: str, edit_tags: list):
                text_content = text_content.strip()
                if not text_content:
                    return
                if native_chunking or not enable_chunking:
                    chunks = [text_content]
                else:
                    from utils.text.chunking import ImprovedChatterBoxChunker

                    max_chars = ImprovedChatterBoxChunker.validate_chunking_params(
                        max_chars_per_chunk
                    )
                    chunks = ImprovedChatterBoxChunker.split_into_chunks(
                        text_content, max_chars=max_chars
                    )

                self._log_generation_text(
                    character_name=char_name,
                    text_content=text_content,
                    language=segment_language,
                    voice_ref=voice_ref,
                    chunk_count=len(chunks),
                    parameter_log=parameter_log,
                    show_text_content=show_text_logging,
                )

                for chunk_idx, chunk in enumerate(chunks):
                    self._check_interrupt(
                        f"Ming 3.0 TTS interrupted at chunk {chunk_idx + 1}/{len(chunks)}"
                    )
                    audio = self.adapter.generate_single(
                        text=chunk,
                        voice_ref=voice_ref,
                        seed=current_seed + chunk_idx,
                        enable_audio_cache=enable_audio_cache,
                        character_name=char_name,
                    )
                    if audio.dim() == 1:
                        audio = audio.unsqueeze(0)
                    segment_records.append(
                        {
                            "waveform": audio.cpu(),
                            "sample_rate": self.SAMPLE_RATE,
                            "text": chunk,
                            "edit_tags": edit_tags if chunk_idx == 0 else [],
                        }
                    )

            if PauseTagProcessor.has_pause_tags(segment_text):
                raw_pause_segments, _ = PauseTagProcessor.parse_pause_tags(segment_text)
                for frag_type, frag_content in raw_pause_segments:
                    self._check_interrupt("Ming 3.0 TTS interrupted during pause processing")
                    if frag_type == "text":
                        frag_clean, frag_edit_tags = get_edit_tags_for_segment(frag_content)
                        _generate_chunks(frag_clean, frag_edit_tags)
                    elif frag_type == "pause":
                        silence = PauseTagProcessor.create_silence_segment(
                            frag_content,
                            self.SAMPLE_RATE,
                            torch.device("cpu"),
                            torch.float32,
                        )
                        if silence.dim() == 1:
                            silence = silence.unsqueeze(0)
                        segment_records.append(
                            {
                                "waveform": silence,
                                "sample_rate": self.SAMPLE_RATE,
                                "text": f"[pause:{frag_content}s]",
                                "edit_tags": [],
                                "explicit_pause": True,
                            }
                        )
            else:
                clean_text, edit_tags = get_edit_tags_for_segment(segment_text)
                _generate_chunks(clean_text, edit_tags)

        self._check_interrupt("Ming 3.0 TTS interrupted before audio assembly")

        if (
            apply_edit_postprocessing
            and segment_records
            and any(seg["edit_tags"] for seg in segment_records)
        ):
            segment_records = apply_edit_post_processing(
                segment_records, engine_config=base_config
            )

        return segment_records

    def combine_audio_segments(
        self,
        segments: List[Dict[str, Any]],
        method: str = "auto",
        silence_ms: int = 100,
        original_text: str = "",
        return_info: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        if not segments:
            empty = torch.zeros(0, dtype=torch.float32)
            if return_info:
                return empty, {}
            return empty

        if any(segment.get("explicit_pause") for segment in segments):
            # Blend only contiguous speech runs. Explicit silence must never
            # overlap speech or acquire the combiner's extra silence padding.
            groups = []
            speech_run = []
            for segment in segments:
                if segment.get("explicit_pause"):
                    if speech_run:
                        groups.append(speech_run)
                        speech_run = []
                    groups.append([segment])
                else:
                    speech_run.append(segment)
            if speech_run:
                groups.append(speech_run)

            grouped_segments = []
            group_details = []
            for group in groups:
                group_info = None
                if len(group) == 1:
                    waveform = group[0]["waveform"]
                else:
                    waveform, group_info = self.combine_audio_segments(
                        group, method=method, silence_ms=silence_ms,
                        original_text=" ".join(item.get("text", "") for item in group),
                        return_info=True,
                    )
                label = " ".join(item.get("text", "") for item in group)
                if group_info:
                    label = (
                        f"Speech group ({len(group)} chunks, "
                        f"{group_info['method_used']}): {label}"
                    )
                grouped_segments.append({"waveform": waveform, "text": label})
                group_details.append({
                    "original_chunk_count": len(group),
                    "explicit_pause": bool(group[0].get("explicit_pause")),
                    "combination_info": group_info,
                })
            combined, outer_info = self.combine_audio_segments(
                grouped_segments, method="concatenate", silence_ms=0,
                original_text=original_text, return_info=True,
            )
            if outer_info:
                outer_info["method_used"] = "concatenate between pause-safe groups"
                outer_info["timing_unit"] = "groups"
                outer_info["original_chunk_count"] = len(segments)
                outer_info["group_details"] = group_details
            if return_info:
                return combined, outer_info
            return combined

        audio_segments = [segment["waveform"] for segment in segments]
        text_chunks = [segment.get("text", "") for segment in segments]
        text_length = len(" ".join(text_chunks))

        combined_audio, chunk_info = ChunkTimingHelper.combine_audio_with_timing(
            audio_segments=audio_segments,
            combination_method=method,
            silence_ms=silence_ms,
            crossfade_duration=0.1,
            sample_rate=self.SAMPLE_RATE,
            text_length=text_length,
            original_text=original_text,
            text_chunks=text_chunks,
        )

        if return_info:
            return combined_audio, chunk_info
        return combined_audio
