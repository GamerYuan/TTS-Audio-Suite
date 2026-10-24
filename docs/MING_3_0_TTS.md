# Ming 3.0 TTS

Official sources:

- Code: [inclusionAI/Ming-3.0-tts](https://github.com/inclusionAI/Ming-3.0-tts) (MIT)
- Weights: [inclusionAI/Ming-3.0-tts-0.5B](https://huggingface.co/inclusionAI/Ming-3.0-tts-0.5B) (MIT)

Ming 3.0 TTS is integrated through the unified **TTS Text** and **TTS SRT** nodes. It uses the main ComfyUI Python environment and does not create or install an isolated runtime.

## What the suite supports

- **TTS (voice cloning)**: reference audio; an exact reference transcript enables the full acoustic prompt. Without a transcript, the official API uses the speaker embedding only.
- **Multilingual TTS**: German, Spanish, French, Korean, Japanese (plus Chinese/English through the standard TTS task).
- **Basic attributes**: official speed, pitch, and volume values with a reference voice.
- **Emotion**: seven official values with a reference voice.
- **Dialect**: `cantonese` or `chuanyu dialect` with a reference voice. This task takes no language ID.
- **Built-in voice (IP)**: the official catalog contains 50 Chinese voices without emotion variants, 62 Chinese bases with 429 emotion variants, and five English voices.
- **Voice design**: describe a voice in text; no reference audio.
- **Freeform instruction**: natural-language delivery direction; cannot be combined with structured controls.
- **Inline paralinguistic tags**: `[Breathing]`, `[Cough]`, `[Sigh]`, `[Laughter]`, `[Surprise-ah]`, `[Surprise-oh]`, `[Dissatisfaction-hnn]`, `[Uhm]`, `[Confirmation-en]`, `[Question-ei]`, `[Question-ah]`, `[Question-en]`, `[Question-oh]`, `[Question-huh]`.
- **Character, language, pause, and per-segment parameter tags** in TTS Text and SRT.

Ming has one task per call. Controls from different tasks are rejected by the official prompt builder; the suite does not fake combinations.

## Checkpoint and source setup

The Hugging Face checkpoint contains weights, tokenizer files, and `campplus.onnx`, but **not** executable code. The official GitHub source is required.

1. Let the suite download the checkpoint, or download it manually:

   ```text
   ComfyUI/models/TTS/ming_3_0_tts/Ming-3.0-tts-0.5B/
   ```

2. Put the official source here:

   ```text
   ComfyUI/models/TTS/ming_3_0_tts/upstream/
   ```

   The folder must contain `ming_tts.py`, `modeling_bailingmm.py`, `tts_tasks.py`, `frontend/`, `audio_tokenizer/`, and `fm/`.

   The suite does not download this source automatically because importing it executes it. The engine node's `upstream_path` input can also point to another folder inside ComfyUI's data folders; `MING_3_0_TTS_SOURCE` is available for advanced setups.

3. Install the official runtime dependencies in the active ComfyUI environment. The suite already declares `x-transformers`, `einops`, and `inflect`, and its installer provides `diffusers` and `onnxruntime`. Ming's upstream pins are `torch==2.6`, `torchaudio==2.6`, `transformers==4.46.3`, `x_transformers==2.9.2`, and `diffusers==0.33`.

   This integration intentionally does **not** replace or downgrade the suite's Torch/Transformers stack. Source inspection found the 0.5B runtime using stable `PreTrainedModel`/`Qwen2ForCausalLM` APIs that Transformers 5 still exposes, but live loading in a Transformers 5 environment has not been verified here. If the official source fails to import or load against the active Transformers version, report the exact traceback rather than changing the suite's shared dependencies.

4. Optional FST text normalization requires `pynini`. It is disabled by default because it is optional and awkward to install on Windows. Enabling `text_normalization` without `pynini` stops with an actionable error.

## Generation behavior and reproducibility

- Native semantic fragmenting is preserved. `max_length` defaults to 24; `raw_text=True` is not exposed because it bypasses both the official splitter and normalization.
- Voice Design and Freeform bypass suite character chunking so the official first-fragment-as-reference behavior is preserved.
- Reference audio is copied to a content-addressed temporary WAV. The official prompt cache is keyed by path, so a changed reference gets a new path and cannot reuse a stale prompt latent.
- Seeds are applied to Torch and CUDA before every generation. The upstream prompt-latent cache is cleared first: its stochastic VAE encoding must consume the same RNG draws regardless of earlier requests or Clear VRAM. `cache_prompts` therefore permits reuse only within a synthesis request, not across requests. The suite audio cache includes text, reference identity, task, language, controls, seed, and all behavior-affecting parameters.
- The official 0.5B runtime requires CUDA bf16; CPU/MPS/XPU are not offered.

## Clear VRAM

The wrapper implements ComfyUI's `.to()`/unload hooks, clears the official runtime and prompt cache, drops model references, and runs GC/CUDA cache cleanup. An engine-specific handler is registered so Clear VRAM tears down the runtime instead of only moving weights.

## Current validation limitation

Focused unit tests use mocks and a fake upstream `MingTTS`; they verify the suite contract, lifecycle, cache keys, tag handling, and prompt paths. They do **not** prove inference quality or live loading of the official checkpoint in this environment. No model download or live ComfyUI run was performed here.
