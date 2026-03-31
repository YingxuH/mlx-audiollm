#!/usr/bin/env python3
"""Run MERaLiON inference on Apple Silicon via MLX.

Usage:
    # Transcribe audio (ASR)
    mlx-meralion --model MERaLiON/MERaLiON-2-10B-MLX --audio test.wav --task asr

    # Translate to Chinese
    mlx-meralion --model MERaLiON/MERaLiON-2-10B-MLX --audio test.wav --task translate_zh

    # Spoken question answering
    mlx-meralion --model MERaLiON/MERaLiON-2-10B-MLX --audio test.wav --task sqa --question "What is the speaker talking about?"

    # Custom instruction
    mlx-meralion --model MERaLiON/MERaLiON-2-10B-MLX --audio test.wav --instruction "Summarize this in one sentence."
"""

import argparse
import json
import shutil
import sys
import time
from collections import namedtuple
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .processor import MERaLiONProcessor, get_task_prompt, load_audio, SAMPLE_RATE
from .whisper_encoder import WhisperEncoder, WhisperEncoderConfig
from .adaptor import create_adapter

# ---------------------------------------------------------------------------
# N-gram blocking sampler
# ---------------------------------------------------------------------------

NO_REPEAT_NGRAM_SIZE = 6


def make_no_repeat_ngram_sampler(ngram_size: int = NO_REPEAT_NGRAM_SIZE):
    """Create a greedy sampler that blocks repeated n-grams.

    Tracks generated n-grams and falls back to the next-best token when
    the greedy choice would create a repeated n-gram. This matches the
    behavior of HuggingFace's no_repeat_ngram_size in generation_config.json.

    Note: calls int() per token which forces synchronous evaluation.
    This is acceptable because mlx-lm's generate_step already calls
    y.item() before yield, so the token is already materialized.
    Requires sufficient GPU memory to avoid thrashing.

    Returns a sampler compatible with mlx-lm's generate_step(sampler=...).
    """
    prefix_to_next: dict[tuple[int, ...], set[int]] = {}
    id_list: list[int] = []

    def _register(token: int):
        id_list.append(token)
        if len(id_list) >= ngram_size:
            prefix = tuple(id_list[-ngram_size:-1])
            if prefix not in prefix_to_next:
                prefix_to_next[prefix] = set()
            prefix_to_next[prefix].add(id_list[-1])

    def sampler(logits: mx.array) -> mx.array:
        flat = logits.reshape(-1) if logits.ndim == 2 else logits
        token = mx.argmax(flat)
        tid = int(token)

        if len(id_list) >= ngram_size - 1:
            ctx = tuple(id_list[-(ngram_size - 1):])
            banned = prefix_to_next.get(ctx)
            if banned and tid in banned:
                sorted_ids = mx.argsort(flat)[::-1]
                for candidate in sorted_ids:
                    cid = int(candidate)
                    if cid not in banned:
                        tid = cid
                        token = candidate
                        break

        _register(tid)
        return token.reshape(logits.shape[:-1]) if logits.ndim == 2 else token

    return sampler


# ---------------------------------------------------------------------------
# Model directory detection and auto-conversion
# ---------------------------------------------------------------------------


def is_converted_dir(model_dir: Path) -> bool:
    """Check if model_dir contains converted MLX weights."""
    return (model_dir / "encoder_config.json").exists()


def is_raw_hf_dir(model_dir: Path) -> bool:
    """Check if model_dir contains raw HuggingFace model files."""
    return (model_dir / "config.json").exists() and any(model_dir.glob("model-*.safetensors"))


def auto_convert(model_dir: Path, verbose: bool = True) -> Path:
    """Auto-convert raw HF model to MLX format if needed.

    Returns the path to the converted model directory.
    """
    if is_converted_dir(model_dir):
        return model_dir

    if not is_raw_hf_dir(model_dir):
        raise FileNotFoundError(
            f"{model_dir} is neither a converted MLX dir nor a raw HF dir. "
            "Expected either encoder_config.json (converted) or model-*.safetensors (raw)."
        )

    converted_dir = model_dir.parent / f"{model_dir.name}-mlx"
    if is_converted_dir(converted_dir):
        if verbose:
            print(f"Using existing converted model: {converted_dir}")
        return converted_dir

    if verbose:
        print(f"Raw HF model detected. Auto-converting to {converted_dir}...")

    from .model import (
        load_config,
        load_weights,
        partition_weights,
        remap_whisper_keys,
        remap_adaptor_keys,
        save_component_weights,
    )

    converted_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(model_dir)

    with open(converted_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    weights = load_weights(model_dir)
    encoder_w, ln_w, adaptor_w, decoder_w = partition_weights(weights)

    # Save encoder
    encoder_config = config.get("speech_config", {})
    with open(converted_dir / "encoder_config.json", "w") as f:
        json.dump(encoder_config, f, indent=2)
    remapped_enc = remap_whisper_keys(encoder_w)
    save_component_weights(remapped_enc, converted_dir / "encoder.safetensors")

    # Save adaptor (+ ln_speech)
    adaptor_config = {
        "speech_hidden_size": encoder_config.get("d_model", 1280),
        "text_hidden_size": config.get("text_config", {}).get("hidden_size", 3584),
        "scale_factor": config.get("speech_mlp_scale_factor", 15),
    }
    with open(converted_dir / "adaptor_config.json", "w") as f:
        json.dump(adaptor_config, f, indent=2)
    remapped_adp = remap_adaptor_keys(adaptor_w)
    ln_prefixed = {f"ln_speech.{k}": v for k, v in ln_w.items()}
    remapped_adp.update(ln_prefixed)
    save_component_weights(remapped_adp, converted_dir / "adaptor.safetensors")

    # Save decoder
    text_config = config.get("text_config", {})
    with open(converted_dir / "decoder_config.json", "w") as f:
        json.dump(text_config, f, indent=2)

    # Shard decoder weights if > 4GB
    total_bytes = sum(v.nbytes for v in decoder_w.values())
    max_shard_bytes = 4 * 1024**3
    if total_bytes <= max_shard_bytes:
        save_component_weights(decoder_w, converted_dir / "decoder.safetensors")
    else:
        shard_idx = 0
        current_shard = {}
        current_bytes = 0
        index_map = {}
        for key in sorted(decoder_w.keys()):
            val = decoder_w[key]
            if current_bytes + val.nbytes > max_shard_bytes and current_shard:
                fname = f"decoder-{shard_idx:05d}.safetensors"
                save_component_weights(current_shard, converted_dir / fname)
                shard_idx += 1
                current_shard = {}
                current_bytes = 0
            current_shard[key] = val
            current_bytes += val.nbytes
            fname = f"decoder-{shard_idx:05d}.safetensors"
            index_map[key] = fname
        if current_shard:
            fname = f"decoder-{shard_idx:05d}.safetensors"
            save_component_weights(current_shard, converted_dir / fname)
        with open(converted_dir / "decoder.safetensors.index.json", "w") as f:
            json.dump({"weight_map": index_map}, f, indent=2)

    # Copy tokenizer files
    for pattern in ("tokenizer*", "special_tokens*"):
        for src in model_dir.glob(pattern):
            dst = converted_dir / src.name
            if not dst.exists():
                shutil.copy2(src, dst)

    if verbose:
        print(f"Auto-conversion complete: {converted_dir}\n")

    return converted_dir


def resolve_model_dir(model: str, verbose: bool = True) -> Path:
    """Resolve a model string to a local directory path.

    Accepts:
        - Local path to a converted MLX directory
        - Local path to a raw HF directory (auto-converts)
        - HuggingFace repo ID like "MERaLiON/MERaLiON-2-10B-MLX" (auto-downloads)

    Returns:
        Path to the converted MLX model directory.
    """
    model_path = Path(model)

    # Local directory
    if model_path.exists():
        return auto_convert(model_path, verbose=verbose)

    # HuggingFace repo ID — download via huggingface_hub
    if "/" in model and not model_path.exists():
        if verbose:
            print(f"Downloading model from HuggingFace: {model}")
        from huggingface_hub import snapshot_download

        local_dir = snapshot_download(model)
        return auto_convert(Path(local_dir), verbose=verbose)

    raise FileNotFoundError(
        f"Model not found: {model}. "
        "Provide a local path or HuggingFace repo ID (e.g. MERaLiON/MERaLiON-2-10B-MLX)."
    )


# ---------------------------------------------------------------------------
# Component loading
# ---------------------------------------------------------------------------


def load_encoder(model_dir: Path) -> WhisperEncoder:
    """Load converted Whisper encoder."""
    config_path = model_dir / "encoder_config.json"
    weights_path = model_dir / "encoder.safetensors"

    with open(config_path) as f:
        config_dict = json.load(f)

    config = WhisperEncoderConfig.from_dict(config_dict)
    encoder = WhisperEncoder(config)

    weights = mx.load(str(weights_path))
    encoder.load_weights(list(weights.items()))
    mx.eval(encoder.parameters())

    return encoder


def detect_adaptor_variant(adaptor_weights: dict) -> str:
    """Detect whether weights are v1 (simple) or v2 (gated) adaptor."""
    keys = set(adaptor_weights.keys())
    if any("gate_proj" in k for k in keys):
        return "v2"
    return "v1"


def load_adaptor(model_dir: Path) -> tuple:
    """Load converted MLP adaptor and LayerNorm.

    Auto-detects v1 vs v2 adaptor architecture from weight keys.
    """
    config_path = model_dir / "adaptor_config.json"
    weights_path = model_dir / "adaptor.safetensors"

    with open(config_path) as f:
        config = json.load(f)

    all_weights = mx.load(str(weights_path))

    adaptor_weights = {}
    ln_weights = {}
    for key, value in all_weights.items():
        if key.startswith("ln_speech."):
            ln_weights[key[len("ln_speech.") :]] = value
        else:
            adaptor_weights[key] = value

    variant = detect_adaptor_variant(adaptor_weights)
    adaptor = create_adapter(
        variant=variant,
        speech_hidden_size=config["speech_hidden_size"],
        text_hidden_size=config["text_hidden_size"],
        scale_factor=config["scale_factor"],
    )

    ln_speech = nn.LayerNorm(config["speech_hidden_size"])

    adaptor.load_weights(list(adaptor_weights.items()))
    ln_speech.load_weights(list(ln_weights.items()))
    mx.eval(adaptor.parameters())
    mx.eval(ln_speech.parameters())

    return adaptor, ln_speech


def load_decoder(model_dir: Path):
    """Load text decoder using mlx-lm.

    Returns:
        (model, tokenizer) from mlx-lm
    """
    try:
        from mlx_lm import load as mlx_lm_load

        decoder_config_path = model_dir / "decoder_config.json"
        if not decoder_config_path.exists():
            raise FileNotFoundError(
                f"decoder_config.json not found in {model_dir}. Run convert.py first."
            )

        decoder_dir = model_dir / "decoder"
        decoder_dir.mkdir(exist_ok=True)

        shutil.copy2(decoder_config_path, decoder_dir / "config.json")

        for f in model_dir.glob("decoder*.safetensors"):
            target = decoder_dir / f.name.replace("decoder-", "model-")
            if not target.exists():
                target.symlink_to(f.resolve())

        for f in model_dir.glob("tokenizer*"):
            target = decoder_dir / f.name
            if not target.exists():
                shutil.copy2(f, target)

        for f in model_dir.glob("special_tokens*"):
            target = decoder_dir / f.name
            if not target.exists():
                shutil.copy2(f, target)

        model, tokenizer = mlx_lm_load(str(decoder_dir))
        return model, tokenizer

    except ImportError:
        print("Error: mlx-lm is required for decoder loading.")
        print("Install with: pip install mlx-lm")
        sys.exit(1)


def patch_decoder_for_embeddings(decoder_model):
    """Patch mlx-lm Gemma2 model to accept input_embeddings parameter.

    This enables use with mlx_lm.generate.generate_step(), which passes
    input_embeddings as a keyword argument. The patch adds a code path
    that uses provided embeddings instead of calling embed_tokens, while
    preserving ALL model internals: sqrt scaling, mask creation, attention
    softcapping, layer iteration, final logit softcapping.
    """
    from mlx_lm.models.base import create_attention_mask

    inner = decoder_model.model

    def patched_inner_call(self, inputs, cache=None, input_embeddings=None):
        if input_embeddings is not None:
            h = input_embeddings
        else:
            h = self.embed_tokens(inputs)
        h = h * (self.args.hidden_size**0.5)

        if cache is None:
            cache = [None] * len(self.layers)

        mask = create_attention_mask(h, cache[0], return_array=True)

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)

        return self.norm(h)

    def patched_outer_call(self, inputs, cache=None, input_embeddings=None):
        out = self.model(inputs, cache, input_embeddings=input_embeddings)
        out = self.model.embed_tokens.as_linear(out)
        out = mx.tanh(out / self.final_logit_softcapping)
        out = out * self.final_logit_softcapping
        return out

    type(inner).__call__ = patched_inner_call
    type(decoder_model).__call__ = patched_outer_call


def build_merged_embeddings(
    decoder_model,
    input_ids: mx.array,
    speech_embeds: mx.array,
    speech_token_index: int,
) -> mx.array:
    """Build merged text+speech embeddings (UNscaled).

    Returns UNscaled embeddings because the model's __call__ applies
    sqrt(hidden_size) scaling internally after embed_tokens.
    """
    embed_fn = decoder_model.model.embed_tokens
    text_embeds = embed_fn(input_ids)

    B, S, H = text_embeds.shape

    for b in range(B):
        speech_idx = 0
        for s in range(S):
            if int(input_ids[b, s]) == speech_token_index and speech_idx < speech_embeds.shape[1]:
                text_embeds = text_embeds.at[b, s].add(
                    speech_embeds[b, speech_idx] - text_embeds[b, s]
                )
                speech_idx += 1

    return text_embeds


# ---------------------------------------------------------------------------
# Model loading and segment-level inference
# ---------------------------------------------------------------------------

LoadedModel = namedtuple(
    "LoadedModel",
    [
        "encoder",
        "adaptor",
        "ln_speech",
        "decoder",
        "processor",
    ],
)


def load_model(model: str | Path, verbose: bool = True) -> LoadedModel:
    """Load all model components.

    Args:
        model: Local path to MLX model directory, or HuggingFace repo ID
               (e.g. "MERaLiON/MERaLiON-2-10B-MLX"). HF repos are
               automatically downloaded and cached.
        verbose: Print loading progress

    Returns:
        LoadedModel namedtuple that can be reused across segments.
    """
    model_dir = resolve_model_dir(str(model), verbose=verbose)

    if verbose:
        print("Loading model components...")

    t0 = time.time()
    encoder = load_encoder(model_dir)
    if verbose:
        print(f"  Encoder loaded in {time.time() - t0:.1f}s")

    t0 = time.time()
    adaptor, ln_speech = load_adaptor(model_dir)
    if verbose:
        print(f"  Adaptor loaded in {time.time() - t0:.1f}s")

    t0 = time.time()
    decoder, _ = load_decoder(model_dir)
    if verbose:
        print(f"  Decoder loaded in {time.time() - t0:.1f}s")

    patch_decoder_for_embeddings(decoder)
    processor = MERaLiONProcessor(model_dir)

    return LoadedModel(encoder, adaptor, ln_speech, decoder, processor)


def _infer_segment(
    model: LoadedModel,
    audio_array: np.ndarray,
    instruction: str,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    verbose: bool = True,
) -> str:
    """Run inference on a single audio segment.

    N-gram blocking (size=6) is always enabled to match HuggingFace
    generation_config.json and prevent repetitive output.
    """
    from mlx_lm.generate import generate_step

    # Prepare audio (chunked into 30s pieces internally)
    mel_features, num_chunks = model.processor.prepare_audio(
        audio_array=audio_array,
        max_duration=None,
    )
    mel_mx = mx.array(mel_features)
    if verbose:
        print(f"  Mel: {mel_features.shape} ({num_chunks} chunk{'s' if num_chunks > 1 else ''})")

    # Prepare text (expands <SpeechHere> to 100 * num_chunks positions)
    text_inputs = model.processor.prepare_text(instruction, num_chunks=num_chunks)
    input_ids = mx.array(text_inputs["input_ids"])

    # Encode speech
    t0 = time.time()
    chunk_embeds = []
    for i in range(num_chunks):
        chunk_mel = mel_mx[i : i + 1]
        enc_out = model.encoder(chunk_mel)
        enc_out = model.ln_speech(enc_out)
        chunk_speech = model.adaptor(enc_out)
        chunk_embeds.append(chunk_speech)
    speech_embeds = mx.concatenate(chunk_embeds, axis=1)
    mx.eval(speech_embeds)
    if verbose:
        print(f"  Encoded {num_chunks} chunk(s) in {time.time() - t0:.2f}s")

    # Build merged embeddings
    t0 = time.time()
    merged_embeds = build_merged_embeddings(
        model.decoder,
        input_ids,
        speech_embeds,
        model.processor.speech_token_index,
    )
    mx.eval(merged_embeds)

    # Generate with n-gram blocking sampler
    prompt_tokens = input_ids[0]
    embeddings_2d = merged_embeds[0]

    sampler = make_no_repeat_ngram_sampler(NO_REPEAT_NGRAM_SIZE)

    eos_tokens = {1, 107}
    if (
        hasattr(model.processor.tokenizer, "eos_token_id")
        and model.processor.tokenizer.eos_token_id is not None
    ):
        eos_tokens.add(model.processor.tokenizer.eos_token_id)

    generated_tokens = []
    for token, logprobs in generate_step(
        prompt=prompt_tokens,
        model=model.decoder,
        max_tokens=max_new_tokens,
        sampler=sampler,
        input_embeddings=embeddings_2d,
    ):
        token_id = token.item() if hasattr(token, "item") else int(token)
        if token_id in eos_tokens:
            break
        generated_tokens.append(token_id)

    gen_time = time.time() - t0
    response = model.processor.decode(generated_tokens)

    if verbose:
        n_tokens = len(generated_tokens)
        tps = n_tokens / gen_time if gen_time > 0 else 0
        print(f"  Generated {n_tokens} tokens in {gen_time:.2f}s ({tps:.1f} tok/s)")

    return response


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------

_CHUNK_SAMPLES = 30 * 16000  # 480000 samples = 30s at 16kHz
_MIN_LAST_CHUNK = 10 * 16000  # 160000 samples = 10s


def transcribe(
    model: LoadedModel,
    audio: str | np.ndarray,
    task: str = "asr",
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    verbose: bool = False,
    **task_kwargs,
) -> str:
    """Transcribe or process audio using a loaded MERaLiON model.

    Smart chunking: splits long audio at 30s boundaries, merges short
    (<10s) tail segments with the previous chunk to prevent hallucination.
    N-gram blocking is always enabled.

    Args:
        model: LoadedModel from load_model()
        audio: Path to audio file, or pre-loaded 16kHz float32 numpy array
        task: Task key (asr, translate_zh, translate_id, translate_ms,
              translate_ta, sqa, summarize, paralinguistics)
        max_new_tokens: Maximum tokens to generate per chunk
        temperature: Sampling temperature (0 = greedy)
        verbose: Print progress info
        **task_kwargs: Extra args for task prompt (e.g. question="..." for sqa)

    Returns:
        Generated text string
    """
    instruction = get_task_prompt(task, **task_kwargs)

    # Load audio if path given
    if isinstance(audio, (str, Path)):
        audio_array = load_audio(str(audio))
    else:
        audio_array = audio

    # Short audio: single inference call
    if len(audio_array) <= _CHUNK_SAMPLES:
        return _infer_segment(
            model,
            audio_array,
            instruction,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            verbose=verbose,
        )

    # Long audio: smart chunking
    segments = [
        audio_array[start : start + _CHUNK_SAMPLES]
        for start in range(0, len(audio_array), _CHUNK_SAMPLES)
    ]

    # Merge short last segment into the previous one
    if len(segments) > 1 and len(segments[-1]) < _MIN_LAST_CHUNK:
        segments = segments[:-2] + [np.concatenate([segments[-2], segments[-1]])]

    parts = []
    for i, seg in enumerate(segments):
        if verbose:
            print(f"[Chunk {i + 1}/{len(segments)}]")
        text = _infer_segment(
            model,
            seg,
            instruction,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            verbose=verbose,
        )
        parts.append(text.strip())

    return " ".join(p for p in parts if p)


def _format_time(seconds: float) -> str:
    """Format seconds as M:SS or H:MM:SS."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def run_inference(
    model_dir: Path,
    audio_path: str,
    instruction: str,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    max_audio_duration: float | None = None,
    segment_length: float = 300.0,
    show_segments: bool = True,
    show_full: bool = True,
    verbose: bool = True,
) -> str:
    """Run full MERaLiON inference pipeline with automatic segmentation.

    For audio longer than segment_length, the audio is split into
    independent segments. Each segment is processed through the full
    encode-merge-generate pipeline.

    Returns:
        Tuple of (generated_text, num_segments)
    """
    t_start = time.time()

    model = load_model(model_dir, verbose=verbose)

    if verbose:
        print(f"\nProcessing audio: {audio_path}")
    audio = load_audio(audio_path)
    total_duration = len(audio) / SAMPLE_RATE

    if max_audio_duration is not None:
        max_samples = int(max_audio_duration * SAMPLE_RATE)
        audio = audio[:max_samples]
        total_duration = len(audio) / SAMPLE_RATE

    if verbose:
        print(f"  Duration: {_format_time(total_duration)} ({total_duration:.1f}s)")

    segment_samples = int(segment_length * SAMPLE_RATE)
    num_segments = max(1, -(-len(audio) // segment_samples))

    if num_segments == 1:
        response = _infer_segment(
            model,
            audio,
            instruction,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            verbose=verbose,
        )
        if verbose:
            print(f"  Total pipeline: {time.time() - t_start:.2f}s")
        return response, 1

    if verbose:
        print(
            f"  Splitting into {num_segments} segments (up to {_format_time(segment_length)} each)"
        )

    results = []
    for seg_idx in range(num_segments):
        seg_start = seg_idx * segment_samples
        seg_end = min(seg_start + segment_samples, len(audio))
        seg_audio = audio[seg_start:seg_end]

        start_time = seg_start / SAMPLE_RATE
        end_time = seg_end / SAMPLE_RATE
        header = (
            f"[Segment {seg_idx + 1}/{num_segments}"
            f" | {_format_time(start_time)}\u2013{_format_time(end_time)}]"
        )

        if verbose or show_segments:
            print(f"\n{header}")

        text = _infer_segment(
            model,
            seg_audio,
            instruction,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            verbose=verbose,
        )
        results.append(text)

        if show_segments:
            print(text)

    combined = " ".join(results)

    if verbose:
        print(f"\n  Total pipeline: {time.time() - t_start:.2f}s ({num_segments} segments)")

    if show_full:
        print(f"\n{'=' * 60}")
        print(f"Full transcript:\n{combined}")
        print(f"{'=' * 60}")

    return combined, num_segments


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="MERaLiON inference on Apple Silicon via MLX")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to MLX model directory or HuggingFace repo ID "
        "(e.g. MERaLiON/MERaLiON-2-10B-MLX)",
    )
    parser.add_argument(
        "--audio",
        type=str,
        required=True,
        help="Path to input audio file (WAV, MP3, FLAC, etc.)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Predefined task: asr, translate_zh, translate_id, sqa, summarize, "
        "instruction, paralinguistics",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default=None,
        help="Custom text instruction (overrides --task)",
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Question for SQA task",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate per segment (default: 256)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0 = greedy, default: 0.0)",
    )
    parser.add_argument(
        "--max-audio-duration",
        type=float,
        default=None,
        help="Truncate audio to this many seconds (default: no limit)",
    )
    parser.add_argument(
        "--segment-length",
        type=float,
        default=300.0,
        help="Maximum seconds per inference segment (default: 300).",
    )
    parser.add_argument(
        "--show-segments",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print each segment's text as it completes (default: on)",
    )
    parser.add_argument(
        "--show-full",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print combined full text after all segments (default: on)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )

    args = parser.parse_args()

    if args.segment_length <= 0:
        print("Error: --segment-length must be > 0")
        sys.exit(1)

    if args.instruction:
        instruction = args.instruction
    elif args.task:
        kwargs = {}
        if args.task == "sqa":
            if not args.question:
                print("Error: --question is required for SQA task")
                sys.exit(1)
            kwargs["question"] = args.question
        instruction = get_task_prompt(args.task, **kwargs)
    else:
        instruction = get_task_prompt("asr")

    if not args.quiet:
        print(f"Task instruction: {instruction}")

    response, num_segments = run_inference(
        model_dir=args.model,
        audio_path=args.audio,
        instruction=instruction,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        max_audio_duration=args.max_audio_duration,
        segment_length=args.segment_length,
        show_segments=args.show_segments,
        show_full=args.show_full,
        verbose=not args.quiet,
    )

    if num_segments == 1:
        print(f"\n{'=' * 60}")
        print(f"Response:\n{response}")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
