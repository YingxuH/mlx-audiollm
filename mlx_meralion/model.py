"""Weight loading utilities for MERaLiON MLX.

Handles loading SafeTensors weights from HuggingFace model directories,
partitioning monolithic checkpoints into components, and key remapping
between HF and MLX formats.
"""

import json
from pathlib import Path

import mlx.core as mx
import numpy as np


def load_config(model_dir: str | Path) -> dict:
    """Load MERaLiON config.json."""
    config_path = Path(model_dir) / "config.json"
    with open(config_path) as f:
        return json.load(f)


def load_weights(model_dir: str | Path) -> dict[str, mx.array]:
    """Load all SafeTensors shards from model directory.

    Handles multi-shard SafeTensors files (model-00001-of-00004.safetensors, etc.)

    Args:
        model_dir: Path to model directory with .safetensors files

    Returns:
        Dict mapping weight names to MLX arrays
    """
    model_dir = Path(model_dir)
    weights = {}

    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        shard_files = sorted(set(index["weight_map"].values()))
    else:
        shard_files = sorted(f.name for f in model_dir.glob("*.safetensors"))

    for shard_name in shard_files:
        shard_path = model_dir / shard_name
        print(f"  Loading {shard_name}...")
        shard_weights = mx.load(str(shard_path))
        weights.update(shard_weights)

    return weights


def partition_weights(
    weights: dict[str, mx.array],
) -> tuple[dict, dict, dict, dict]:
    """Partition MERaLiON weights into component groups.

    MERaLiON weight keys follow these prefixes:
        - speech_encoder.* -> Whisper encoder
        - ln_speech.* -> Speech LayerNorm
        - speech_audio_adapter.* -> MLP adaptor
        - text_decoder.* -> Gemma2 decoder

    Returns:
        (encoder_weights, ln_weights, adaptor_weights, decoder_weights)
    """
    encoder_weights = {}
    ln_weights = {}
    adaptor_weights = {}
    decoder_weights = {}

    for key, value in weights.items():
        if key.startswith("speech_encoder."):
            new_key = key[len("speech_encoder.") :]
            encoder_weights[new_key] = value
        elif key.startswith("ln_speech."):
            new_key = key[len("ln_speech.") :]
            ln_weights[new_key] = value
        elif key.startswith("speech_audio_adapter."):
            new_key = key[len("speech_audio_adapter.") :]
            adaptor_weights[new_key] = value
        elif key.startswith("text_decoder."):
            new_key = key[len("text_decoder.") :]
            decoder_weights[new_key] = value
        else:
            decoder_weights[key] = value

    return encoder_weights, ln_weights, adaptor_weights, decoder_weights


def remap_whisper_keys(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    """Remap HuggingFace Whisper weight keys to our MLX Whisper format.

    HF format: encoder.layers.0.self_attn.q_proj.weight
    MLX format: layers.0.self_attn.q_proj.weight

    Also handles Conv1d weight transposition (HF uses [out, in, kernel],
    MLX Conv1d expects [out, kernel, in]).
    """
    remapped = {}

    for key, value in weights.items():
        new_key = key
        if new_key.startswith("encoder."):
            new_key = new_key[len("encoder.") :]

        if "embed_positions" in new_key:
            new_key = "embed_positions"
            remapped[new_key] = value
            continue

        # Conv1d weight transposition: HF (out_ch, in_ch, kernel) -> MLX (out_ch, kernel, in_ch)
        if ("conv1.weight" in new_key or "conv2.weight" in new_key) and value.ndim == 3:
            value = mx.transpose(value, axes=(0, 2, 1))

        remapped[new_key] = value

    return remapped


def remap_adaptor_keys(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    """Remap adaptor weight keys from HF to MLX format.

    nn.Sequential in MLX uses .layers. prefix for indexed children:
        HF: mlp_adapter.0.weight -> MLX: mlp_adapter.layers.0.weight
        HF: speech_llm_proj.0.weight -> MLX: speech_llm_proj.layers.0.weight

    Direct nn.Linear attributes (gate_proj, pool_proj, out_proj) need no remapping.
    """
    remapped = {}

    for key, value in weights.items():
        new_key = key

        for prefix in ("mlp_adapter", "speech_llm_proj"):
            if key.startswith(f"{prefix}.") and not key.startswith(f"{prefix}.layers."):
                suffix = key[len(f"{prefix}.") :]
                if suffix and suffix[0].isdigit():
                    new_key = f"{prefix}.layers.{suffix}"
                break

        remapped[new_key] = value

    return remapped


def save_component_weights(
    weights: dict[str, mx.array],
    output_path: str | Path,
):
    """Save component weights as SafeTensors."""
    from safetensors.numpy import save_file

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np_weights = {}
    for key, value in weights.items():
        if value.dtype == mx.bfloat16:
            value = value.astype(mx.float16)
        np_weights[key] = np.array(value)

    save_file(np_weights, str(output_path))
    size_mb = output_path.stat().st_size / (1024**2)
    print(f"  Saved {output_path.name} ({size_mb:.1f} MB, {len(np_weights)} tensors)")
