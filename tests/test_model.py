"""Tests for weight loading utilities."""

import json

import mlx.core as mx
import pytest

from mlx_meralion.model import (
    load_config,
    partition_weights,
    remap_adaptor_keys,
    remap_whisper_keys,
)


class TestPartitionWeights:
    """Tests for partition_weights()."""

    def test_partitions_by_prefix(self):
        weights = {
            "speech_encoder.layers.0.weight": mx.zeros((10,)),
            "ln_speech.weight": mx.zeros((10,)),
            "speech_audio_adapter.mlp.weight": mx.zeros((10,)),
            "text_decoder.layers.0.weight": mx.zeros((10,)),
        }
        enc, ln, adp, dec = partition_weights(weights)
        assert "layers.0.weight" in enc
        assert "weight" in ln
        assert "mlp.weight" in adp
        assert "layers.0.weight" in dec

    def test_strips_prefix(self):
        weights = {
            "speech_encoder.conv1.weight": mx.zeros((3,)),
            "text_decoder.embed_tokens.weight": mx.zeros((3,)),
        }
        enc, ln, adp, dec = partition_weights(weights)
        assert "conv1.weight" in enc
        assert "embed_tokens.weight" in dec

    def test_unknown_prefix_goes_to_decoder(self):
        weights = {
            "unknown_key": mx.zeros((3,)),
        }
        enc, ln, adp, dec = partition_weights(weights)
        assert len(enc) == 0
        assert len(ln) == 0
        assert len(adp) == 0
        assert "unknown_key" in dec

    def test_empty_weights(self):
        enc, ln, adp, dec = partition_weights({})
        assert len(enc) == 0
        assert len(ln) == 0
        assert len(adp) == 0
        assert len(dec) == 0


class TestRemapWhisperKeys:
    """Tests for remap_whisper_keys()."""

    def test_strips_encoder_prefix(self):
        weights = {"encoder.layers.0.weight": mx.zeros((3,))}
        remapped = remap_whisper_keys(weights)
        assert "layers.0.weight" in remapped
        assert "encoder.layers.0.weight" not in remapped

    def test_embed_positions_remapped(self):
        weights = {"encoder.embed_positions.weight": mx.zeros((1500, 1280))}
        remapped = remap_whisper_keys(weights)
        assert "embed_positions" in remapped

    def test_conv_weight_transposed(self):
        # HF conv1d: (out_ch, in_ch, kernel) = (1280, 80, 3)
        w = mx.zeros((1280, 80, 3))
        weights = {"conv1.weight": w}
        remapped = remap_whisper_keys(weights)
        # MLX conv1d: (out_ch, kernel, in_ch) = (1280, 3, 80)
        assert remapped["conv1.weight"].shape == (1280, 3, 80)

    def test_non_conv_weight_unchanged(self):
        w = mx.zeros((1280, 1280))
        weights = {"layers.0.self_attn.q_proj.weight": w}
        remapped = remap_whisper_keys(weights)
        assert remapped["layers.0.self_attn.q_proj.weight"].shape == (1280, 1280)


class TestRemapAdaptorKeys:
    """Tests for remap_adaptor_keys()."""

    def test_sequential_index_remapped(self):
        weights = {
            "mlp_adapter.0.weight": mx.zeros((3,)),
            "mlp_adapter.0.bias": mx.zeros((3,)),
            "speech_llm_proj.0.weight": mx.zeros((3,)),
            "speech_llm_proj.2.weight": mx.zeros((3,)),
        }
        remapped = remap_adaptor_keys(weights)
        assert "mlp_adapter.layers.0.weight" in remapped
        assert "mlp_adapter.layers.0.bias" in remapped
        assert "speech_llm_proj.layers.0.weight" in remapped
        assert "speech_llm_proj.layers.2.weight" in remapped

    def test_already_remapped_unchanged(self):
        weights = {"mlp_adapter.layers.0.weight": mx.zeros((3,))}
        remapped = remap_adaptor_keys(weights)
        assert "mlp_adapter.layers.0.weight" in remapped

    def test_non_sequential_keys_unchanged(self):
        weights = {
            "gate_proj.weight": mx.zeros((3,)),
            "out_proj.weight": mx.zeros((3,)),
        }
        remapped = remap_adaptor_keys(weights)
        assert "gate_proj.weight" in remapped
        assert "out_proj.weight" in remapped


class TestLoadConfig:
    """Tests for load_config()."""

    def test_loads_config_json(self, tmp_path):
        config = {"speech_token_index": 255999, "text_config": {"hidden_size": 3584}}
        with open(tmp_path / "config.json", "w") as f:
            json.dump(config, f)
        loaded = load_config(tmp_path)
        assert loaded["speech_token_index"] == 255999

    def test_missing_config_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path)
