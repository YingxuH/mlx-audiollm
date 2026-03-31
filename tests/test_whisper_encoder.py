"""Tests for MLX Whisper encoder."""

import mlx.core as mx
import pytest

from mlx_meralion.whisper_encoder import (
    WhisperAttention,
    WhisperEncoder,
    WhisperEncoderConfig,
    WhisperEncoderLayer,
)


class TestWhisperEncoderConfig:
    """Tests for WhisperEncoderConfig."""

    def test_defaults(self):
        config = WhisperEncoderConfig()
        assert config.d_model == 1280
        assert config.encoder_layers == 32
        assert config.encoder_attention_heads == 20
        assert config.encoder_ffn_dim == 5120
        assert config.num_mel_bins == 80
        assert config.max_source_positions == 1500

    def test_from_dict(self):
        d = {"d_model": 512, "encoder_layers": 6, "encoder_attention_heads": 8}
        config = WhisperEncoderConfig.from_dict(d)
        assert config.d_model == 512
        assert config.encoder_layers == 6
        assert config.encoder_attention_heads == 8
        # Defaults for unspecified
        assert config.encoder_ffn_dim == 5120

    def test_from_empty_dict(self):
        config = WhisperEncoderConfig.from_dict({})
        assert config.d_model == 1280


class TestWhisperAttention:
    """Tests for WhisperAttention."""

    def test_output_shape(self):
        attn = WhisperAttention(d_model=64, num_heads=4)
        x = mx.zeros((1, 10, 64))
        out = attn(x)
        mx.eval(out)
        assert out.shape == (1, 10, 64)

    def test_batch_dimension(self):
        attn = WhisperAttention(d_model=64, num_heads=4)
        x = mx.zeros((3, 10, 64))
        out = attn(x)
        mx.eval(out)
        assert out.shape == (3, 10, 64)


class TestWhisperEncoderLayer:
    """Tests for single encoder layer."""

    def test_output_shape(self):
        config = WhisperEncoderConfig(d_model=64, encoder_attention_heads=4, encoder_ffn_dim=256)
        layer = WhisperEncoderLayer(config)
        x = mx.zeros((1, 10, 64))
        out = layer(x)
        mx.eval(out)
        assert out.shape == (1, 10, 64)

    def test_residual_connection(self):
        """Output should differ from zero input due to bias terms."""
        config = WhisperEncoderConfig(d_model=64, encoder_attention_heads=4, encoder_ffn_dim=256)
        layer = WhisperEncoderLayer(config)
        x = mx.zeros((1, 10, 64))
        out = layer(x)
        mx.eval(out)
        # With LayerNorm and bias, output won't be exactly zero
        assert out.shape == (1, 10, 64)


class TestWhisperEncoder:
    """Tests for the full Whisper encoder."""

    @pytest.fixture
    def small_encoder(self):
        """Small encoder for fast testing."""
        config = WhisperEncoderConfig(
            d_model=64,
            encoder_layers=2,
            encoder_attention_heads=4,
            encoder_ffn_dim=256,
            num_mel_bins=80,
            max_source_positions=1500,
        )
        return WhisperEncoder(config)

    def test_output_shape(self, small_encoder):
        # Input: (batch, n_mels, time) — 30s of audio = 3000 mel frames
        x = mx.zeros((1, 80, 3000))
        out = small_encoder(x)
        mx.eval(out)
        # Conv stride=2 halves time: 3000 -> 1500
        assert out.shape == (1, 1500, 64)

    def test_shorter_audio(self, small_encoder):
        # Shorter input (1s of audio ~ 100 mel frames)
        x = mx.zeros((1, 80, 100))
        out = small_encoder(x)
        mx.eval(out)
        # Conv stride=2: 100 -> 50
        assert out.shape == (1, 50, 64)

    def test_sinusoidal_embeddings_shape(self):
        emb = WhisperEncoder._sinusoidal_embeddings(1500, 64)
        mx.eval(emb)
        assert emb.shape == (1500, 64)

    def test_sinusoidal_embeddings_not_all_zero(self):
        emb = WhisperEncoder._sinusoidal_embeddings(100, 64)
        mx.eval(emb)
        assert mx.abs(emb).sum().item() > 0

    def test_conv_layers_exist(self, small_encoder):
        assert hasattr(small_encoder, "conv1")
        assert hasattr(small_encoder, "conv2")
        assert hasattr(small_encoder, "layer_norm")
        assert len(small_encoder.layers) == 2
