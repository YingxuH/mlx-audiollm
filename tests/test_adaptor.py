"""Tests for MERaLiON MLP adaptor module."""

import mlx.core as mx

from mlx_meralion.adaptor import (
    MERaLiONSpeechAudioAdapter,
    MERaLiONSpeechAudioAdapterLarge,
    create_adapter,
)


class TestMERaLiONSpeechAudioAdapter:
    """Tests for v1 simple MLP adaptor."""

    def test_output_shape(self):
        adaptor = MERaLiONSpeechAudioAdapter(
            speech_hidden_size=1280,
            text_hidden_size=3584,
            scale_factor=15,
        )
        # Whisper encoder output: (batch, 1500, 1280)
        x = mx.zeros((1, 1500, 1280))
        out = adaptor(x)
        mx.eval(out)
        # 1500 / 15 = 100 timesteps
        assert out.shape == (1, 100, 3584)

    def test_batch_dimension(self):
        adaptor = MERaLiONSpeechAudioAdapter(
            speech_hidden_size=1280,
            text_hidden_size=3584,
            scale_factor=15,
        )
        x = mx.zeros((4, 1500, 1280))
        out = adaptor(x)
        mx.eval(out)
        assert out.shape == (4, 100, 3584)

    def test_truncates_to_scale_factor_multiple(self):
        adaptor = MERaLiONSpeechAudioAdapter(
            speech_hidden_size=64,
            text_hidden_size=128,
            scale_factor=15,
        )
        # 1507 is not divisible by 15; floor(1507/15)*15 = 1500
        x = mx.zeros((1, 1507, 64))
        out = adaptor(x)
        mx.eval(out)
        assert out.shape == (1, 100, 128)

    def test_small_dimensions(self):
        adaptor = MERaLiONSpeechAudioAdapter(
            speech_hidden_size=64,
            text_hidden_size=128,
            scale_factor=4,
        )
        x = mx.zeros((1, 20, 64))
        out = adaptor(x)
        mx.eval(out)
        assert out.shape == (1, 5, 128)


class TestMERaLiONSpeechAudioAdapterLarge:
    """Tests for v2 gated MLP adaptor."""

    def test_output_shape(self):
        adaptor = MERaLiONSpeechAudioAdapterLarge(
            speech_hidden_size=1280,
            text_hidden_size=3584,
            scale_factor=15,
        )
        x = mx.zeros((1, 1500, 1280))
        out = adaptor(x)
        mx.eval(out)
        assert out.shape == (1, 100, 3584)

    def test_has_gate_proj(self):
        adaptor = MERaLiONSpeechAudioAdapterLarge(
            speech_hidden_size=64,
            text_hidden_size=128,
            scale_factor=4,
        )
        assert hasattr(adaptor, "gate_proj")
        assert hasattr(adaptor, "pool_proj")
        assert hasattr(adaptor, "out_proj")


class TestCreateAdapter:
    """Tests for create_adapter() factory."""

    def test_v1_variant(self):
        adaptor = create_adapter("v1", 64, 128, 4)
        assert isinstance(adaptor, MERaLiONSpeechAudioAdapter)

    def test_v2_variant(self):
        adaptor = create_adapter("v2", 64, 128, 4)
        assert isinstance(adaptor, MERaLiONSpeechAudioAdapterLarge)

    def test_large_variant(self):
        adaptor = create_adapter("large", 64, 128, 4)
        assert isinstance(adaptor, MERaLiONSpeechAudioAdapterLarge)

    def test_meralion2_variant(self):
        adaptor = create_adapter("meralion2", 64, 128, 4)
        assert isinstance(adaptor, MERaLiONSpeechAudioAdapterLarge)

    def test_unknown_defaults_to_v1(self):
        adaptor = create_adapter("unknown", 64, 128, 4)
        assert isinstance(adaptor, MERaLiONSpeechAudioAdapter)
