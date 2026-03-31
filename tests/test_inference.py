"""Tests for inference module (n-gram blocking, utilities)."""

import mlx.core as mx

from mlx_meralion.inference import (
    NO_REPEAT_NGRAM_SIZE,
    _make_ngram_logits_processor,
    is_converted_dir,
    is_raw_hf_dir,
)


class TestNgramLogitsProcessor:
    """Tests for _make_ngram_logits_processor()."""

    def test_default_ngram_size(self):
        assert NO_REPEAT_NGRAM_SIZE == 6

    def test_processor_returns_logits_unchanged_initially(self):
        processor, id_list = _make_ngram_logits_processor(3)
        logits = mx.array([0.1, 0.5, 0.3, 0.8, 0.2])
        result = processor(logits)
        mx.eval(result)
        # No tokens tracked yet, logits should be unchanged
        assert result.shape == logits.shape

    def test_processor_masks_banned_token(self):
        processor, id_list = _make_ngram_logits_processor(3)
        # Simulate generating [5, 5, 5] — after this, prefix (5,5)->5 is banned
        id_list.extend([5, 5, 5])
        # Call processor to update ngram table, then call again to apply ban
        logits = mx.array([0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0])
        _ = processor(logits)  # updates table
        # Now context is still (5,5), token 5 should be banned
        result = processor(logits)
        mx.eval(result)
        # Token 5 should have very negative logit
        assert float(result[5]) < -1e8

    def test_processor_allows_different_context(self):
        processor, id_list = _make_ngram_logits_processor(3)
        id_list.extend([5, 5, 5])
        _ = processor(mx.zeros(10))  # update table
        # Change context by adding token 3
        id_list.append(3)
        logits = mx.array([0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0])
        result = processor(logits)
        mx.eval(result)
        # Context is (5,3), not (5,5), so 5 should NOT be banned
        assert float(result[5]) > 0

    def test_processor_2d_logits(self):
        processor, id_list = _make_ngram_logits_processor(3)
        logits = mx.array([[0.1, 0.5, 0.3, 0.8, 0.2]])
        result = processor(logits)
        mx.eval(result)
        assert result.shape == (1, 5)

    def test_processor_ngram_size_2(self):
        processor, id_list = _make_ngram_logits_processor(2)
        id_list.extend([2, 2])
        _ = processor(mx.zeros(5))  # update table: prefix (2,)->2
        logits = mx.array([0.0, 0.0, 10.0, 0.0, 0.0])
        result = processor(logits)
        mx.eval(result)
        # Token 2 should be banned (context is (2,), banned={2})
        assert float(result[2]) < -1e8


class TestDirectoryDetection:
    """Tests for model directory detection."""

    def test_is_converted_dir_nonexistent(self, tmp_path):
        assert is_converted_dir(tmp_path) is False

    def test_is_converted_dir_with_encoder_config(self, tmp_path):
        (tmp_path / "encoder_config.json").write_text("{}")
        assert is_converted_dir(tmp_path) is True

    def test_is_raw_hf_dir_nonexistent(self, tmp_path):
        assert is_raw_hf_dir(tmp_path) is False

    def test_is_raw_hf_dir_with_model_files(self, tmp_path):
        (tmp_path / "config.json").write_text("{}")
        (tmp_path / "model-00001.safetensors").write_text("")
        assert is_raw_hf_dir(tmp_path) is True

    def test_is_raw_hf_dir_missing_safetensors(self, tmp_path):
        (tmp_path / "config.json").write_text("{}")
        assert is_raw_hf_dir(tmp_path) is False
