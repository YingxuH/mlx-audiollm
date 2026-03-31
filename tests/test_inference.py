"""Tests for inference module (n-gram sampler, utilities)."""

import mlx.core as mx

from mlx_meralion.inference import (
    NO_REPEAT_NGRAM_SIZE,
    _wrap_sampler_with_ngram_blocking,
    is_converted_dir,
    is_raw_hf_dir,
    make_no_repeat_ngram_sampler,
)


class TestNoRepeatNgramSampler:
    """Tests for the n-gram blocking sampler."""

    def test_default_ngram_size(self):
        assert NO_REPEAT_NGRAM_SIZE == 6

    def test_sampler_returns_token(self):
        sampler = make_no_repeat_ngram_sampler(3)
        logits = mx.array([0.1, 0.5, 0.3, 0.8, 0.2])
        token = sampler(logits)
        mx.eval(token)
        assert int(token) == 3  # argmax

    def test_sampler_blocks_repeated_ngram(self):
        sampler = make_no_repeat_ngram_sampler(3)
        # Create logits where token 5 is always the argmax
        logits = mx.zeros(10)
        logits_with_5 = logits.at[5].add(10.0)

        # Generate tokens: [5, 5, 5] — now the 3-gram (5,5)->5 is recorded
        for _ in range(3):
            sampler(logits_with_5)

        # Next call: context is (5,5), token 5 should be banned
        token = sampler(logits_with_5)
        mx.eval(token)
        assert int(token) != 5

    def test_sampler_allows_different_context(self):
        sampler = make_no_repeat_ngram_sampler(3)
        # Generate [5, 5, 5]
        logits_5 = mx.zeros(10)
        logits_5 = logits_5.at[5].add(10.0)
        for _ in range(3):
            sampler(logits_5)

        # Now generate token 3 (different context)
        logits_3 = mx.zeros(10)
        logits_3 = logits_3.at[3].add(10.0)
        token = sampler(logits_3)
        mx.eval(token)
        assert int(token) == 3

        # Context is now (5, 3), so 5 should be allowed again
        token = sampler(logits_5)
        mx.eval(token)
        assert int(token) == 5

    def test_sampler_2d_logits(self):
        sampler = make_no_repeat_ngram_sampler(3)
        logits = mx.array([[0.1, 0.5, 0.3, 0.8, 0.2]])
        token = sampler(logits)
        mx.eval(token)
        assert int(token) == 3

    def test_ngram_size_2_blocks_bigram_repeats(self):
        sampler = make_no_repeat_ngram_sampler(2)
        logits = mx.zeros(5)
        logits = logits.at[2].add(10.0)

        # First call: token 2. id_list=[2], no prefix registered yet (len < 2)
        token = sampler(logits)
        mx.eval(token)
        assert int(token) == 2

        # Second call: token 2. id_list=[2,2], registers prefix (2,)->2
        token = sampler(logits)
        mx.eval(token)
        assert int(token) == 2

        # Third call: context=(2,), token 2 is banned
        token = sampler(logits)
        mx.eval(token)
        assert int(token) != 2


class TestWrapSamplerWithNgramBlocking:
    """Tests for _wrap_sampler_with_ngram_blocking()."""

    def test_none_base_returns_greedy_sampler(self):
        sampler = _wrap_sampler_with_ngram_blocking(None, 3)
        logits = mx.array([0.1, 0.5, 0.3, 0.8, 0.2])
        token = sampler(logits)
        mx.eval(token)
        assert int(token) == 3  # greedy argmax

    def test_wraps_existing_sampler(self):
        # A dummy sampler that always returns token 0
        def always_zero(logits):
            return mx.array(0)

        wrapped = _wrap_sampler_with_ngram_blocking(always_zero, 2)
        logits = mx.array([10.0, 5.0, 3.0])

        # First call: returns 0. id_list=[0], len < ngram_size, no prefix yet
        token = wrapped(logits)
        mx.eval(token)
        assert int(token) == 0

        # Second call: returns 0. id_list=[0,0], registers prefix (0,)->0
        token = wrapped(logits)
        mx.eval(token)
        assert int(token) == 0

        # Third call: context=(0,), token 0 is banned, falls back to argmax
        token = wrapped(logits)
        mx.eval(token)
        assert int(token) != 0


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
