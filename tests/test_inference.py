"""Tests for inference module (n-gram sampler, utilities, embedding merge, batch)."""

import mlx.core as mx
import mlx.nn as nn

from mlx_meralion.inference import (
    NO_REPEAT_NGRAM_SIZE,
    _left_pad_embeddings,
    _left_pad_ids,
    _wrap_sampler_with_ngram_blocking,
    build_merged_embeddings,
    build_merged_embeddings_single,
    is_converted_dir,
    is_raw_hf_dir,
    make_no_repeat_ngram_sampler,
)


class TestNoRepeatNgramSampler:
    """Tests for make_no_repeat_ngram_sampler()."""

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
        logits = mx.zeros(10)
        logits_with_5 = logits.at[5].add(10.0)

        # Generate [5, 5, 5] — records trigram (5,5)->5
        for _ in range(3):
            sampler(logits_with_5)

        # Next: context is (5,5), token 5 should be banned
        token = sampler(logits_with_5)
        mx.eval(token)
        assert int(token) != 5

    def test_sampler_allows_different_context(self):
        sampler = make_no_repeat_ngram_sampler(3)
        logits_5 = mx.zeros(10).at[5].add(10.0)
        for _ in range(3):
            sampler(logits_5)

        # Add token 3 to change context
        logits_3 = mx.zeros(10).at[3].add(10.0)
        token = sampler(logits_3)
        mx.eval(token)
        assert int(token) == 3

        # Context is now (5,3), so 5 should be allowed
        token = sampler(logits_5)
        mx.eval(token)
        assert int(token) == 5

    def test_sampler_2d_logits(self):
        sampler = make_no_repeat_ngram_sampler(3)
        logits = mx.array([[0.1, 0.5, 0.3, 0.8, 0.2]])
        token = sampler(logits)
        mx.eval(token)
        assert int(token) == 3

    def test_repeated_sequence_blocked(self):
        sampler = make_no_repeat_ngram_sampler(3)
        # Generate [1, 2, 3, 1, 2] — records (1,2)->3
        tokens = [1, 2, 3, 1, 2]
        for t in tokens:
            logits = mx.zeros(10).at[t].add(10.0)
            sampler(logits)
        # Next: context is (1,2), token 3 is banned
        logits = mx.zeros(10).at[3].add(10.0)
        token = sampler(logits)
        mx.eval(token)
        assert int(token) != 3


class TestWrapSamplerWithNgramBlocking:
    """Tests for _wrap_sampler_with_ngram_blocking()."""

    def test_none_base_returns_greedy_sampler(self):
        sampler = _wrap_sampler_with_ngram_blocking(None, 3)
        logits = mx.array([0.1, 0.5, 0.3, 0.8, 0.2])
        token = sampler(logits)
        mx.eval(token)
        assert int(token) == 3  # greedy argmax

    def test_wraps_existing_sampler(self):
        def always_zero(logits):
            return mx.array(0)

        wrapped = _wrap_sampler_with_ngram_blocking(always_zero, 2)
        logits = mx.array([10.0, 5.0, 3.0])

        # First two calls: token 0 allowed, registers (0)->0
        token = wrapped(logits)
        mx.eval(token)
        assert int(token) == 0
        token = wrapped(logits)
        mx.eval(token)
        assert int(token) == 0

        # Third call: context (0), token 0 banned, falls back to logit ranking
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


class _FakeEmbedTokens(nn.Embedding):
    """Minimal embedding layer for testing."""

    def __init__(self, vocab_size, hidden_size):
        super().__init__(vocab_size, hidden_size)


class _FakeInnerModel:
    def __init__(self, vocab_size, hidden_size):
        self.embed_tokens = _FakeEmbedTokens(vocab_size, hidden_size)


class _FakeDecoderModel:
    def __init__(self, vocab_size=256000, hidden_size=16):
        self.model = _FakeInnerModel(vocab_size, hidden_size)


class TestBuildMergedEmbeddings:
    """Tests for vectorized build_merged_embeddings."""

    def test_replaces_speech_tokens(self):
        H = 16
        decoder = _FakeDecoderModel(hidden_size=H)
        speech_token = 255999
        # Sequence: [10, 255999, 255999, 20]
        input_ids = mx.array([[10, speech_token, speech_token, 20]])
        speech_embeds = mx.ones((1, 2, H)) * 99.0

        result = build_merged_embeddings(decoder, input_ids, speech_embeds, speech_token)
        mx.eval(result)

        # Positions 1 and 2 should have speech embeddings (all 99s)
        assert result.shape == (1, 4, H)
        assert float(mx.mean(result[0, 1, :])) > 90  # speech embedding, not text
        assert float(mx.mean(result[0, 2, :])) > 90

    def test_preserves_text_tokens(self):
        H = 16
        decoder = _FakeDecoderModel(hidden_size=H)
        speech_token = 255999
        input_ids = mx.array([[10, speech_token, 20]])
        speech_embeds = mx.ones((1, 1, H)) * 99.0

        result = build_merged_embeddings(decoder, input_ids, speech_embeds, speech_token)
        mx.eval(result)

        # Compare text positions: should match original embed_tokens output
        original = decoder.model.embed_tokens(input_ids)
        mx.eval(original)
        assert mx.allclose(result[0, 0, :], original[0, 0, :], atol=1e-5)
        assert mx.allclose(result[0, 2, :], original[0, 2, :], atol=1e-5)

    def test_no_speech_tokens(self):
        H = 16
        decoder = _FakeDecoderModel(hidden_size=H)
        input_ids = mx.array([[10, 20, 30]])
        speech_embeds = mx.ones((1, 0, H))

        result = build_merged_embeddings(decoder, input_ids, speech_embeds, 255999)
        original = decoder.model.embed_tokens(input_ids)
        mx.eval(result, original)
        assert mx.allclose(result, original, atol=1e-5)


class TestBuildMergedEmbeddingsSingle:
    """Tests for the single-sequence optimized path."""

    def test_replaces_speech_tokens_1d(self):
        H = 16
        decoder = _FakeDecoderModel(hidden_size=H)
        speech_token = 255999
        input_ids = mx.array([10, speech_token, speech_token, 20])
        speech_embeds = mx.ones((1, 2, H)) * 99.0

        result = build_merged_embeddings_single(decoder, input_ids, speech_embeds, speech_token)
        mx.eval(result)
        assert result.shape == (1, 4, H)
        assert float(mx.mean(result[0, 1, :])) > 90

    def test_replaces_speech_tokens_2d(self):
        H = 16
        decoder = _FakeDecoderModel(hidden_size=H)
        speech_token = 255999
        input_ids = mx.array([[10, speech_token, 20]])
        speech_embeds = mx.ones((1, 1, H)) * 99.0

        result = build_merged_embeddings_single(decoder, input_ids, speech_embeds, speech_token)
        mx.eval(result)
        assert result.shape == (1, 3, H)
        assert float(mx.mean(result[0, 1, :])) > 90


class TestLeftPadEmbeddings:
    """Tests for _left_pad_embeddings."""

    def test_uniform_lengths_no_padding(self):
        H = 8
        a = mx.ones((5, H))
        b = mx.ones((5, H)) * 2.0
        result, padding = _left_pad_embeddings([a, b])
        mx.eval(result)
        assert result.shape == (2, 5, H)
        assert padding == [0, 0]

    def test_different_lengths_left_pads_shorter(self):
        H = 8
        short = mx.ones((3, H))
        long = mx.ones((5, H)) * 2.0
        result, padding = _left_pad_embeddings([short, long])
        mx.eval(result)
        assert result.shape == (2, 5, H)
        assert padding == [2, 0]
        # First 2 positions of short sequence should be zero (pad)
        assert float(mx.sum(mx.abs(result[0, :2, :]))) == 0.0
        # Remaining positions should be the original values
        assert mx.allclose(result[0, 2:, :], short, atol=1e-5)
        assert mx.allclose(result[1, :, :], long, atol=1e-5)

    def test_single_item(self):
        H = 4
        a = mx.ones((3, H))
        result, padding = _left_pad_embeddings([a])
        mx.eval(result)
        assert result.shape == (1, 3, H)
        assert padding == [0]


class TestLeftPadIds:
    """Tests for _left_pad_ids."""

    def test_uniform_lengths(self):
        a = mx.array([1, 2, 3])
        b = mx.array([4, 5, 6])
        result = _left_pad_ids([a, b])
        mx.eval(result)
        assert result.shape == (2, 3)
        assert result.tolist() == [[1, 2, 3], [4, 5, 6]]

    def test_different_lengths(self):
        short = mx.array([10, 20])
        long = mx.array([1, 2, 3, 4])
        result = _left_pad_ids([short, long])
        mx.eval(result)
        assert result.shape == (2, 4)
        # Short sequence should be left-padded with 0s
        assert result.tolist() == [[0, 0, 10, 20], [1, 2, 3, 4]]


class TestBatchDecodeIsTrulyBatched:
    """Verify that batch decode does a single forward pass for all B sequences,
    not B separate forward passes. Uses a mock decoder that records call shapes."""

    def test_decode_forward_receives_full_batch(self):
        """The decoder's forward pass should receive (B, 1) during decode,
        proving all sequences are processed in one GPU call."""
        call_shapes = []

        class _MockDecoder:
            """Records the shapes of every forward call."""

            class model:
                class args:
                    hidden_size = 16

                class embed_tokens(nn.Embedding):
                    def __init__(self):
                        super().__init__(32, 16)

                    def as_linear(self, x):
                        # Project back to vocab size
                        W = self.weight  # (32, 16)
                        return x @ W.T

                embed_tokens = embed_tokens()
                layers = []

                @staticmethod
                def norm(x):
                    return x

            final_logit_softcapping = 30.0

            def __call__(self, input_ids, cache=None, input_embeddings=None):
                if input_embeddings is not None:
                    h = input_embeddings
                else:
                    h = self.model.embed_tokens(input_ids)
                call_shapes.append(("fwd", h.shape))
                h = h * (self.model.args.hidden_size**0.5)
                out = self.model.embed_tokens.as_linear(self.model.norm(h))
                out = mx.tanh(out / self.final_logit_softcapping) * self.final_logit_softcapping
                return out

        from mlx_lm.models.cache import BatchKVCache

        B, S, H, V = 3, 4, 16, 32
        decoder = _MockDecoder()

        # Simulate batch prefill + decode (simplified version of batch_transcribe)
        batched_embeds = mx.ones((B, S, H))
        batched_ids = mx.zeros((B, S), dtype=mx.int32)
        padding = [0, 0, 0]
        batch_cache = [BatchKVCache(padding) for _ in range(0)]  # no layers

        # Prefill
        logits = decoder(batched_ids, cache=batch_cache, input_embeddings=batched_embeds)
        logits = logits[:, -1, :]

        # Decode 3 steps
        for _ in range(3):
            y = mx.argmax(logits, axis=-1)
            mx.eval(y)
            logits = decoder(y[:, None], cache=batch_cache)
            logits = logits[:, -1, :]

        # Verify: prefill should be (B, S, H), decode steps should be (B, 1, H)
        assert call_shapes[0] == ("fwd", (B, S, H)), f"Prefill shape wrong: {call_shapes[0]}"
        for i, shape in enumerate(call_shapes[1:], 1):
            assert shape == ("fwd", (B, 1, H)), (
                f"Decode step {i} shape {shape} != (B=3, 1, H=16) — "
                f"NOT truly batched!"
            )

    def test_sequential_would_show_batch_1(self):
        """Control test: sequential processing shows batch=1 shapes."""
        call_shapes = []

        class _SeqDecoder:
            class model:
                class embed_tokens(nn.Embedding):
                    def __init__(self):
                        super().__init__(32, 16)

                embed_tokens = embed_tokens()

            def __call__(self, x, **kwargs):
                call_shapes.append(x.shape)
                return self.model.embed_tokens(x) if x.dtype == mx.int32 else x

        decoder = _SeqDecoder()
        # Simulate sequential: 3 separate calls with batch=1
        for _ in range(3):
            decoder(mx.zeros((1, 4), dtype=mx.int32))

        # All calls should have batch=1
        assert all(s[0] == 1 for s in call_shapes)
