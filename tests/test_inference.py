"""Tests for inference module (n-gram blocking, utilities)."""

from mlx_meralion.inference import (
    NO_REPEAT_NGRAM_SIZE,
    NgramBlocker,
    is_converted_dir,
    is_raw_hf_dir,
)


class TestNgramBlocker:
    """Tests for NgramBlocker."""

    def test_default_ngram_size(self):
        assert NO_REPEAT_NGRAM_SIZE == 6

    def test_no_repeat_initially(self):
        blocker = NgramBlocker(3)
        assert blocker.add_and_check(5) is False
        assert blocker.add_and_check(5) is False

    def test_detects_repeated_trigram(self):
        blocker = NgramBlocker(3)
        # Generate [1, 2, 3] — records trigram (1,2)->3
        for t in [1, 2, 3]:
            assert blocker.add_and_check(t) is False
        # Generate [1, 2] again — no repeat yet
        assert blocker.add_and_check(1) is False
        assert blocker.add_and_check(2) is False
        # Next token 3 would complete repeat of (1,2)->3
        assert blocker.add_and_check(3) is True

    def test_same_token_repeat(self):
        blocker = NgramBlocker(3)
        # [5, 5, 5] — records (5,5)->5
        assert blocker.add_and_check(5) is False
        assert blocker.add_and_check(5) is False
        assert blocker.add_and_check(5) is False
        # 4th 5: context is (5,5) and 5 is banned
        assert blocker.add_and_check(5) is True

    def test_different_context_allowed(self):
        blocker = NgramBlocker(3)
        for t in [1, 2, 3]:
            blocker.add_and_check(t)
        # Context (2,3), next=4 is fine
        assert blocker.add_and_check(4) is False

    def test_ngram_size_2(self):
        blocker = NgramBlocker(2)
        assert blocker.add_and_check(5) is False
        assert blocker.add_and_check(7) is False  # records (5)->7
        assert blocker.add_and_check(5) is False  # records (7)->5
        assert blocker.add_and_check(7) is True   # (5)->7 already seen

    def test_popped_token_not_in_list(self):
        blocker = NgramBlocker(2)
        blocker.add_and_check(1)
        blocker.add_and_check(2)  # records (1)->2
        blocker.add_and_check(1)  # records (2)->1
        # (1)->2 is banned
        assert blocker.add_and_check(2) is True
        # Token 2 should NOT be in id_list (it was popped)
        assert blocker.id_list == [1, 2, 1]


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
