"""Tests for MERaLiON audio/text processor."""

import pytest

from mlx_meralion.processor import (
    CHUNK_LENGTH,
    FIXED_SPEECH_LENGTH,
    SAMPLE_RATE,
    SPEECH_TOKEN_INDEX,
    TASK_PROMPTS,
    get_task_prompt,
    load_audio,
)


class TestGetTaskPrompt:
    """Tests for get_task_prompt()."""

    def test_asr_prompt(self):
        prompt = get_task_prompt("asr")
        assert prompt == "Please transcribe this speech."

    def test_translate_zh_prompt(self):
        prompt = get_task_prompt("translate_zh")
        assert "Chinese" in prompt

    def test_sqa_with_question(self):
        prompt = get_task_prompt("sqa", question="What language?")
        assert "What language?" in prompt

    def test_all_tasks_valid(self):
        for task in TASK_PROMPTS:
            if task == "sqa":
                prompt = get_task_prompt(task, question="test")
            else:
                prompt = get_task_prompt(task)
            assert isinstance(prompt, str)
            assert len(prompt) > 0

    def test_unknown_task_raises(self):
        with pytest.raises(ValueError, match="Unknown task"):
            get_task_prompt("nonexistent_task")

    def test_sqa_without_question_raises(self):
        with pytest.raises(KeyError):
            get_task_prompt("sqa")


class TestLoadAudio:
    """Tests for load_audio()."""

    def test_load_nonexistent_raises(self):
        with pytest.raises(Exception):
            load_audio("/nonexistent/path/audio.wav")


class TestConstants:
    """Tests for processor constants."""

    def test_sample_rate(self):
        assert SAMPLE_RATE == 16000

    def test_chunk_length(self):
        assert CHUNK_LENGTH == 30

    def test_speech_token_index(self):
        assert SPEECH_TOKEN_INDEX == 255999

    def test_fixed_speech_length(self):
        assert FIXED_SPEECH_LENGTH == 100
