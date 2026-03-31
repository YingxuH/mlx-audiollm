"""Shared fixtures for mlx-meralion tests."""

import numpy as np
import pytest


@pytest.fixture
def dummy_audio_1s() -> np.ndarray:
    """1 second of random audio at 16kHz."""
    return np.random.randn(16000).astype(np.float32)


@pytest.fixture
def dummy_audio_5s() -> np.ndarray:
    """5 seconds of random audio at 16kHz."""
    return np.random.randn(5 * 16000).astype(np.float32)


@pytest.fixture
def dummy_audio_35s() -> np.ndarray:
    """35 seconds of random audio at 16kHz (triggers multi-chunk)."""
    return np.random.randn(35 * 16000).astype(np.float32)


@pytest.fixture
def dummy_audio_65s() -> np.ndarray:
    """65 seconds of random audio at 16kHz (triggers smart chunking with short tail)."""
    return np.random.randn(65 * 16000).astype(np.float32)


@pytest.fixture
def silence_audio_30s() -> np.ndarray:
    """30 seconds of silence at 16kHz."""
    return np.zeros(30 * 16000, dtype=np.float32)
