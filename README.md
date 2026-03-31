# mlx-meralion

[![CI](https://github.com/SiruiHe/mlx-audiollm/actions/workflows/ci.yml/badge.svg)](https://github.com/SiruiHe/mlx-audiollm/actions/workflows/ci.yml)

MLX-native inference for [MERaLiON AudioLLM](https://huggingface.co/MERaLiON) on Apple Silicon.

MERaLiON is A\*STAR's multimodal audio-language model for speech transcription, translation, spoken question answering, and more.

## Installation

```bash
pip install mlx-meralion
```

Requires macOS on Apple Silicon (M1/M2/M3/M4) and Python 3.10+.

## Quick Start

### Python API

```python
from mlx_meralion import load_model, transcribe

# Load model (auto-downloads from HuggingFace on first use)
model = load_model("MERaLiON/MERaLiON-2-10B-MLX")  # 10B 8-bit, recommended
# model = load_model("MERaLiON/MERaLiON-2-3B-MLX")   # 3B fp16, smaller

# Transcribe speech
text = transcribe(model, "audio.wav")
print(text)

# Translate to Chinese
text = transcribe(model, "audio.wav", task="translate_zh")

# Spoken question answering
text = transcribe(model, "audio.wav", task="sqa", question="What is the speaker talking about?")

# Summarize dialogue
text = transcribe(model, "audio.wav", task="summarize")
```

### Batch Inference

Process multiple audio files with GPU-batched decoding for higher throughput:

```python
from mlx_meralion import load_model, batch_transcribe

model = load_model("MERaLiON/MERaLiON-2-10B-MLX")

results = batch_transcribe(model, ["a.wav", "b.wav", "c.wav", "d.wav"])
for text in results:
    print(text)
```

### CLI

```bash
# ASR (default task)
mlx-meralion --model MERaLiON/MERaLiON-2-10B-MLX --audio audio.wav --task asr

# Translation
mlx-meralion --model MERaLiON/MERaLiON-2-10B-MLX --audio audio.wav --task translate_zh

# Custom instruction
mlx-meralion --model MERaLiON/MERaLiON-2-10B-MLX --audio audio.wav --instruction "Summarize this in one sentence."
```

## Supported Tasks

| Task | Description |
|------|-------------|
| `asr` | Speech-to-text transcription |
| `translate_zh` | Translate to Chinese |
| `translate_id` | Translate to Indonesian |
| `translate_ms` | Translate to Malay |
| `translate_ta` | Translate to Tamil |
| `sqa` | Spoken question answering (requires `question=`) |
| `summarize` | Dialogue summarization |
| `paralinguistics` | Speaker characteristic analysis |

## Available Models

| Model | Size | RAM | Quality | HuggingFace |
|-------|------|-----|---------|-------------|
| MERaLiON-2-10B-MLX | ~10 GB | 16+ GB | Best | [MERaLiON/MERaLiON-2-10B-MLX](https://huggingface.co/MERaLiON/MERaLiON-2-10B-MLX) |
| MERaLiON-2-3B-MLX | ~6 GB | 8+ GB | Good | [MERaLiON/MERaLiON-2-3B-MLX](https://huggingface.co/MERaLiON/MERaLiON-2-3B-MLX) |

## Batch Inference Benchmark

Measured on Apple M4 Max with MERaLiON-2-10B-MLX, 8 TTS-generated audio samples (~25s each), max 256 tokens:

| Method | B | Time | Throughput | Speedup | Correct |
|--------|---|------|------------|---------|---------|
| Sequential (for-loop) | 8 | 38.96s | 0.21 aud/s | 1.00x | --- |
| `batch_transcribe` | 4 | 13.93s | 0.29 aud/s | 1.40x | PASS |
| `batch_transcribe` | 8 | 23.25s | 0.34 aud/s | 1.68x | PASS |

Correctness is validated by comparing batch outputs against sequential outputs token-for-token. The benchmark script (`tests/benchmark_batch.py`) exits with a non-zero code if any mismatch is detected.

The decoder-only speedup is ~2.2x at B=4. Overall speedup is bounded by the Whisper encoder, which is compute-bound and does not benefit from batching (Amdahl's law). Larger batch sizes and longer generated sequences yield higher speedup.

## Features

- **Apple Silicon native**: Runs entirely on MLX with GPU acceleration
- **Batch inference**: GPU-batched decoding via `BatchKVCache` with left-padding alignment
- **N-gram blocking**: Automatically prevents repetitive output (matching HuggingFace quality)
- **Smart chunking**: Long audio split at 30s boundaries; short tails merged to prevent hallucination
- **Auto-download**: HuggingFace models are downloaded and cached automatically
- **Multiple tasks**: ASR, translation, QA, summarization, and more

## Architecture

```
Audio (WAV/MP3/FLAC)
  -> Whisper Encoder (1280-d)
    -> LayerNorm + MLP Adaptor
      -> Speech embeddings merged into text sequence
        -> Gemma2 Decoder -> text output
```

### Batch Inference Pipeline

```
Multiple audios
  -> Mel features extracted per audio
    -> All chunks batched through Whisper encoder in one forward pass
      -> Per-audio embedding merge (speech + text tokens)
        -> Left-pad to uniform length
          -> BatchKVCache prefill (embeddings baked into KV cache)
            -> GPU-batched autoregressive decode (token IDs only)
```
