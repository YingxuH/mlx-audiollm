# mlx-meralion

MLX-native inference for [MERaLiON AudioLLM](https://huggingface.co/MERaLiON) on Apple Silicon.

MERaLiON is A*STAR's multimodal audio-language model for speech transcription, translation, spoken question answering, and more.

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

## Features

- **Apple Silicon native**: Runs entirely on MLX with GPU acceleration
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
