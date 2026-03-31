#!/usr/bin/env python3
"""Benchmark: sequential transcribe() vs batch_transcribe().

Usage:
    python tests/benchmark_batch.py --model MERaLiON/MERaLiON-2-10B-MLX
    python tests/benchmark_batch.py --model MERaLiON/MERaLiON-2-3B-MLX --batch-sizes 2,4
    python tests/benchmark_batch.py --audio-dir /path/to/wavs --num-audios 8

Generates TTS audio via macOS `say` (or falls back to synthetic sine waves),
runs both sequential and batch paths, validates output correctness,
and reports wall-clock time and throughput.
"""

import argparse
import os
import subprocess
import sys
import tempfile
import time

import numpy as np


# Sentences for TTS generation — varied content for distinguishable outputs
TTS_SENTENCES = [
    "Hello, this is a test of the speech recognition system.",
    "The quick brown fox jumps over the lazy dog near the river.",
    "Machine learning models can process audio signals efficiently.",
    "Singapore is a beautiful city known for its diverse culture.",
    "Artificial intelligence is transforming how we interact with technology.",
    "The weather today is sunny with a chance of afternoon showers.",
    "Please remember to submit your report before the deadline tomorrow.",
    "Natural language processing enables computers to understand human speech.",
    "Deep learning has revolutionized computer vision and speech recognition.",
    "The research team published their findings in a leading journal.",
    "Batch processing allows us to handle multiple requests simultaneously.",
    "The conference will be held next week at the convention center.",
]


def generate_tts_audio(sentence: str, wav_path: str, sample_rate: int = 16000) -> bool:
    """Generate a WAV file using macOS say. Returns True on success."""
    try:
        subprocess.run(
            ["say", "-o", wav_path, f"--data-format=LEI16@{sample_rate}", sentence],
            check=True,
            capture_output=True,
            timeout=30,
        )
        return os.path.getsize(wav_path) > 1000
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


def load_wav_as_float32(wav_path: str) -> np.ndarray:
    """Load a 16-bit WAV file as float32 array."""
    from scipy.io import wavfile
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sr, data = wavfile.read(wav_path)
    if data.dtype == np.int16:
        return data.astype(np.float32) / 32768.0
    return data.astype(np.float32)


def generate_synthetic_audio(
    duration_sec: float, freq: float = 440.0, sample_rate: int = 16000
) -> np.ndarray:
    """Fallback: generate a sine wave as synthetic test audio."""
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), dtype=np.float32)
    return 0.5 * np.sin(2 * np.pi * freq * t)


def prepare_audios(num_audios: int, audio_dir: str | None = None) -> list[np.ndarray]:
    """Prepare test audio arrays: TTS if available, else synthetic."""
    if audio_dir:
        # Load existing WAV files from directory
        wavs = sorted(f for f in os.listdir(audio_dir) if f.endswith(".wav"))[:num_audios]
        if not wavs:
            print(f"ERROR: No .wav files found in {audio_dir}")
            sys.exit(1)
        audios = [load_wav_as_float32(os.path.join(audio_dir, w)) for w in wavs]
        print(f"Loaded {len(audios)} WAV files from {audio_dir}")
        return audios

    # Try TTS first
    tmpdir = tempfile.mkdtemp(prefix="bench_audio_")
    audios = []
    use_tts = generate_tts_audio("test", os.path.join(tmpdir, "probe.wav"))

    if use_tts:
        print(f"Generating {num_audios} TTS audio samples...")
        for i in range(num_audios):
            sentence = TTS_SENTENCES[i % len(TTS_SENTENCES)]
            wav_path = os.path.join(tmpdir, f"audio_{i}.wav")
            if generate_tts_audio(sentence, wav_path):
                audios.append(load_wav_as_float32(wav_path))
            else:
                # Fallback for this sample
                audios.append(generate_synthetic_audio(5.0, freq=220 + i * 110))
        durations = [len(a) / 16000 for a in audios]
        print(f"  Durations: {[f'{d:.1f}s' for d in durations]}")
    else:
        print(f"macOS TTS not available, using synthetic sine waves (limited decode output)")
        freqs = [220 + i * 110 for i in range(num_audios)]
        audios = [generate_synthetic_audio(5.0, freq=f) for f in freqs]
        print(f"  Generated {num_audios} sine waves @ {freqs[:5]}{'...' if len(freqs) > 5 else ''}")

    return audios


def run_sequential(model, audios, task="asr", max_new_tokens=64):
    """Run transcribe() sequentially for each audio."""
    from mlx_meralion import transcribe

    results = []
    for audio in audios:
        text = transcribe(model, audio, task=task, max_new_tokens=max_new_tokens, verbose=False)
        results.append(text)
    return results


def run_batch(model, audios, task="asr", max_new_tokens=64):
    """Run batch_transcribe() for all audios at once."""
    from mlx_meralion import batch_transcribe

    return batch_transcribe(model, audios, task=task, max_new_tokens=max_new_tokens, verbose=False)


def validate_outputs(seq_results, batch_results, label=""):
    """Check that batch outputs match sequential outputs."""
    ok = True
    for i, (s, b) in enumerate(zip(seq_results, batch_results)):
        if s != b:
            print(f"  MISMATCH {label} audio {i}:")
            print(f"    seq:   {s[:120]}")
            print(f"    batch: {b[:120]}")
            ok = False
    return ok


def benchmark(
    model_id: str,
    num_audios: int = 8,
    max_new_tokens: int = 128,
    batch_sizes: list[int] | None = None,
    audio_dir: str | None = None,
):
    from mlx_meralion import load_model

    print(f"Loading model: {model_id}")
    t0 = time.time()
    model = load_model(model_id, verbose=True)
    print(f"Model loaded in {time.time() - t0:.1f}s\n")

    # Prepare audio
    audios = prepare_audios(num_audios, audio_dir)
    print()

    # Warmup
    print("Warming up (1 inference)...")
    from mlx_meralion import transcribe
    warmup_text = transcribe(model, audios[0], task="asr", max_new_tokens=16, verbose=False)
    print(f"  Warmup output: {warmup_text[:100]}")
    print("Warmup done.\n")

    if batch_sizes is None:
        batch_sizes = [4, 8] if num_audios >= 8 else [num_audios]

    # --- Sequential baseline (always use max batch size count) ---
    n_seq = max(batch_sizes)
    if n_seq > len(audios):
        n_seq = len(audios)
    test_audios = audios[:n_seq]

    print(f"Running sequential ({n_seq} audios, max_tokens={max_new_tokens})...")
    t0 = time.time()
    seq_results = run_sequential(model, test_audios, max_new_tokens=max_new_tokens)
    seq_time = time.time() - t0
    seq_throughput = n_seq / seq_time

    # Sanity check: sequential outputs
    for i, r in enumerate(seq_results):
        if not r.strip():
            print(f"  WARNING: sequential audio {i} produced empty output!")

    print(f"Sequential done: {seq_time:.2f}s, {seq_throughput:.2f} audios/s\n")

    # --- Batch runs + correctness validation ---
    print("=" * 78)
    print(f"{'Method':<25} {'B':>3} {'Time(s)':>8} {'Aud/s':>7} {'Speedup':>8}  Correct?")
    print("=" * 78)
    print(f"{'Sequential (for-loop)':<25} {n_seq:>3} {seq_time:>8.2f} {seq_throughput:>7.2f} {'1.00x':>8}  ---")

    all_correct = True
    for bs in batch_sizes:
        if bs > len(audios):
            continue
        batch_audios = audios[:bs]

        t0 = time.time()
        batch_results = run_batch(model, batch_audios, max_new_tokens=max_new_tokens)
        batch_time = time.time() - t0
        batch_throughput = bs / batch_time

        # Speedup = throughput ratio
        speedup = batch_throughput / seq_throughput

        # Correctness: compare with sequential results for same audios
        correct = validate_outputs(seq_results[:bs], batch_results, label=f"B={bs}")
        status = "PASS" if correct else "FAIL"
        if not correct:
            all_correct = False

        print(f"{'batch_transcribe':<25} {bs:>3} {batch_time:>8.2f} {batch_throughput:>7.2f} {speedup:>7.2f}x  {status}")

    print("=" * 78)

    # --- Output samples ---
    print(f"\nSample outputs (first 200 chars):")
    for i in range(min(3, n_seq)):
        print(f"  [{i}] seq:   {seq_results[i][:200]}")
    # Reuse last batch results for display
    if batch_sizes:
        bs = min(batch_sizes[-1], len(audios))
        last_batch = run_batch(model, audios[:bs], max_new_tokens=max_new_tokens)
        for i in range(min(3, bs)):
            print(f"  [{i}] batch: {last_batch[i][:200]}")

    if not all_correct:
        print("\nERROR: Some batch outputs do not match sequential outputs!")
        sys.exit(1)
    else:
        print("\nAll correctness checks PASSED.")


def main():
    parser = argparse.ArgumentParser(description="Benchmark sequential vs batch inference")
    parser.add_argument(
        "--model",
        type=str,
        default="MERaLiON/MERaLiON-2-10B-MLX",
        help="Model path or HuggingFace repo ID",
    )
    parser.add_argument("--num-audios", type=int, default=8, help="Number of test audios")
    parser.add_argument("--max-tokens", type=int, default=128, help="Max tokens to generate")
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default=None,
        help="Comma-separated batch sizes to test (e.g. 4,8)",
    )
    parser.add_argument(
        "--audio-dir",
        type=str,
        default=None,
        help="Directory of .wav files to use instead of generating audio",
    )
    args = parser.parse_args()

    batch_sizes = None
    if args.batch_sizes:
        batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    benchmark(
        model_id=args.model,
        num_audios=args.num_audios,
        max_new_tokens=args.max_tokens,
        batch_sizes=batch_sizes,
        audio_dir=args.audio_dir,
    )


if __name__ == "__main__":
    main()
