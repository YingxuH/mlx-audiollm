# CHANGELOG


## v0.1.0 (2026-03-31)

### Bug Fixes

- Enable n-gram blocking for both greedy and sampling decoding
  ([`8e4d9d7`](https://github.com/YingxuH/mlx-audiollm/commit/8e4d9d7347f1f776cd80970833579083b91e4478))

Restore _wrap_sampler_with_ngram_blocking() so n-gram blocking (size=6) is always on regardless of
  temperature. For temp=0 uses greedy sampler with blocking; for temp>0 wraps the temperature
  sampler and rejects banned tokens by falling back to logit ranking.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

- Replace n-gram sampler with logits processor to preserve MLX async pipeline
  ([`1659082`](https://github.com/YingxuH/mlx-audiollm/commit/16590828203bfaed04a390f6b91aaf0151eca849))

The previous n-gram sampler called int() on tokens inside generate_step's hot loop, forcing
  synchronous GPU evaluation on every token and causing 100-1000x slowdown (320s -> 21s for 14
  tokens on 3B model).

New approach: pure MLX logits processor that masks banned tokens with -inf using mx.array indexing.
  Token IDs are fed back from the already-materialized yield values outside the hot loop, so the
  async GPU pipeline is preserved.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

- Revert to sampler-based n-gram blocking for correct WER
  ([`0209a68`](https://github.com/YingxuH/mlx-audiollm/commit/0209a683663d01c6e29906e67f3fa3afaba29a7b))

NgramBlocker (early-stop on repeat) caused 40-75% WER regression vs HF because it truncated output
  instead of substituting tokens. Revert to the original sampler approach which picks the next-best
  token when the greedy choice would repeat a 6-gram — matching HF generate() behavior.

The sampler's int() call is safe because mlx-lm's generate_step already materializes each token
  before yield. Previous slowness was caused by system memory pressure (OOM thrashing), not the
  sampler itself.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

- Use NgramBlocker post-check instead of sampler/logits processor
  ([`3d5df65`](https://github.com/YingxuH/mlx-audiollm/commit/3d5df659263c02498111983bd5df8d9ec0e5a548))

Replace the sampler-based n-gram blocking with a post-generation check that runs on
  already-materialized token IDs. When a repeated n-gram is detected, generation stops early (like
  EOS). This avoids any Python code in the generate_step hot loop, preserving MLX's async GPU
  pipeline.

Previous approaches (sampler with int(), logits processor with mx.array scatter) all introduced
  Python overhead in the generation loop.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

### Chores

- Remove MIT license, delete v0.1.0 tag
  ([`6dfc4a4`](https://github.com/YingxuH/mlx-audiollm/commit/6dfc4a4135c4d06af423e7937943f411e9458a8d))

- Remove LICENSE file - Remove license field from pyproject.toml - Remove MIT classifier and license
  section from README

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

### Continuous Integration

- Add security workflows, semantic release, and PyPI publish
  ([`1290f2b`](https://github.com/YingxuH/mlx-audiollm/commit/1290f2b9b089abb5dcf3efadbb38b0fbe03c4fff))

- CodeQL static analysis (Python + Actions) - Bandit SAST scanning with .bandit.yml config -
  pip-audit dependency vulnerability scanning - Dependency review on PRs - Semantic release workflow
  (python-semantic-release v9) - PyPI publish with SLSA provenance attestation - CI: ruff
  lint/format + pytest on macOS (Py 3.10 + 3.12) + build check

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

- Fix ruff format with latest ruff 0.15.8
  ([`ee87ef7`](https://github.com/YingxuH/mlx-audiollm/commit/ee87ef7a9e34edf40fcf5b3d48f80c8d22c4224e))

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

### Documentation

- Add CI and security badges to README
  ([`63c7ba7`](https://github.com/YingxuH/mlx-audiollm/commit/63c7ba78483ebecb3cc923c96af389ebaec26112))

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

### Features

- Initial release of mlx-meralion inference package
  ([`722d15c`](https://github.com/YingxuH/mlx-audiollm/commit/722d15cd38306d5695e90cd697ca90acdb4d9d11))

MLX-native inference for MERaLiON AudioLLM on Apple Silicon.

- Whisper encoder + MLP adaptor + Gemma2 decoder pipeline - N-gram blocking always enabled (matches
  HF generation_config) - Smart chunking for long audio (30s splits, short tail merging) -
  Auto-download from HuggingFace repos via load_model(repo_id) - High-level API: load_model() +
  transcribe() - CLI: mlx-meralion --model <repo> --audio <file> --task asr

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

### Testing

- Add 60 unit tests for all package modules
  ([`1c554a7`](https://github.com/YingxuH/mlx-audiollm/commit/1c554a75c5ac7d17907c61a57d4e2ad377ae7144))

- test_adaptor: v1/v2 adaptor shapes, create_adapter factory (11 tests) - test_whisper_encoder:
  config, attention, encoder layer, full encoder (12 tests) - test_inference: n-gram sampler
  blocking, wrap sampler, dir detection (13 tests) - test_model: weight partitioning, key remapping,
  config loading (13 tests) - test_processor: task prompts, constants, error handling (11 tests) -
  conftest: shared fixtures for dummy audio data

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
