# CHANGELOG


## v0.1.1 (2026-03-31)

### Bug Fixes

- Enable n-gram blocking for both greedy and sampling decoding
- Revert to sampler-based n-gram blocking for correct WER (NgramBlocker early-stop caused 40-75% WER regression)

### Chores

- Remove MIT license

### Continuous Integration

- Add CodeQL, Bandit, pip-audit, dependency review workflows
- Add semantic release and PyPI publish workflows
- Fix ruff format with latest ruff 0.15.8

### Documentation

- Add CI and security badges to README

### Testing

- Add 60 unit tests for all package modules


## v0.1.0 (2026-03-31)

### Features

- Initial release of mlx-meralion inference package
- Whisper encoder + MLP adaptor + Gemma2 decoder pipeline
- N-gram blocking always enabled (matches HF generation_config)
- Smart chunking for long audio (30s splits, short tail merging)
- Auto-download from HuggingFace repos via load_model(repo_id)
- High-level API: load_model() + transcribe()
- CLI: mlx-meralion --model <repo> --audio <file> --task asr
