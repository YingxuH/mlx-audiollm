# CHANGELOG

## v0.1.0 (2026-03-31)

### Features

- Initial release of `mlx-meralion` package
- Whisper encoder + MLP adaptor + Gemma2 decoder pipeline
- N-gram blocking always enabled (matches HF `generation_config.json`)
- Smart chunking for long audio (30s splits, short tail merging)
- Auto-download from HuggingFace repos via `load_model(repo_id)`
- High-level API: `load_model()` + `transcribe()`
- CLI: `mlx-meralion --model <repo> --audio <file> --task asr`
- Support for ASR, translation (zh/id/ms/ta), SQA, summarization, paralinguistics
