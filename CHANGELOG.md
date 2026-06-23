# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed (breaking)
- **Replaced the mock training notebook with a real, fully-wired Tunix GRPO
  loop.** `tunix_reasoning_trainer.ipynb` now uses Tunix's `RLCluster` and
  `GRPOLearner` to actually update Gemma 3 1B LoRA weights, runs on a free-tier
  Google Colab accelerator (T4 GPU or TPU), and supports both GSM8K (default)
  and this project's own dataset via a `DATASET` toggle. Its reward functions
  reuse `reasoning_rewards.py` so the notebook and tests share one
  implementation.
- Retargeted the project from Gemma 2 2B (Keras/Kaggle scaffold) to
  Gemma 3 1B-IT (Tunix/JAX), and updated `requirements.txt` and the README
  accordingly.

### Added
- `reasoning_rewards.py`: dependency-free, documented, and unit-tested
  implementation of the format / correctness / reasoning-quality reward
  functions, reusable outside the notebook.
- `tests/` test suite (pytest) covering the reward functions and the training
  data generator.
- GitHub Actions CI running lint and tests on Python 3.8, 3.10, and 3.12.
- `pyproject.toml` with project metadata, ruff, and pytest configuration.
- `Makefile` with `install`, `test`, `lint`, `format`, `data`, and `clean`
  targets.
- `requirements-dev.txt`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, issue and
  pull-request templates.

### Changed
- Rewrote `setup.sh` into a working local-setup script (virtualenv + dev deps
  + tests) instead of the previous one-off repo-bootstrap snippet.
- Reworked `requirements.txt` to match the actual imports (added `keras`,
  `keras-nlp`, `kagglehub`; removed the unused `pandas`/`seaborn`) with
  hardware-specific JAX install notes.

### Fixed
- Renamed the malformed `.gitignore:` file to `.gitignore` so ignore rules
  actually take effect.
- Resolved linter findings (ambiguous variable names, whitespace) in
  `generate_training_data.py`.
