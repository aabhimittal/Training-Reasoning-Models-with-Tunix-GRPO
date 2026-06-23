# Training Reasoning Models with Tunix GRPO

[![CI](https://github.com/aabhimittal/Training-Reasoning-Models-with-Tunix-GRPO/actions/workflows/ci.yml/badge.svg)](https://github.com/aabhimittal/Training-Reasoning-Models-with-Tunix-GRPO/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-TPU%20Accelerated-orange.svg)](https://github.com/google/jax)

**Train Gemma models to produce transparent, step-by-step reasoning using Group Relative Policy Optimization (GRPO) via Google's Tunix library.**

*Google Tunix Hackathon Submission*

---

## Table of Contents

- [Overview](#overview)
- [Why This Matters](#why-this-matters)
- [GRPO Training](#grpo-training)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Training Data](#training-data)
- [Model Output Format](#model-output-format)
- [Reward Functions](#reward-functions)
- [Development & Testing](#development--testing)
- [Project Structure](#project-structure)
- [License](#license)

---

## Overview

Traditional language models often jump straight to answers without explanation. This project trains **Gemma 3 1B** using Google's **Tunix library** to produce explicit reasoning traces before answering questions, making AI more **transparent**, **trustworthy**, and **debuggable**.

The training notebook ([`tunix_reasoning_trainer.ipynb`](tunix_reasoning_trainer.ipynb)) is a **real, fully-wired GRPO loop** built on Tunix's `RLCluster` and `GRPOLearner` — it actually updates the model's LoRA weights — and is designed to run on a **free-tier Google Colab** accelerator (T4 GPU or TPU).

### The Problem

```python
# Traditional Model
Q: "What is 15% of 240?"
A: "36"  # No explanation - just an answer
```

### Our Solution

```python
# Reasoning Model (trained with this project)
Q: "What is 15% of 240?"
A: "<reasoning>
To find 15% of 240, I need to convert 15% to a decimal (0.15) 
and multiply by 240. 
Calculation: 0.15 x 240 = 36
</reasoning>
<answer>36</answer>"
```

---

## Why This Matters

| Benefit | Description |
|---------|-------------|
| **Transparency** | See exactly how the model reaches conclusions |
| **Trustworthiness** | Verify reasoning validity before accepting answers |
| **Debuggability** | Identify where reasoning goes wrong |
| **Educational** | Learn problem-solving approaches from the model |

---

## GRPO Training

This project uses **Group Relative Policy Optimization (GRPO)** via Google's Tunix library.

### What is GRPO?

GRPO is a reinforcement learning algorithm for LLM alignment that:
- Generates multiple responses (G) per prompt
- Computes rewards for each response
- Uses relative advantages within the group (no value function needed)
- More memory-efficient than PPO

### Key Parameters (Tunix `GRPOConfig` / rollout)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_generations` | 2 | Responses sampled per prompt (the group `G`) |
| `beta` | 0.08 | KL penalty toward the frozen reference model |
| `epsilon` | 0.2 | PPO-style clip range |
| `learning_rate` | 3e-6 | AdamW peak learning rate (warmup + cosine decay) |
| `temperature` | 0.9 | Rollout sampling temperature |

---

## Quick Start

### Option 1: Run on Google Colab (Recommended)

1. Open [`tunix_reasoning_trainer.ipynb`](tunix_reasoning_trainer.ipynb) in Colab
   (use the **Open in Colab** badge at the top of the notebook).
2. **Runtime → Change runtime type → T4 GPU** (or a TPU runtime).
3. Accept the [Gemma license on Hugging Face](https://huggingface.co/google/gemma-3-1b-it)
   and create a read token.
4. Run all cells top to bottom. Pick the dataset with the `DATASET` flag
   (`"gsm8k"` — the proven default — or `"repo"` to use this project's own data).

The notebook does a short validation run (`MAX_STEPS = 30`) out of the box;
raise `MAX_STEPS` / `NUM_TRAIN_BATCHES` for a meaningful quality improvement.

### Option 2: Local development (no accelerator needed)

The reward functions and data generator are pure Python and fully testable
without a GPU/TPU:

```bash
git clone https://github.com/aabhimittal/Training-Reasoning-Models-with-Tunix-GRPO.git
cd Training-Reasoning-Models-with-Tunix-GRPO

./setup.sh                                       # venv + dev deps + tests
python generate_training_data.py --count 1000    # (re)generate training data
```

---

## Installation

> **Note:** The reward functions (`reasoning_rewards.py`) and the data
> generator (`generate_training_data.py`) use only the Python standard
> library — no installation is needed to run or test them. The dependencies
> below are only required to *train* the model on an accelerator.

### Training stack

The Colab notebook installs everything it needs in its first cell (it pins
JAX / Flax / Tunix / Qwix to source builds, matching the official Tunix GRPO
demo). If you want to set the environment up yourself:

```bash
pip install kagglehub tensorflow tensorflow_datasets grain \
    transformers huggingface_hub datasets "numpy>2"
pip install git+https://github.com/jax-ml/jax
pip install git+https://github.com/google/tunix
pip install git+https://github.com/google/qwix
pip install git+https://github.com/google/flax
```

> JAX picks up the accelerator (TPU/GPU) provided by the Colab runtime
> automatically; you don't install a hardware-specific JAX build on Colab.

---

## Usage

### Training the model (inside the notebook)

The notebook wires Tunix's real GRPO components together:

```python
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.grpo.grpo_learner import GRPOConfig, GRPOLearner

# RLCluster holds the trainable LoRA actor, the frozen reference, and tokenizer.
rl_cluster = rl_cluster_lib.RLCluster(
    actor=lora_policy,
    reference=gemma3,
    tokenizer=tokenizer,
    cluster_config=cluster_config,
)

grpo_trainer = GRPOLearner(
    rl_cluster=rl_cluster,
    reward_fns=[reward_format, reward_correctness, reward_reasoning_quality],
    algo_config=GRPOConfig(num_generations=2, num_iterations=1, beta=0.08, epsilon=0.2),
)

grpo_trainer.train(train_dataset, val_dataset)   # actually updates LoRA weights
```

The three `reward_fns` reuse this project's tested
[`reasoning_rewards.py`](reasoning_rewards.py) library, so the notebook and the
unit tests share one implementation.

### Generating Training Data

```bash
python generate_training_data.py --count 1000 --output reasoning_training_data.json --seed 42
```

---

## Configuration

The notebook's **Configuration** cell exposes the key knobs. Free-tier-friendly
defaults (tuned for a 16 GB T4) are shown below:

| Setting | Default | Notes |
|---------|---------|-------|
| `MODEL_ID` | `google/gemma-3-1b-it` | Gated; accept the license on Hugging Face |
| `DATASET` | `"gsm8k"` | or `"repo"` for this project's own data |
| `RANK` / `ALPHA` | 32 / 64 | LoRA adapter size |
| `NUM_GENERATIONS` | 2 | Responses per prompt |
| `TRAIN_MICRO_BATCH_SIZE` | 1 | Keep at 1 on free tier |
| `MAX_PROMPT_LENGTH` | 256 | Prompt token budget |
| `TOTAL_GENERATION_STEPS` | 512 | Response token budget |
| `MAX_STEPS` | 30 | Bump to 300–1000 for real gains |
| `LEARNING_RATE` | 3e-6 | AdamW peak (warmup + cosine decay) |
| `BETA` / `EPSILON` | 0.08 / 0.2 | KL penalty / clip range |

---

## Training Data

| Domain | Description |
|--------|-------------|
| **Mathematics** | Arithmetic, algebra, percentages |
| **Logic** | Deductive reasoning, puzzles |
| **Science** | Physics, biology, chemistry concepts |
| **Common Sense** | Everyday reasoning tasks |

### Data Format

```json
{
  "question": "What is 15% of 240?",
  "answer": "36",
  "type": "math",
  "difficulty": "easy"
}
```

---

## Model Output Format

```
<reasoning>
[Step-by-step reasoning process]
</reasoning>
<answer>
[Final answer]
</answer>
```

---

## Reward Functions

1. **Format Reward (0.3)**: Correct use of reasoning/answer tags
2. **Correctness Reward (0.5)**: Answer matches expected value
3. **Reasoning Quality Reward (0.2)**: Step-by-step explanations

These are implemented as a small, dependency-free, unit-tested library in
[`reasoning_rewards.py`](reasoning_rewards.py) so you can reuse them in your own
GRPO pipeline:

```python
from reasoning_rewards import compute_reward

response = "<reasoning>0.15 * 240 = 36</reasoning><answer>36</answer>"
compute_reward(response, expected_answer="36")  # -> 0.88
```

---

## Development & Testing

The core logic (reward functions and data generator) is pure Python and fully
testable without any ML dependencies or accelerators.

```bash
# One-step setup: virtualenv + dev deps + run tests
./setup.sh
source .venv/bin/activate

# Or use the Makefile targets
make install   # install dev dependencies
make test      # run the pytest suite
make lint      # lint with ruff
make data      # regenerate reasoning_training_data.json
```

Continuous integration runs lint and tests on Python 3.8, 3.10, and 3.12 (see
[`.github/workflows/ci.yml`](.github/workflows/ci.yml)). Contributions are
welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Project Structure

```
Training-Reasoning-Models-with-Tunix-GRPO/
├── tunix_reasoning_trainer.ipynb   # GRPO training notebook (run on TPU/GPU)
├── reasoning_rewards.py            # Reusable, tested reward functions
├── generate_training_data.py       # Synthetic reasoning-data generator (CLI)
├── reasoning_training_data.json    # Sample generated dataset
├── tests/                          # pytest suite
├── requirements.txt                # Training dependencies
├── requirements-dev.txt            # Dev/test dependencies
├── pyproject.toml                  # Project metadata + ruff/pytest config
├── Makefile                        # Common dev tasks
├── setup.sh                        # Local environment bootstrap
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── CHANGELOG.md
├── LICENSE
└── README.md
```

---

## References

- [Tunix Documentation](https://tunix.readthedocs.io/en/stable/)
- [Tunix GitHub](https://github.com/google/tunix)
- [GRPO Demo - Kaggle](https://www.kaggle.com/code/windmaple/grpo-demo-gemma2-2b)

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- Google for Tunix and Gemma models
- JAX/Flax team
- Kaggle for TPU resources

---

Built for transparent AI reasoning - Google Tunix Hackathon