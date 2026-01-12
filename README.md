# Training Reasoning Models with Tunix GRPO

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
- [Reproducibility](#reproducibility)
- [License](#license)

---

## Overview

Traditional language models often jump straight to answers without explanation. This project trains **Gemma2 2B** using Google's **Tunix library** to produce explicit reasoning traces before answering questions, making AI more **transparent**, **trustworthy**, and **debuggable**.

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

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_generations` | 4 | Responses generated per prompt |
| `learning_rate` | 1e-6 | Training learning rate |
| `kl_coef` | 0.1 | KL divergence coefficient |
| `clip_range` | 0.2 | PPO-style clipping range |

---

## Quick Start

### Option 1: Run on Kaggle (Recommended)

1. Open `tunix_reasoning_trainer.ipynb` in Kaggle
2. Enable **TPU v3-8** accelerator
3. Accept Gemma license on Kaggle
4. Upload `reasoning_training_data.json` as dataset
5. Run all cells

### Option 2: Local Setup

```bash
# Clone the repository
git clone https://github.com/aabhimittal/Training-Reasoning-Models-with-Tunix-GRPO.git
cd Training-Reasoning-Models-with-Tunix-GRPO

# Install dependencies
pip install -r requirements.txt

# Generate training data (optional)
python generate_training_data.py --count 1000 --output reasoning_training_data.json

# Open the notebook
jupyter notebook tunix_reasoning_trainer.ipynb
```

---

## Installation

### Requirements

```txt
jax[tpu]>=0.4.20
flax>=0.7.5
optax>=0.1.7
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
tqdm>=4.65.0
keras>=3.0.0
keras-nlp>=0.8.0
```

### Install Tunix

```bash
pip install git+https://github.com/google/tunix.git
```

---

## Usage

### Training the Model

```python
config = GRPOConfig(
    num_generations=4,
    learning_rate=1e-6,
    num_training_steps=500,
    batch_size=4
)

trainer = GRPOTrainer(
    model=gemma_lm,
    config=config,
    training_data=training_data
)

trainer.train(num_steps=500)
```

### Generating Training Data

```bash
python generate_training_data.py --count 1000 --output reasoning_training_data.json --seed 42
```

---

## Configuration

Key hyperparameters in `GRPOConfig`:

```python
@dataclass
class GRPOConfig:
    num_generations: int = 4
    max_prompt_length: int = 256
    max_response_length: int = 512
    learning_rate: float = 1e-6
    num_training_steps: int = 500
    batch_size: int = 4
    kl_coef: float = 0.1
    clip_range: float = 0.2
    temperature: float = 0.9
    format_reward_weight: float = 0.3
    correctness_reward_weight: float = 0.5
    reasoning_quality_weight: float = 0.2
```

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

---

## Project Structure

```
Training-Reasoning-Models-with-Tunix-GRPO/
|-- tunix_reasoning_trainer.ipynb
|-- generate_training_data.py
|-- reasoning_training_data.json
|-- requirements.txt
|-- LICENSE
+-- README.md
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