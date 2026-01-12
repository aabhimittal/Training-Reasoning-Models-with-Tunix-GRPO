# 🧠 Training Reasoning Models with Tunix GRPO

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-TPU%20Accelerated-orange.svg)](https://github.com/google/jax)

**Train Gemma models to produce transparent, step-by-step reasoning using Group Relative Policy Optimization (GRPO) via Google's Tunix library.**

> 🏆 *Google Tunix Hackathon Submission*

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Why This Matters](#-why-this-matters)
- [Novel Techniques](#-novel-techniques)
- [Architecture](#-architecture)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Usage](#-usage)
- [Configuration](#️-configuration)
- [Training Data](#-training-data)
- [Model Output Format](#-model-output-format)
- [Evaluation Domains](#-evaluation-domains)
- [Reproducibility](#-reproducibility)
- [License](#-license)

---

## 🎯 Overview

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
Calculation: 0.15 × 240 = 36
</reasoning>
<answer>36</answer>"
```

---

## 💡 Why This Matters

| Benefit | Description |
|---------|-------------|
| 🔍 **Transparency** | See exactly how the model reaches conclusions |
| ✅ **Trustworthiness** | Verify reasoning validity before accepting answers |
| 🐛 **Debuggability** | Identify where reasoning goes wrong |
| 📚 **Educational** | Learn problem-solving approaches from the model |

---

## 🔬 Novel Techniques

This project implements four cutting-edge techniques for reasoning model training:

### 1. GRPO (Group Relative Policy Optimization)
Core training algorithm using Tunix that optimizes reasoning quality through relative comparisons within groups of responses.

**Parameters:**
- Group Size: 4
- Clip Range: 0.2
- Value Coefficient: 0.1
- Entropy Coefficient: 0.01

### 2. Quantum-Inspired Strategy Optimization
Uses simulated annealing with quantum tunneling to find optimal reasoning strategies for different problem types.

```python
# Optimizes across strategies: 
['chain_of_thought', 'decomposition', 'analogy', 'elimination', 'verification']
```

### 3. Multi-Agent Debate System
Multiple reasoning "agents" debate to find the best solution through structured argumentation.

### 4. MCTS Tree Search
Monte Carlo Tree Search for exploring and refining reasoning paths to find optimal solutions.

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Training Pipeline                             │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │   Dataset   │──│    GRPO     │──│   Gemma2    │              │
│  │   Loader    │  │   Trainer   │  │    2B       │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
│         │                │                │                      │
│         ▼                ▼                ▼                      │
│  ┌─────────────────────────────────────────────────────┐        │
│  │              Novel Enhancement Layer                 │        │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐            │        │
│  │  │ Quantum  │ │  Multi-  │ │   MCTS   │            │        │
│  │  │Optimizer │ │  Agent   │ │  Search  │            │        │
│  │  └──────────┘ └──────────┘ └──────────┘            │        │
│  └─────────────────────────────────────────────────────┘        │
│                           │                                      │
│                           ▼                                      │
│                    ┌─────────────┐                               │
│                    │   Reward    │                               │
│                    │  Composer   │                               │
│                    └─────────────┘                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Option 1: Run on Kaggle (Recommended)

1. Open `tunix_reasoning_trainer.ipynb` in Kaggle
2. Enable **TPU v3-8** accelerator
3. Upload `reasoning_training_data.json`
4. Run all cells

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

## 📦 Installation

### Requirements

```txt
jax[tpu]>=0.4.20
flax>=0.7.5
optax>=0.1.7
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
```

### Install Tunix

```bash
pip install git+https://github.com/google/tunix.git
```

---

## 📖 Usage

### Training the Model

```python
# In the notebook, configure and run:
DEMO_STEPS = 5000  # For full 8-hour training

# Initialize pipeline
pipeline = IntegratedTrainingPipeline(
    config=config,
    dataset=dataset,
    reward_composer=reward_composer,
    quantum_optimizer=quantum_optimizer,
    debate_system=debate_system,
    mcts_system=mcts_system
)

# Run training
pipeline.train(num_steps=DEMO_STEPS)
```

### Generating Training Data

```bash
python generate_training_data.py --count 1000 --output reasoning_training_data.json --seed 42
```

**Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--count` | 1000 | Number of examples to generate |
| `--output` | reasoning_training_data.json | Output filename |
| `--seed` | 42 | Random seed for reproducibility |

---

## ⚙️ Configuration

Key hyperparameters in `ReasoningTrainingConfig`:

```python
@dataclass
class ReasoningTrainingConfig:
    # Model
    model_name: str = "gemma2-2b"
    model_path: str = "google/gemma-2-2b"
    vocab_size: int = 256000
    
    # GRPO parameters
    grpo_group_size: int = 4
    grpo_clip_range: float = 0.2
    grpo_value_coef: float = 0.1
    grpo_entropy_coef: float = 0.01
    
    # Training
    learning_rate: float = 1e-5
    warmup_steps: int = 100
    max_steps: int = 5000  # ~8 hours on TPU
    batch_size: int = 16
    gradient_accumulation_steps: int = 4
    
    # Generation
    max_reasoning_tokens: int = 512
    max_answer_tokens: int = 128
    temperature: float = 0.9
```

---

## 📊 Training Data

The training data contains diverse reasoning examples across multiple domains:

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
  "answer": "<reasoning>To find 15% of 240...</reasoning><answer>36</answer>",
  "type": "math",
  "difficulty": "easy"
}
```

---

## 📝 Model Output Format

All trained models produce outputs in this structured format:

```
<reasoning>
[Step-by-step reasoning process]
</reasoning>
<answer>
[Final answer]
</answer>
```

This ensures:
- Clear separation of reasoning and answer
- Easy parsing for evaluation
- Consistent output structure

---

## 📈 Evaluation Domains

The model is evaluated across:

- **Format Accuracy**: Correct use of `<reasoning>` and `<answer>` tags
- **Reasoning Quality**: Logical coherence and step validity
- **Answer Correctness**: Final answer accuracy
- **Diversity**: Variety in reasoning approaches

---

## 🔄 Reproducibility

For reproducible results:

1. **Set random seed**: `--seed 42` when generating data
2. **Use fixed configuration**: Don't modify hyperparameters between runs
3. **Same hardware**: TPU v3-8 for consistent training times
4. **Version lock**: Use exact package versions from `requirements.txt`

### Model Checkpoint Location

After training, checkpoints are saved to:
```
/kaggle/working/checkpoints/
```

---

## 📁 Project Structure

```
Training-Reasoning-Models-with-Tunix-GRPO/
├── tunix_reasoning_trainer.ipynb   # Main training notebook
├── generate_training_data.py       # Training data generator
├── reasoning_training_data.json    # Pre-generated training data
├── requirements.txt                # Python dependencies
├── setup.sh                        # Setup script
├── LICENSE                         # MIT License
└── README.md                       # This file
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Google** for the Tunix library and Gemma models
- **JAX/Flax** team for the excellent ML framework
- **Kaggle** for providing TPU compute resources

---

<p align="center">
  Made with ❤️ for the Google Tunix Hackathon
</p>
