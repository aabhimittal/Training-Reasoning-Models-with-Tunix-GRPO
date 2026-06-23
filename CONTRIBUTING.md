# Contributing

Thanks for your interest in improving **Training Reasoning Models with Tunix GRPO**!
Contributions of all kinds are welcome: bug reports, new reasoning-task
generators, reward-function improvements, documentation, and tests.

## Getting started

```bash
git clone https://github.com/aabhimittal/Training-Reasoning-Models-with-Tunix-GRPO.git
cd Training-Reasoning-Models-with-Tunix-GRPO
./setup.sh          # creates .venv, installs dev deps, runs tests
source .venv/bin/activate
```

The reward functions (`reasoning_rewards.py`) and the data generator
(`generate_training_data.py`) depend only on the Python standard library, so
you can develop and test them without a GPU/TPU or any heavy ML dependencies.

## Development workflow

| Task                | Command        |
|---------------------|----------------|
| Run tests           | `make test`    |
| Lint                | `make lint`    |
| Auto-format         | `make format`  |
| Regenerate dataset  | `make data`    |

Before opening a pull request, please make sure:

1. `make lint` reports no errors.
2. `make test` passes.
3. New behavior is covered by a test in `tests/`.

## Adding a new reasoning-task generator

1. Add a `@staticmethod` to the relevant generator class in
   `generate_training_data.py` (or add a new generator class).
2. Return a dict containing `question`, `answer`, `type`, `difficulty`,
   `domain`, and a `metadata` block (see existing generators for the shape).
3. Register it in the class's `generate_examples` list.
4. Add a test in `tests/test_data_generator.py`.

## Changing reward functions

`reasoning_rewards.py` is the canonical implementation. Keep it dependency-free
and update `tests/test_rewards.py` to cover any new behavior. If you change the
scoring, please mirror the change in the notebook's reward cell so the two stay
consistent.

## Commit messages

Use clear, descriptive commit messages in the imperative mood
(e.g. "Add geometry-area generator", "Fix numeric tolerance in correctness reward").

## Code of Conduct

By participating, you agree to abide by our
[Code of Conduct](CODE_OF_CONDUCT.md).
