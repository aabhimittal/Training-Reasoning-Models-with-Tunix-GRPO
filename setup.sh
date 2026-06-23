#!/usr/bin/env bash
#
# Local development setup for Training-Reasoning-Models-with-Tunix-GRPO.
#
# Creates a virtual environment, installs the development dependencies, and
# runs the test suite to confirm everything works. Training dependencies
# (JAX/Keras/Tunix) are NOT installed here -- see requirements.txt and the
# README for hardware-specific instructions.
#
# Usage:
#   ./setup.sh
set -euo pipefail

PYTHON="${PYTHON:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

echo "==> Creating virtual environment in ${VENV_DIR}"
"${PYTHON}" -m venv "${VENV_DIR}"
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

echo "==> Upgrading pip"
python -m pip install --upgrade pip

echo "==> Installing development dependencies"
python -m pip install -r requirements-dev.txt

echo "==> Running test suite"
python -m pytest

cat <<'EOF'

Setup complete.

Next steps:
  source .venv/bin/activate                 # activate the environment
  python generate_training_data.py --count 1000   # (re)generate training data
  make test                                 # run tests
  make lint                                 # run the linter

To train the model, open tunix_reasoning_trainer.ipynb on a TPU/GPU runtime
(Kaggle is recommended) and follow the README.
EOF
