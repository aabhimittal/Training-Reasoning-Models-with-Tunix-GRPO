.PHONY: help install test lint format data clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

install: ## Install development dependencies
	python -m pip install -r requirements-dev.txt

test: ## Run the test suite
	python -m pytest

lint: ## Lint the codebase with ruff
	ruff check .

format: ## Auto-format the codebase with ruff
	ruff format .
	ruff check --fix .

data: ## Regenerate the sample training dataset (1000 examples)
	python generate_training_data.py --count 1000 --output reasoning_training_data.json --seed 42

clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
