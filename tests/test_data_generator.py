"""Unit tests for the training data generator."""

import random

import pytest

from generate_training_data import (
    CodeGenerator,
    LogicGenerator,
    MathGenerator,
    ScienceGenerator,
    TrainingDataGenerator,
)

REQUIRED_KEYS = {"question", "answer", "type", "difficulty", "domain", "metadata"}


@pytest.fixture(autouse=True)
def _fixed_seed():
    """Make every test deterministic."""
    random.seed(1234)


def _assert_valid_example(example):
    assert REQUIRED_KEYS.issubset(example.keys())
    assert isinstance(example["question"], str) and example["question"]
    assert isinstance(example["answer"], str) and example["answer"]
    assert "verification" in example["metadata"]


class TestIndividualGenerators:
    @pytest.mark.parametrize(
        "generator",
        [
            MathGenerator.generate_percentage_problem,
            MathGenerator.generate_word_problem,
            MathGenerator.generate_geometry_problem,
            MathGenerator.generate_proportion_problem,
            CodeGenerator.generate_python_output,
            CodeGenerator.generate_debugging_problem,
            CodeGenerator.generate_complexity_problem,
            ScienceGenerator.generate_physics_problem,
            LogicGenerator.generate_syllogism,
            LogicGenerator.generate_weighing_problem,
        ],
    )
    def test_generator_produces_valid_example(self, generator):
        _assert_valid_example(generator())

    def test_linear_equation_returns_valid_or_none(self):
        # This generator returns None when x is not an integer; both are valid.
        result = MathGenerator.generate_linear_equation()
        if result is not None:
            _assert_valid_example(result)


class TestBatchGenerators:
    def test_math_batch_size(self):
        assert len(MathGenerator.generate_examples(20)) == 20

    def test_code_batch_size(self):
        assert len(CodeGenerator.generate_examples(15)) == 15


class TestTrainingDataGenerator:
    def test_generate_is_deterministic_with_seed(self):
        random.seed(42)
        first = TrainingDataGenerator().generate(60)
        random.seed(42)
        second = TrainingDataGenerator().generate(60)
        assert first == second

    def test_all_examples_valid(self):
        examples = TrainingDataGenerator().generate(60)
        assert len(examples) > 0
        for example in examples:
            _assert_valid_example(example)

    def test_custom_distribution(self):
        generator = TrainingDataGenerator(
            target_distribution={
                "math": 0.5,
                "code": 0.5,
                "logic_puzzle": 0.0,
                "science": 0.0,
                "creative": 0.0,
                "summarization": 0.0,
            }
        )
        examples = generator.generate(20)
        types = {ex["type"] for ex in examples}
        assert types <= {"math", "code"}
