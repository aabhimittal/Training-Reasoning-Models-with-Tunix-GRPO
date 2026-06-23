"""Unit tests for the reasoning reward functions."""

import pytest

from reasoning_rewards import (
    RewardConfig,
    compute_reward,
    correctness_reward,
    extract_reasoning_and_answer,
    format_reward,
    reasoning_quality_reward,
)

WELL_FORMED = (
    "<reasoning>\n"
    "First, convert 15% to a decimal: 0.15. Then multiply by 240, "
    "so 0.15 * 240 = 36. Therefore the answer is 36.\n"
    "</reasoning>\n"
    "<answer>36</answer>"
)


class TestExtract:
    def test_extracts_both_blocks(self):
        reasoning, answer = extract_reasoning_and_answer(WELL_FORMED)
        assert "convert 15%" in reasoning
        assert answer == "36"

    def test_missing_tags_return_empty(self):
        assert extract_reasoning_and_answer("just text") == ("", "")

    def test_only_answer(self):
        reasoning, answer = extract_reasoning_and_answer("<answer>42</answer>")
        assert reasoning == ""
        assert answer == "42"


class TestFormatReward:
    def test_both_tags(self):
        assert format_reward(WELL_FORMED) == 1.0

    def test_one_tag(self):
        assert format_reward("<answer>36</answer>") == 0.5
        assert format_reward("<reasoning>because</reasoning>") == 0.5

    def test_no_tags(self):
        assert format_reward("the answer is 36") == 0.0


class TestCorrectnessReward:
    def test_exact_match(self):
        assert correctness_reward("<answer>36</answer>", "36") == 1.0

    def test_case_insensitive(self):
        assert correctness_reward("<answer>Paris</answer>", "paris") == 1.0

    def test_numerical_match_ignoring_units(self):
        assert correctness_reward("<answer>150 miles</answer>", "150") == 1.0
        assert correctness_reward("<answer>$64</answer>", "64") == 1.0

    def test_numerical_tolerance(self):
        assert correctness_reward("<answer>36.001</answer>", "36") == 1.0

    def test_partial_credit_substring(self):
        # Non-numeric answer so the numerical-match path is skipped.
        reward = correctness_reward(
            "<answer>the process is photosynthesis indeed</answer>",
            "photosynthesis",
        )
        assert reward == pytest.approx(0.7)

    def test_wrong_answer(self):
        assert correctness_reward("<answer>99</answer>", "36") == 0.0

    def test_no_answer_tag(self):
        assert correctness_reward("36", "36") == 0.0


class TestReasoningQualityReward:
    def test_substantive_reasoning_scores_high(self):
        assert reasoning_quality_reward(WELL_FORMED) >= 0.8

    def test_no_reasoning_scores_zero(self):
        assert reasoning_quality_reward("<answer>36</answer>") == 0.0

    def test_bounded_between_zero_and_one(self):
        score = reasoning_quality_reward(WELL_FORMED)
        assert 0.0 <= score <= 1.0


class TestComputeReward:
    def test_perfect_response(self):
        # format(1.0)*0.3 + correctness(1.0)*0.5 + quality(1.0)*0.2 = 1.0
        assert compute_reward(WELL_FORMED, "36") == pytest.approx(1.0)

    def test_empty_response_scores_zero(self):
        assert compute_reward("", "36") == 0.0

    def test_custom_weights(self):
        config = RewardConfig(
            format_reward_weight=1.0,
            correctness_reward_weight=0.0,
            reasoning_quality_weight=0.0,
        )
        assert compute_reward("<answer>x</answer>", "y", config) == 0.5

    def test_result_is_bounded(self):
        score = compute_reward(WELL_FORMED, "wrong")
        assert 0.0 <= score <= 1.0
