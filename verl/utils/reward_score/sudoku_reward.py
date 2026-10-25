"""
Sudoku reward function for single-turn RL training.

Adapted from temp/DARE/verl/utils/reward_score/math_reward.py.

Usage (via env.reward_fn config):
    env.reward_fn=verl.utils.reward_score.sudoku_reward.math_rm
"""

import json
import re
from typing import Optional, Union


def _validate_sudoku_solution(solution_str: Optional[str], ground_truth: str, puzzle: str) -> float:
    """
    Partial-credit scoring: only empty cells (puzzle[i]=='0') are evaluated.
    Returns correct_cells / total_empty_cells in [0.0, 1.0].
    """
    if solution_str is None or len(solution_str) == 0:
        return 0.0

    if len(solution_str) < 16:
        solution_str = solution_str + "0" * (16 - len(solution_str))
    elif len(solution_str) > 16:
        solution_str = solution_str[:16]

    empty_indices = [i for i in range(16) if puzzle[i] == "0"]
    if not empty_indices:
        return 0.0

    correct_cells = sum(1 for i in empty_indices if solution_str[i] == ground_truth[i])
    score = correct_cells / len(empty_indices)
    print(
        f"correct_cells: {correct_cells}, total_empty: {len(empty_indices)}, score: {score:.4f}"
    )
    return score


def sudoku_reward_func(
    solution_str: str, ground_truth: str, puzzle: str
):
    """
    Extract <answer>...</answer> from solution_str and score against ground_truth.

    Returns:
        (pred, score): pred is the extracted digit string (or None), score in [0,1].
    """
    answer_pattern = r"<answer>(.*?)</answer>"
    matches = re.findall(answer_pattern, solution_str, re.DOTALL)
    pred = (
        "".join(ch for ch in matches[-1].strip() if ch.isdigit())
        if matches
        else None
    )
    print(f"puzzle: {puzzle}, pred: {pred}, ground_truth: {ground_truth}")
    score = _validate_sudoku_solution(pred, ground_truth, puzzle)
    return pred, score


def math_rm(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[Union[dict, str]] = None,
) -> dict:
    """
    Unified reward entry point compatible with verl's reward_fn interface.

    Args:
        data_source:   Dataset identifier (should start with "sudoku").
        solution_str:  Raw model output string.
        ground_truth:  Correct 16-char answer string from reward_model.ground_truth.
        extra_info:    Dict (or JSON string) containing at least {"puzzle": "..."}.

    Returns:
        {"score": float, "acc": float, "pred": str}
    """
    if isinstance(extra_info, str):
        try:
            extra_info = json.loads(extra_info)
        except Exception:
            extra_info = {}

    if extra_info is None:
        extra_info = {}

    if data_source.startswith("sudoku"):
        puzzle = extra_info.get("puzzle", "0" * 16)
        pred, acc = sudoku_reward_func(solution_str, ground_truth, puzzle)
    else:
        # Fallback: exact match
        pred = solution_str.strip()
        acc = float(pred == ground_truth.strip())

    if pred is None:
        pred = "ERROR: Answer Extraction Failed"

    return {
        "score": acc,
        "acc": acc,
        "pred": pred,
    }
