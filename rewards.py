from typing import Dict, Any


def calculate_reward(state: Dict[str, Any], action: Any, task: Dict[str, Any]) -> float:
    """Calculate reward normalized between 0.0 and 1.0"""
    reward = 0.0

    if action.action_type == "identify_issue":
        claimed = state.get("claimed_indices", set())

        match_idx = next(
            (i for i, exp in enumerate(state["expected_issues"])
             if i not in claimed
             and exp["type"] == action.issue_type
             and abs(exp["line"] - action.line_number) <= 2),
            None
        )

        if match_idx is not None:
            severity_bonus = {
                "critical": 0.5,
                "high": 0.4,
                "medium": 0.3,
                "low": 0.2
            }.get(action.severity, 0.3)

            reward += 0.4 + severity_bonus   # max ≈ 0.9
        else:
            reward -= 0.2

    elif action.action_type == "approve":
        found = len(state["found_issues"])
        expected = len(state["expected_issues"])

        completion_ratio = (found / expected) if expected > 0 else 1.0

        if found == expected and state["false_positives"] == 0:
            reward += 1.0
        else:
            reward += 0.5 * completion_ratio

    elif action.action_type == "suggest_fix":
        reward += 0.05

    elif action.action_type == "request_changes":
        found = len(state["found_issues"])
        expected = len(state["expected_issues"])

        if expected > 0:
            reward += 0.3 * (found / expected)

    # Step efficiency penalty
    reward -= 0.01 * state["step_count"]

    # Clamp to [0.0, 1.0]
    reward = max(0.0, min(1.0, reward))

    return reward
