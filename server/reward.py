from __future__ import annotations

import math
from typing import Any

GPU_MAX = {
    "A100-80GB": 312.0,
    "H100-80GB": 989.0,
    "B200-180GB": 2250.0,
}

TASK_TARGETS = {
    "easy": 1.5,
    "medium": 2.0,
    "medium_hard": 2.5,
    "hard": 3.0,
}


def compute_reward(
    metrics: dict[str, Any],
    baseline_tok_s: float,
    task_tier: str,
    trajectory: list[float],
) -> tuple[float, dict[str, Any]]:
    """Score a hardware run, with anti-cheat checks from the brief."""

    throughput = float(metrics.get("throughput_tok_s") or 0.0)
    quality = float(metrics.get("quality_match_rate") or 0.0)
    acceptance_rate_raw = metrics.get("acceptance_rate")
    acceptance_rate = (
        float(acceptance_rate_raw) if acceptance_rate_raw is not None else 0.0
    )
    gpu = str(metrics.get("gpu", ""))

    if baseline_tok_s <= 0:
        return 0.0, {
            "violation": "missing_baseline",
            "reward_pre_gate": 0.0,
            "reward_post_gate": 0.0,
            "quality_gate_pass": False,
            "speedup": 0.0,
        }

    target = TASK_TARGETS[task_tier]
    speedup = throughput / baseline_tok_s
    reward_pre_gate = min(speedup / target, 1.0) * quality

    if trajectory and throughput > max(trajectory + [0.0]):
        reward_pre_gate += 0.1

    if speedup > target:
        reward_pre_gate = 1.0 + 0.1 * math.log(speedup / target)

    reward_pre_gate = min(reward_pre_gate, 1.2)

    if quality < 0.95:
        return 0.0, {
            "violation": "quality_gate",
            "reward_pre_gate": reward_pre_gate,
            "reward_post_gate": 0.0,
            "quality_gate_pass": False,
            "speedup": speedup,
        }

    if acceptance_rate > 0.99:
        return 0.0, {
            "violation": "acceptance_exploit",
            "reward_pre_gate": reward_pre_gate,
            "reward_post_gate": 0.0,
            "quality_gate_pass": True,
            "speedup": speedup,
        }

    if gpu in GPU_MAX and throughput > GPU_MAX[gpu]:
        return 0.0, {
            "violation": "physics_violation",
            "reward_pre_gate": reward_pre_gate,
            "reward_post_gate": 0.0,
            "quality_gate_pass": True,
            "speedup": speedup,
        }

    return reward_pre_gate, {
        "ok": True,
        "reward_pre_gate": reward_pre_gate,
        "reward_post_gate": reward_pre_gate,
        "quality_gate_pass": True,
        "speedup": speedup,
    }
