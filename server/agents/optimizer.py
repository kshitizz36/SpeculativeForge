from __future__ import annotations


def plateau_count(trajectory: list[float], *, window: int = 2, epsilon: float = 1e-3) -> int:
    if len(trajectory) < window + 1:
        return 0

    count = 0
    for idx in range(len(trajectory) - 1, 0, -1):
        improvement = trajectory[idx] - trajectory[idx - 1]
        if improvement <= epsilon:
            count += 1
        else:
            break
    return count


def choose_phase(
    iteration: int,
    *,
    trajectory: list[float] | None = None,
    best_reward: float = 0.0,
    current_bottleneck: str | None = None,
) -> str:
    history = trajectory or []
    stalls = plateau_count(history)

    if iteration < 2:
        return "exploration"
    if current_bottleneck == "quality_regression":
        return "exploration"
    if stalls >= 2:
        return "exploration"
    if best_reward <= 0.0:
        return "exploration"
    return "exploitation"
