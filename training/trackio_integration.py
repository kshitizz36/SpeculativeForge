from __future__ import annotations

from typing import Any

try:
    import trackio
except ImportError:  # pragma: no cover - optional during local scaffolding
    trackio = None


DEFAULT_CONFIG = {
    "project": "speculate-forge",
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "task": "task1_easy_a100",
    "bf16": True,
}

QUALITY_GATE = 0.95


def init_tracking(space_id: str | None = None, config: dict[str, Any] | None = None) -> None:
    if trackio is None:
        return
    init_config = dict(DEFAULT_CONFIG)
    if config:
        init_config.update(config)
    kwargs: dict[str, Any] = {
        "project": init_config.pop("project", "speculate-forge"),
        "config": init_config,
    }
    if space_id:
        kwargs["space_id"] = space_id
    trackio.init(**kwargs)


def log_step_metrics(metrics: dict[str, Any]) -> None:
    if trackio is None:
        return
    trackio.log(metrics)


def gate_metrics(
    *,
    quality_match_rate: float,
    reward_pre_gate: float,
    reward_post_gate: float,
) -> dict[str, float]:
    return {
        "quality/exact_match": quality_match_rate,
        "quality/near_miss": 1.0 if QUALITY_GATE > quality_match_rate >= 0.90 else 0.0,
        "quality/pass": 1.0 if quality_match_rate >= QUALITY_GATE else 0.0,
        "reward/pre_gate": reward_pre_gate,
        "reward/post_gate": reward_post_gate,
        "reward/quality_blocked": (
            1.0 if quality_match_rate < QUALITY_GATE and reward_pre_gate > 0.0 else 0.0
        ),
    }


def command_center_metrics(
    *,
    scenario_name: str,
    traffic_level: str,
    operating_mode: str,
    incident_status: str,
    risk_level: str,
    budget_remaining_usd: float,
    observed_queue_pressure: str | None = None,
    oversight_status: str | None = None,
    carryover_penalty: float | None = None,
    delayed_risk_penalty: float | None = None,
    terminal_reason: str | None = None,
) -> dict[str, float | str]:
    metrics: dict[str, float | str] = {
        "ops/scenario": scenario_name,
        "ops/traffic_level": traffic_level,
        "ops/operating_mode": operating_mode,
        "ops/incident_status": incident_status,
        "ops/risk_level": risk_level,
        "ops/budget_remaining_usd": budget_remaining_usd,
    }
    if observed_queue_pressure is not None:
        metrics["ops/queue_pressure"] = observed_queue_pressure
    if oversight_status is not None:
        metrics["ops/oversight_status"] = oversight_status
    if carryover_penalty is not None:
        metrics["reward/carryover_penalty"] = carryover_penalty
    if delayed_risk_penalty is not None:
        metrics["reward/delayed_risk_penalty"] = delayed_risk_penalty
    if terminal_reason is not None:
        metrics["ops/terminal_reason"] = terminal_reason
    return metrics


def manual_step_metrics(payload: dict[str, Any]) -> dict[str, float | str]:
    info = payload.get("info") or {}
    observation = payload.get("observation") or {}
    result = payload.get("result") or {}
    reward = float(payload.get("reward") or 0.0)
    reward_diagnostics = info.get("reward_diagnostics") or {}
    quality_diagnostics = info.get("quality_diagnostics") or {}
    ops_diagnostics = info.get("ops_diagnostics") or {}
    long_horizon = info.get("long_horizon") or {}

    quality_match_rate = float(
        result.get(
            "quality_match_rate",
            quality_diagnostics.get("best_quality", observation.get("best_quality", 0.0)),
        )
        or 0.0
    )
    reward_pre_gate = float(reward_diagnostics.get("reward_pre_gate") or reward)
    throughput = float(result.get("throughput_tok_s") or 0.0)
    speedup = float(result.get("speedup") or 0.0)

    metrics: dict[str, float | str] = {
        "reward/step": reward,
        "throughput/tok_s": throughput,
        "throughput/speedup": speedup,
        "quality/pass_count": float(quality_diagnostics.get("pass_count") or 0.0),
        "quality/near_miss_count": float(
            quality_diagnostics.get("near_miss_count") or 0.0
        ),
        "ops/terminal_triggered": (
            1.0 if long_horizon.get("terminal_reason") else 0.0
        ),
        "env/backend_live": 1.0 if info.get("backend_status") == "live_modal" else 0.0,
    }
    metrics.update(
        gate_metrics(
            quality_match_rate=quality_match_rate,
            reward_pre_gate=reward_pre_gate,
            reward_post_gate=reward,
        )
    )
    metrics.update(
        command_center_metrics(
            scenario_name=str(
                observation.get("scenario_name")
                or observation.get("scenario_id")
                or "unknown"
            ),
            traffic_level=str(observation.get("traffic_level") or "unknown"),
            operating_mode=str(observation.get("operating_mode") or "unknown"),
            incident_status=str(observation.get("incident_status") or "unknown"),
            risk_level=str(observation.get("risk_level") or "unknown"),
            budget_remaining_usd=float(
                observation.get("budget_remaining_usd") or 0.0
            ),
            observed_queue_pressure=str(
                ops_diagnostics.get("observed_queue_pressure")
                or observation.get("observed_queue_pressure")
                or "unknown"
            ),
            oversight_status=str(
                ops_diagnostics.get("oversight_status")
                or observation.get("oversight_status")
                or "unknown"
            ),
            carryover_penalty=(
                float(long_horizon.get("carryover_penalty"))
                if long_horizon.get("carryover_penalty") is not None
                else None
            ),
            delayed_risk_penalty=(
                float(long_horizon.get("post_step_penalty"))
                if long_horizon.get("post_step_penalty") is not None
                else float(long_horizon.get("manual_penalty"))
                if long_horizon.get("manual_penalty") is not None
                else None
            ),
            terminal_reason=(
                str(long_horizon.get("terminal_reason"))
                if long_horizon.get("terminal_reason")
                else None
            ),
        )
    )
    return metrics


def finish_tracking() -> None:
    if trackio is None:
        return
    trackio.finish()
