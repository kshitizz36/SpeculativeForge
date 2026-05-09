from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from server.agents.evaluator import evaluate_results
from server.agents.optimizer import choose_phase
from server.agents.planner import generate_candidate_configs
from server.agents.scheduler import dispatch_parallel
from server.reward import compute_reward
from speculate_forge.models import (
    SpeculateAction,
    SpeculateConfig,
    SpeculateObservation,
    SpeculateResetResponse,
    SpeculateState,
    SpeculateStateResponse,
    SpeculateStepResponse,
    TrialResult,
)

TASK_SPECS = {
    1: {
        "id": "task1",
        "slug": "task1_easy_a100",
        "name": "Basic Speculation on A100",
        "difficulty": "Easy",
        "task_tier": "easy",
        "gpu": "A100-80GB",
        "target_model": "Llama-3.2-3B",
        "draft_model": "Qwen-0.5B",
        "baseline_expected_tok_s": 45.0,
        "max_steps": 8,
        "target_speedup": 1.5,
        "frontier_success_rate": 0.80,
        "prompt_file": "reference_prompts.jsonl",
        "rollout_stage": "live_validated",
        "worker_key": "task1",
    },
    2: {
        "id": "task2",
        "slug": "task2_medium_h100",
        "name": "FP8 Regime on H100",
        "difficulty": "Medium",
        "task_tier": "medium",
        "gpu": "H100-80GB",
        "target_model": "Llama-3.1-8B-FP8",
        "draft_model": "Llama-3.2-1B",
        "baseline_expected_tok_s": 65.0,
        "max_steps": 8,
        "target_speedup": 2.0,
        "frontier_success_rate": 0.40,
        "prompt_file": "task2_reference_prompts.jsonl",
        "rollout_stage": "live_validated",
        "worker_key": "task2",
    },
    3: {
        "id": "task3",
        "slug": "task3_medium_hard_h100",
        "name": "Tree Speculation on H100",
        "difficulty": "Medium-Hard",
        "task_tier": "medium_hard",
        "gpu": "H100-80GB",
        "target_model": "Llama-3.1-8B",
        "draft_model": "Llama-3.2-1B",
        "baseline_expected_tok_s": 65.0,
        "max_steps": 10,
        "target_speedup": 2.5,
        "frontier_success_rate": 0.20,
        "prompt_file": "task3_reference_prompts.jsonl",
        "rollout_stage": "live_validated",
        "worker_key": "task3",
    },
    4: {
        "id": "task4",
        "slug": "task4_hard_b200",
        "name": "NVFP4 Frontier on B200",
        "difficulty": "Hard",
        "task_tier": "hard",
        "gpu": "B200-180GB",
        "target_model": "Llama-3.1-70B-NVFP4",
        "draft_model": "Llama-3.2-1B",
        "baseline_expected_tok_s": 42.3,
        "max_steps": 10,
        "target_speedup": 3.0,
        "frontier_success_rate": 0.05,
        "prompt_file": "task4_reference_prompts.jsonl",
        "rollout_stage": "live_validated",
        "worker_key": "task4",
    },
}

MODAL_CLASS_BY_TASK = {
    1: "SpeculationA100",
    2: "SpeculationH100",
    3: "SpeculationH100",
    4: "SpeculationB200",
}

COMMAND_CENTER_AGENTS = [
    "Latency Agent",
    "Quality Agent",
    "Cost Agent",
    "Orchestrator",
]

GPU_COST_ESTIMATE_PER_SEC = {
    "A100-80GB": 0.00055,
    "H100-80GB": 0.00095,
    "B200-180GB": 0.00185,
}

TRAFFIC_DEMAND_INDEX = {
    "steady": 1.00,
    "elevated": 1.12,
    "spike": 1.22,
    "sustained_spike": 1.30,
    "burst": 1.35,
    "sustained_burst": 1.42,
}

SCENARIO_PROFILES = {
    1: {
        "scenario_id": "quality_guard_rollout",
        "scenario_name": "Quality Guard Rollout",
        "scenario_summary": (
            "Ship a safe speculative-decoding lane under a strict 95% quality SLA "
            "before widening the search."
        ),
        "traffic_level": "steady",
        "workload_profile": "chat_assistant",
        "latency_sla_ms": 1200,
        "quality_sla": 0.95,
        "budget_cap_usd": 4.5,
        "incident_status": "nominal",
        "operating_mode": "safe",
        "primary_objective": "prove safe speedup",
        "risk_level": "low",
        "available_gpu_pool": ["A100-80GB", "H100-80GB"],
    },
    2: {
        "scenario_id": "latency_spike_fp8",
        "scenario_name": "Latency Spike Control",
        "scenario_summary": (
            "Traffic surged on the premium tier. Keep latency inside the H100 SLA "
            "without opening a cost runaway."
        ),
        "traffic_level": "spike",
        "workload_profile": "interactive_chat",
        "latency_sla_ms": 900,
        "quality_sla": 0.95,
        "budget_cap_usd": 7.5,
        "incident_status": "flash_traffic",
        "operating_mode": "balanced",
        "primary_objective": "protect latency under load",
        "risk_level": "medium",
        "available_gpu_pool": ["H100-80GB", "A100-80GB"],
    },
    3: {
        "scenario_id": "tree_search_recovery",
        "scenario_name": "Tree Search Recovery",
        "scenario_summary": (
            "A harder reasoning workload needs extra throughput, but recent tree "
            "configurations are flirting with the quality boundary."
        ),
        "traffic_level": "elevated",
        "workload_profile": "reasoning_heavy",
        "latency_sla_ms": 1050,
        "quality_sla": 0.95,
        "budget_cap_usd": 8.5,
        "incident_status": "quality_watch",
        "operating_mode": "balanced",
        "primary_objective": "recover quality and reopen speed",
        "risk_level": "high",
        "available_gpu_pool": ["H100-80GB", "A100-80GB"],
    },
    4: {
        "scenario_id": "frontier_budget_firebreak",
        "scenario_name": "Frontier Budget Firebreak",
        "scenario_summary": (
            "B200 serving is live under premium traffic and budget pressure. Push "
            "frontier throughput, but avoid unsafe expensive branches."
        ),
        "traffic_level": "burst",
        "workload_profile": "frontier_reasoning",
        "latency_sla_ms": 1100,
        "quality_sla": 0.95,
        "budget_cap_usd": 14.0,
        "incident_status": "budget_pressure",
        "operating_mode": "aggressive",
        "primary_objective": "maximize safe frontier throughput",
        "risk_level": "high",
        "available_gpu_pool": ["B200-180GB", "H100-80GB"],
    },
}


@dataclass
class EnvironmentEvent:
    event: str
    payload: dict[str, Any]


@dataclass
class HiddenDynamics:
    true_demand_index: float = 1.0
    latent_queue_pressure: float = 0.0
    latent_failure_risk: float = 0.0
    delayed_sla_risk: float = 0.0
    metric_noise_level: float = 0.03
    quality_lock_active: bool = False
    oversight_required: bool = False
    oversight_last_result: str = "not_requested"
    last_operation: str = "tune"
    last_reroute_policy: str = "current"
    last_allocation_policy: str = "balanced"


class SpeculateForgeEnvironment:
    """Phase-aware environment shell with optional live Modal A100 wiring."""

    def __init__(self) -> None:
        self._current_state = SpeculateState()
        self._hidden = HiddenDynamics()
        self._last_task_level = 1
        self._modal_backend_enabled = (
            os.getenv("SPECFORGE_ENABLE_MODAL_BACKEND", "0").lower()
            in {"1", "true", "yes", "on"}
        )
        self._modal_app_name = os.getenv(
            "SPECFORGE_MODAL_APP_NAME",
            "speculate-forge-workers",
        )
        self._modal_prompt_limit = int(os.getenv("SPECFORGE_MODAL_PROMPT_LIMIT", "5"))
        modal_max_new_tokens = os.getenv("SPECFORGE_MODAL_MAX_NEW_TOKENS")
        self._modal_max_new_tokens: int | None = (
            int(modal_max_new_tokens)
            if modal_max_new_tokens not in (None, "")
            else None
        )
        self._h100_tasks_enabled = (
            os.getenv("SPECFORGE_ENABLE_H100_TASKS", "0").lower()
            in {"1", "true", "yes", "on"}
        )
        self._b200_task_enabled = (
            os.getenv("SPECFORGE_ENABLE_B200_TASK4", "0").lower()
            in {"1", "true", "yes", "on"}
        )
        self._last_backend_error: str | None = None

    def _task_spec(self, task_level: int) -> dict[str, Any]:
        if task_level not in TASK_SPECS:
            raise ValueError(f"Unsupported task level: {task_level}")
        return TASK_SPECS[task_level]

    def _scenario_profile(self, task_level: int) -> dict[str, Any]:
        profile = dict(SCENARIO_PROFILES[task_level])
        profile["available_gpu_pool"] = list(profile.get("available_gpu_pool", []))
        return profile

    def _apply_scenario_profile(self, task_level: int) -> dict[str, Any]:
        profile = self._scenario_profile(task_level)
        self._current_state.scenario_id = profile["scenario_id"]
        self._current_state.scenario_name = profile["scenario_name"]
        self._current_state.scenario_summary = profile["scenario_summary"]
        self._current_state.traffic_level = profile["traffic_level"]
        self._current_state.workload_profile = profile["workload_profile"]
        self._current_state.latency_sla_ms = profile["latency_sla_ms"]
        self._current_state.quality_sla = profile["quality_sla"]
        self._current_state.budget_cap_usd = profile["budget_cap_usd"]
        self._current_state.budget_remaining_usd = profile["budget_cap_usd"]
        self._current_state.incident_status = profile["incident_status"]
        self._current_state.operating_mode = profile["operating_mode"]
        self._current_state.primary_objective = profile["primary_objective"]
        self._current_state.risk_level = profile["risk_level"]
        self._current_state.available_gpu_pool = profile["available_gpu_pool"]
        self._current_state.active_agents = list(COMMAND_CENTER_AGENTS)
        self._hidden.true_demand_index = TRAFFIC_DEMAND_INDEX.get(
            profile["traffic_level"], 1.0
        )
        self._hidden.latent_queue_pressure = 0.0
        self._hidden.latent_failure_risk = 0.05
        self._hidden.delayed_sla_risk = 0.0
        self._hidden.metric_noise_level = 0.03
        self._hidden.quality_lock_active = False
        self._hidden.oversight_required = False
        self._hidden.oversight_last_result = "not_requested"
        self._hidden.last_operation = "tune"
        self._hidden.last_reroute_policy = "current"
        self._hidden.last_allocation_policy = "balanced"
        return profile

    def _observed_queue_pressure(self) -> str:
        pressure = self._hidden.latent_queue_pressure
        if pressure >= 0.75:
            return "critical"
        if pressure >= 0.5:
            return "high"
        if pressure >= 0.25:
            return "rising"
        return "low"

    def _oversight_status(self) -> str:
        if self._hidden.oversight_required:
            return "required"
        if self._hidden.oversight_last_result == "approved":
            return "approved"
        if self._hidden.oversight_last_result == "denied":
            return "denied"
        return "clear"

    def _ops_diagnostics(self) -> dict[str, Any]:
        return {
            "observed_queue_pressure": self._observed_queue_pressure(),
            "oversight_status": self._oversight_status(),
        }

    def _candidate_configs_for_operation(
        self,
        action: SpeculateAction,
        spec: dict[str, Any],
        snapshot: SpeculateObservation,
    ) -> list[SpeculateConfig]:
        if action.rollback_to_best and self._current_state.best_config is not None:
            base = self._current_state.best_config.model_copy(
                update={"label": "rollback_best_safe"}
            )
            return [
                base,
                base.model_copy(update={"label": "rollback_best_balanced"}),
                base.model_copy(update={"label": "rollback_best_lock"}),
            ]

        if action.candidate_configs:
            return list(action.candidate_configs)

        return generate_candidate_configs(snapshot, action.phase)

    def _apply_operational_controls(
        self,
        configs: list[SpeculateConfig],
        action: SpeculateAction,
    ) -> list[SpeculateConfig]:
        adjusted: list[SpeculateConfig] = []
        for idx, config in enumerate(configs):
            update: dict[str, Any] = {}

            if self._hidden.quality_lock_active:
                update["num_speculative_tokens"] = min(config.num_speculative_tokens, 4)
                update["acceptance_threshold"] = max(config.acceptance_threshold, 0.9)
                update["tree_depth"] = 1
                update["tree_branching"] = 1
                update["adaptive_depth"] = False

            mode = self._current_state.operating_mode
            if mode == "safe":
                update["num_speculative_tokens"] = min(
                    update.get("num_speculative_tokens", config.num_speculative_tokens),
                    5,
                )
                update["acceptance_threshold"] = max(
                    update.get("acceptance_threshold", config.acceptance_threshold),
                    0.9,
                )
            elif mode == "aggressive":
                update["num_speculative_tokens"] = min(
                    max(
                        update.get(
                            "num_speculative_tokens", config.num_speculative_tokens
                        ),
                        config.num_speculative_tokens + 1,
                    ),
                    16,
                )
                update["acceptance_threshold"] = min(
                    update.get("acceptance_threshold", config.acceptance_threshold),
                    0.92,
                )
            elif mode == "efficient":
                update["ngram_cache_size"] = max(
                    update.get("ngram_cache_size", config.ngram_cache_size),
                    256 if config.ngram_cache_size == 0 else config.ngram_cache_size,
                )

            if action.risk_tolerance == "low":
                update["acceptance_threshold"] = max(
                    update.get("acceptance_threshold", config.acceptance_threshold),
                    0.9,
                )
            elif action.risk_tolerance == "high":
                update["num_speculative_tokens"] = min(
                    update.get("num_speculative_tokens", config.num_speculative_tokens)
                    + 1,
                    16,
                )

            label_suffix = self._hidden.last_operation
            adjusted.append(
                config.model_copy(
                    update={
                        **update,
                        "label": f"{config.label}_{label_suffix}_{idx + 1}",
                    }
                )
            )
        return adjusted

    def _apply_strategic_action(
        self,
        action: SpeculateAction,
        spec: dict[str, Any],
    ) -> dict[str, Any]:
        notes: list[str] = []

        self._hidden.last_operation = action.operation
        self._hidden.last_reroute_policy = (
            action.reroute_policy or self._hidden.last_reroute_policy
        )
        self._hidden.last_allocation_policy = (
            action.allocation_policy or self._hidden.last_allocation_policy
        )

        if action.target_mode:
            self._current_state.operating_mode = action.target_mode
            notes.append(f"Operating mode set to {action.target_mode}.")

        if action.request_quality_lock or action.operation == "quality_lock":
            self._hidden.quality_lock_active = True
            self._current_state.operating_mode = "safe"
            self._current_state.primary_objective = "protect quality and stabilize service"
            notes.append("Quality lock engaged; aggressive configs will be clamped.")

        if action.reroute_policy:
            self._hidden.last_reroute_policy = action.reroute_policy
            if action.reroute_policy == "premium":
                self._current_state.available_gpu_pool = sorted(
                    self._current_state.available_gpu_pool,
                    key=lambda item: ("B200" not in item, "H100" not in item),
                )
                notes.append("Traffic reroute preference set to premium hardware.")
            elif action.reroute_policy == "cost_saver":
                self._current_state.available_gpu_pool = sorted(
                    self._current_state.available_gpu_pool,
                    key=lambda item: ("A100" not in item, "H100" not in item),
                )
                notes.append("Traffic reroute preference set to cost-saving hardware.")
            elif action.reroute_policy == "resilient":
                notes.append("Traffic reroute preference set to resilience-first.")

        if action.allocation_policy:
            self._hidden.last_allocation_policy = action.allocation_policy
            notes.append(f"Allocation policy set to {action.allocation_policy}.")

        if action.rollback_to_best or action.operation == "rollback":
            if self._current_state.best_config is not None:
                self._current_state.operating_mode = "safe"
                self._current_state.incident_status = "rollback_recovery"
                self._current_state.primary_objective = "restore the last known-good configuration"
                self._hidden.latent_queue_pressure *= 0.75
                self._hidden.latent_failure_risk *= 0.55
                self._hidden.delayed_sla_risk *= 0.6
                notes.append("Rollback armed against the last known-good configuration.")
            else:
                notes.append("Rollback requested, but no best config is available yet.")

        oversight_result = "not_requested"
        if action.request_oversight_review or action.operation == "oversight_review":
            review_risk = (
                self._hidden.latent_failure_risk
                + self._hidden.delayed_sla_risk
                + (0.1 if action.risk_tolerance == "high" else 0.0)
                + (0.08 if self._current_state.operating_mode == "aggressive" else 0.0)
            )
            if review_risk > 1.0:
                oversight_result = "denied"
                self._hidden.quality_lock_active = True
                self._current_state.operating_mode = "safe"
                self._hidden.oversight_required = True
                notes.append("Oversight denied the risky move and forced safe mode.")
            else:
                oversight_result = "approved"
                self._hidden.oversight_required = False
                notes.append("Oversight approved the requested operating plan.")
            self._hidden.oversight_last_result = oversight_result

        return {
            "operation": action.operation,
            "target_mode": self._current_state.operating_mode,
            "risk_tolerance": action.risk_tolerance,
            "allocation_policy": self._hidden.last_allocation_policy,
            "reroute_policy": self._hidden.last_reroute_policy,
            "quality_lock_active": self._hidden.quality_lock_active,
            "oversight_result": oversight_result,
            "notes": notes,
        }

    def _accrue_cost_estimate(
        self,
        trial_results: list[TrialResult],
        spec: dict[str, Any],
    ) -> None:
        cost_rate = GPU_COST_ESTIMATE_PER_SEC.get(spec["gpu"], 0.0)
        multiplier = 1.0
        if self._hidden.last_allocation_policy == "performance":
            multiplier += 0.12
        elif self._hidden.last_allocation_policy == "cost_save":
            multiplier -= 0.12

        if self._hidden.last_reroute_policy == "premium":
            multiplier += 0.08
        elif self._hidden.last_reroute_policy == "cost_saver":
            multiplier -= 0.08

        elapsed_total = sum(max(item.elapsed_sec, 0.0) for item in trial_results)
        self._current_state.total_cost_usd += elapsed_total * cost_rate * multiplier
        self._current_state.budget_remaining_usd = max(
            self._current_state.budget_cap_usd - self._current_state.total_cost_usd,
            0.0,
        )

    def _advance_operating_context(
        self,
        spec: dict[str, Any],
        *,
        bottleneck: str,
        reward: float = 0.0,
        quality_pass_count: int = 0,
    ) -> None:
        if bottleneck == "quality_regression":
            self._current_state.incident_status = "quality_watch"
            self._current_state.operating_mode = "safe"
            self._current_state.primary_objective = "restore quality above the SLA"
            self._current_state.risk_level = "high"
        elif bottleneck == "limited_speedup":
            self._current_state.incident_status = (
                "latency_pressure"
                if self._current_state.traffic_level in {"spike", "burst", "elevated"}
                else "nominal"
            )
            self._current_state.operating_mode = "balanced"
            self._current_state.primary_objective = "lift throughput without breaking SLA"
            self._current_state.risk_level = "medium"
        elif bottleneck == "promising_candidate":
            self._current_state.incident_status = "nominal"
            self._current_state.operating_mode = (
                "aggressive" if reward >= 0.75 else "balanced"
            )
            self._current_state.primary_objective = "exploit the best safe config lane"
            self._current_state.risk_level = (
                "medium" if self._current_state.iteration < 3 else "low"
            )
        elif bottleneck == "target_exceeded":
            self._current_state.incident_status = "nominal"
            self._current_state.operating_mode = "lock_in"
            self._current_state.primary_objective = "stabilize the winning configuration"
            self._current_state.risk_level = "low"
        elif bottleneck == "backend_error":
            self._current_state.incident_status = "backend_degraded"
            self._current_state.operating_mode = "safe"
            self._current_state.primary_objective = "recover hardware execution"
            self._current_state.risk_level = "high"
        elif bottleneck == "backend_pending":
            self._current_state.incident_status = "backend_pending"
            self._current_state.primary_objective = "keep the search policy ready"
            self._current_state.risk_level = "medium"
        elif bottleneck == "worker_gated":
            self._current_state.incident_status = "rollout_gate"
            self._current_state.primary_objective = "wait for gated tier promotion"
            self._current_state.risk_level = "medium"

        if self._current_state.total_cost_usd >= self._current_state.budget_cap_usd * 0.7:
            self._current_state.incident_status = "budget_pressure"
            self._current_state.operating_mode = "efficient"
            self._current_state.primary_objective = "protect budget while staying within SLA"
            self._current_state.risk_level = "high"

        if self._current_state.iteration >= max(spec["max_steps"] // 2, 2):
            if self._current_state.traffic_level == "steady":
                self._current_state.traffic_level = "elevated"
            elif self._current_state.traffic_level == "spike":
                self._current_state.traffic_level = "sustained_spike"
            elif self._current_state.traffic_level == "burst":
                self._current_state.traffic_level = "sustained_burst"

        if quality_pass_count == 0 and bottleneck != "backend_error":
            self._current_state.primary_objective = "recover quality before widening the search"

    def _update_hidden_dynamics(
        self,
        spec: dict[str, Any],
        *,
        action: SpeculateAction,
        best_trial: TrialResult | None,
        bottleneck: str,
        quality_pass_count: int,
    ) -> None:
        self._hidden.true_demand_index = TRAFFIC_DEMAND_INDEX.get(
            self._current_state.traffic_level,
            self._hidden.true_demand_index,
        )

        achieved_speedup = best_trial.speedup if best_trial is not None else 1.0
        required_speedup = 1.0 + max(self._hidden.true_demand_index - 1.0, 0.0) * 0.9
        queue_delta = max(required_speedup - achieved_speedup, 0.0) * 0.45

        if self._hidden.last_allocation_policy == "performance":
            queue_delta -= 0.08
        elif self._hidden.last_allocation_policy == "cost_save":
            queue_delta += 0.08

        if self._hidden.last_reroute_policy == "premium":
            queue_delta -= 0.05
        elif self._hidden.last_reroute_policy == "cost_saver":
            queue_delta += 0.05
        elif self._hidden.last_reroute_policy == "resilient":
            queue_delta += 0.02

        if bottleneck == "promising_candidate" and quality_pass_count > 0:
            queue_delta -= 0.06

        self._hidden.latent_queue_pressure = max(
            0.0,
            min(1.0, self._hidden.latent_queue_pressure + queue_delta),
        )

        failure_delta = 0.0
        if self._current_state.operating_mode == "aggressive":
            failure_delta += 0.10
        if action.risk_tolerance == "high":
            failure_delta += 0.08
        elif action.risk_tolerance == "low":
            failure_delta -= 0.05
        if bottleneck == "quality_regression":
            failure_delta += 0.12
        if self._hidden.quality_lock_active:
            failure_delta -= 0.10
        if self._hidden.last_reroute_policy == "resilient":
            failure_delta -= 0.05
        if self._hidden.oversight_last_result == "denied":
            failure_delta -= 0.04

        self._hidden.latent_failure_risk = max(
            0.0,
            min(1.0, self._hidden.latent_failure_risk + failure_delta),
        )
        self._hidden.delayed_sla_risk = min(
            1.0,
            (self._hidden.latent_queue_pressure * 0.6)
            + (self._hidden.latent_failure_risk * 0.4),
        )

        if self._hidden.delayed_sla_risk >= 0.78:
            self._current_state.incident_status = "sla_breach_risk"
            self._current_state.primary_objective = "avoid a delayed SLA breach"
            self._current_state.risk_level = "high"
        elif self._hidden.latent_queue_pressure >= 0.68:
            self._current_state.incident_status = "queue_backlog"
            self._current_state.primary_objective = "drain queue pressure before the next spike"
            self._current_state.risk_level = "high"
        elif self._hidden.latent_failure_risk >= 0.7:
            self._current_state.incident_status = "instability_watch"
            self._current_state.primary_objective = "reduce hidden failure risk"
            self._current_state.risk_level = "high"

        self._hidden.oversight_required = (
            self._hidden.delayed_sla_risk >= 0.72
            or self._hidden.latent_failure_risk >= 0.72
        )

    def _carryover_penalty(self) -> float:
        budget_pressure = 0.0
        if self._current_state.budget_cap_usd > 0:
            budget_used_ratio = min(
                self._current_state.total_cost_usd / self._current_state.budget_cap_usd,
                1.5,
            )
            budget_pressure = max(budget_used_ratio - 0.65, 0.0)

        penalty = (
            max(self._hidden.latent_queue_pressure - 0.30, 0.0) * 0.18
            + max(self._hidden.delayed_sla_risk - 0.25, 0.0) * 0.22
            + max(self._hidden.latent_failure_risk - 0.35, 0.0) * 0.12
            + budget_pressure * 0.10
        )
        return min(0.35, penalty)

    def _post_step_penalty(self) -> float:
        budget_pressure = 0.0
        if self._current_state.budget_cap_usd > 0:
            budget_used_ratio = min(
                self._current_state.total_cost_usd / self._current_state.budget_cap_usd,
                1.5,
            )
            budget_pressure = max(budget_used_ratio - 0.70, 0.0)

        penalty = (
            max(self._hidden.latent_queue_pressure - 0.40, 0.0) * 0.30
            + max(self._hidden.delayed_sla_risk - 0.35, 0.0) * 0.45
            + max(self._hidden.latent_failure_risk - 0.45, 0.0) * 0.22
            + budget_pressure * 0.16
        )
        return min(0.75, penalty)

    def _apply_reward_penalty(
        self,
        trial: TrialResult,
        *,
        penalty: float,
        tag: str,
    ) -> None:
        if penalty <= 0:
            return
        reward_before = trial.reward
        reward_after = max(reward_before - penalty, 0.0)
        trial.reward = reward_after
        trial.sol_score = reward_after
        trial.profiling[f"{tag}_penalty"] = penalty
        trial.profiling[f"{tag}_reward_before"] = reward_before
        trial.profiling[f"{tag}_reward_after"] = reward_after

    def _terminal_state(self, spec: dict[str, Any]) -> tuple[bool, str | None]:
        if self._current_state.budget_remaining_usd <= 0:
            self._current_state.incident_status = "budget_exhausted"
            self._current_state.primary_objective = "recover service under exhausted budget"
            self._current_state.risk_level = "high"
            return True, "budget_exhausted"

        if self._hidden.delayed_sla_risk >= 0.92:
            self._current_state.incident_status = "sla_breach_active"
            self._current_state.primary_objective = "recover from an active delayed SLA breach"
            self._current_state.risk_level = "high"
            return True, "sla_breach_active"

        if (
            self._hidden.latent_failure_risk >= 0.88
            and self._current_state.operating_mode == "aggressive"
        ):
            self._current_state.incident_status = "instability_incident"
            self._current_state.primary_objective = "recover from a stability incident"
            self._current_state.risk_level = "high"
            return True, "instability_incident"

        if (
            self._hidden.latent_queue_pressure >= 0.9
            and self._current_state.iteration >= max(spec["max_steps"] // 2, 2)
        ):
            self._current_state.incident_status = "queue_overflow"
            self._current_state.primary_objective = "drain a critical backlog before recovery"
            self._current_state.risk_level = "high"
            return True, "queue_overflow"

        return False, None
    def _modal_enabled_for_task(self, task_level: int) -> bool:
        if not self._modal_backend_enabled:
            return False
        if task_level == 1:
            return True
        if task_level in {2, 3}:
            return self._h100_tasks_enabled
        if task_level == 4:
            return self._b200_task_enabled
        return False

    def _rollout_status(self, task_level: int) -> str:
        if self._modal_enabled_for_task(task_level):
            return "live_enabled"
        return TASK_SPECS[task_level]["rollout_stage"]

    def _task_catalog(self) -> list[dict[str, Any]]:
        catalog: list[dict[str, Any]] = []
        for level, spec in TASK_SPECS.items():
            scenario = self._scenario_profile(level)
            catalog.append(
                {
                    "level": level,
                    "id": spec["id"],
                    "slug": spec["slug"],
                    "name": spec["name"],
                    "difficulty": spec["difficulty"],
                    "gpu": spec["gpu"],
                    "target_model": spec["target_model"],
                    "draft_model": spec["draft_model"],
                    "baseline_expected_tok_s": spec["baseline_expected_tok_s"],
                    "max_steps": spec["max_steps"],
                    "target_speedup": spec["target_speedup"],
                    "prompt_file": spec["prompt_file"],
                    "rollout_status": self._rollout_status(level),
                    "scenario_name": scenario["scenario_name"],
                    "scenario_summary": scenario["scenario_summary"],
                    "primary_objective": scenario["primary_objective"],
                    "traffic_level": scenario["traffic_level"],
                    "latency_sla_ms": scenario["latency_sla_ms"],
                    "quality_sla": scenario["quality_sla"],
                    "budget_cap_usd": scenario["budget_cap_usd"],
                }
            )
        return catalog

    def _candidate_summary(self, configs: list[SpeculateConfig]) -> list[dict[str, Any]]:
        return [
            {
                "label": config.label,
                "num_speculative_tokens": config.num_speculative_tokens,
                "acceptance_threshold": config.acceptance_threshold,
                "tree_depth": config.tree_depth,
                "tree_branching": config.tree_branching,
                "ngram_cache_size": config.ngram_cache_size,
                "adaptive_depth": config.adaptive_depth,
            }
            for config in configs
        ]

    def _worker_status(self) -> dict[str, str]:
        return {
            "task1_a100": self._rollout_status(1),
            "task2_h100": self._rollout_status(2),
            "task3_h100": self._rollout_status(3),
            "task4_b200": self._rollout_status(4),
        }

    def _quality_diagnostics(
        self,
        trial_results: list[TrialResult],
    ) -> dict[str, Any]:
        pass_count = sum(1 for item in trial_results if item.quality_match_rate >= 0.95)
        near_miss_count = sum(
            1 for item in trial_results if 0.90 <= item.quality_match_rate < 0.95
        )
        best_quality = max(
            (item.quality_match_rate for item in trial_results),
            default=self._current_state.best_quality,
        )
        best_exact_match_rate = max(
            (
                float(item.profiling.get("exact_match_rate") or 0.0)
                for item in trial_results
            ),
            default=0.0,
        )
        generation_strategies = sorted(
            {
                str(item.profiling.get("generation_strategy"))
                for item in trial_results
                if item.profiling.get("generation_strategy")
            }
        )
        return {
            "gate": 0.95,
            "pass_count": pass_count,
            "near_miss_count": near_miss_count,
            "best_quality": best_quality,
            "best_exact_match_rate": best_exact_match_rate,
            "generation_strategies": generation_strategies,
        }

    def _role_logs(
        self,
        *,
        task_spec: dict[str, Any],
        phase: str,
        candidate_count: int,
        backend_status: str,
        planner_origin: str = "user_supplied",
        completed_count: int = 0,
        evaluator_feedback: str | None = None,
        bottleneck: str | None = None,
        recommended_phase: str | None = None,
        strategic_action: dict[str, Any] | None = None,
    ) -> list[dict[str, str]]:
        strategic_action = strategic_action or {}
        operation = strategic_action.get("operation", "tune")
        notes = strategic_action.get("notes", [])
        if backend_status == "live_modal":
            planner_message = (
                "Generated the latency-focused candidate slate for this step."
                if planner_origin == "planner_generated"
                else (
                    "Filled the missing candidate slots using the current trajectory and SLA state."
                    if planner_origin == "planner_filled"
                    else "Accepted the caller-provided candidate configs and aligned them with the live operating context."
                )
            )
            cost_message = (
                f"Dispatched {candidate_count} config(s) in parallel, received {completed_count} live result(s), "
                f"and kept estimated burn at ${self._current_state.total_cost_usd:.3f} / "
                f"${self._current_state.budget_cap_usd:.2f}."
            )
            orchestrator_message = (
                f"Executed {operation}, recommends {recommended_phase} next, and keeps the system in {self._current_state.operating_mode} mode."
                if recommended_phase
                else "Waiting for the next trajectory update before changing operating mode."
            )
            logs = [
                {
                    "role": "Latency Agent",
                    "color": "purple",
                    "message": planner_message,
                },
                {
                    "role": "Cost Agent",
                    "color": "cyan",
                    "message": cost_message,
                },
                {
                    "role": "Quality Agent",
                    "color": "amber",
                    "message": evaluator_feedback or "Live evaluation completed.",
                },
                {
                    "role": "Orchestrator",
                    "color": "green",
                    "message": orchestrator_message,
                },
            ]
            if notes:
                logs.append(
                    {
                        "role": "Oversight Agent",
                        "color": "pink",
                        "message": notes[-1],
                    }
                )
            return logs

        if backend_status == "backend_error":
            logs = [
                {
                    "role": "Latency Agent",
                    "color": "purple",
                    "message": (
                        f"Prepared {candidate_count} config(s) for {task_spec['slug']} "
                        f"during {phase}."
                    ),
                },
                {
                    "role": "Cost Agent",
                    "color": "cyan",
                    "message": (
                        "A live Modal call was attempted but failed, so the command center fell back to the honest stub path."
                    ),
                },
                {
                    "role": "Quality Agent",
                    "color": "amber",
                    "message": (
                        f"Backend error: {self._last_backend_error}"
                        if self._last_backend_error
                        else "Backend error while reaching the live worker."
                    ),
                },
                {
                    "role": "Orchestrator",
                    "color": "green",
                    "message": "Retry live execution only after the worker issue is fixed.",
                },
            ]
            if notes:
                logs.append(
                    {
                        "role": "Oversight Agent",
                        "color": "pink",
                        "message": notes[-1],
                    }
                )
            return logs

        if backend_status == "worker_gated":
            logs = [
                {
                    "role": "Latency Agent",
                    "color": "purple",
                    "message": (
                        f"Prepared {candidate_count} config(s) for {task_spec['slug']} "
                        f"during {phase}."
                    ),
                },
                {
                    "role": "Cost Agent",
                    "color": "cyan",
                    "message": (
                        f"The {task_spec['gpu']} worker is wired for this task, but the rollout gate is still closed."
                    ),
                },
                {
                    "role": "Quality Agent",
                    "color": "amber",
                    "message": (
                        f"{task_spec['slug']} is planner-ready and staged, but live hardware execution is intentionally blocked right now."
                    ),
                },
                {
                    "role": "Orchestrator",
                    "color": "green",
                    "message": "Keep refining the search policy until this task is promoted to live execution.",
                },
            ]
            if notes:
                logs.append(
                    {
                        "role": "Oversight Agent",
                        "color": "pink",
                        "message": notes[-1],
                    }
                )
            return logs

        logs = [
            {
                "role": "Latency Agent",
                "color": "purple",
                "message": (
                    f"Prepared {candidate_count} config(s) for {task_spec['slug']} "
                    f"during {phase}."
                ),
            },
            {
                "role": "Cost Agent",
                "color": "cyan",
                "message": (
                    "Modal workers are not connected yet, so parallel GPU dispatch is "
                    "intentionally blocked."
                ),
            },
            {
                "role": "Quality Agent",
                "color": "amber",
                "message": (
                    "The environment is returning an honest stub response until live "
                    "hardware is enabled."
                ),
            },
            {
                "role": "Orchestrator",
                "color": "green",
                "message": "Keep the search policy ready and protect the SLA until Modal is wired.",
            },
        ]
        if notes:
            logs.append(
                {
                    "role": "Oversight Agent",
                    "color": "pink",
                    "message": notes[-1],
                }
            )
        return logs

    def _manual_preflight_notes(
        self,
        config: SpeculateConfig,
        task_spec: dict[str, Any],
        task_level: int,
    ) -> list[str]:
        notes: list[str] = []
        rollout_status = self._rollout_status(task_level)
        if self._modal_enabled_for_task(task_level):
            notes.append(
                f"Live Modal execution is enabled for {task_spec['gpu']} in this workspace, so this config can be benchmarked for real."
            )
        else:
            notes.append(
                f"{task_spec['slug']} is currently in {rollout_status} mode, so this is a preflight check only."
            )
        if config.acceptance_threshold < 0.3:
            notes.append(
                "Low acceptance thresholds often risk falling under the 95% quality gate."
            )
        if config.num_speculative_tokens > 8:
            notes.append(
                "Large draft bursts can overrun verification efficiency on smaller drafts."
            )
        if config.tree_depth > 1 and task_spec["gpu"] == "A100-80GB":
            notes.append(
                "Tree speculation is usually a later-phase experiment on the A100 task."
            )
        if config.ngram_cache_size > 2048:
            notes.append(
                "Large n-gram caches can trade memory pressure for throughput gains."
            )
        elif config.ngram_cache_size > 0 and task_spec["gpu"] == "A100-80GB":
            notes.append(
                "On the current A100 worker, ngram_cache_size activates prompt-lookup assisted decoding as a quality-safe lane."
            )
        if task_level == 2:
            notes.append(
                "Task 2 is the FP8/H100 regime, so prefer shallow trees and moderate speculative width before widening the search."
            )
        if task_level == 3:
            notes.append(
                "Task 3 is the tree-speculation regime, so deeper trees and adaptive depth are expected later in the search."
            )
        if task_level == 4:
            notes.append(
                "Task 4 is the frontier B200 regime, so wide speculative fronts should be treated as expensive and staged carefully."
            )
        if len(notes) == 1:
            notes.append("No obvious preflight red flags from the static config shape.")
        return notes

    def _get_modal_worker(self, task_level: int) -> Any:
        import modal

        worker_name = MODAL_CLASS_BY_TASK[task_level]
        worker_cls = modal.Cls.from_name(self._modal_app_name, worker_name)
        return worker_cls()

    def _fetch_live_baseline(self, task_level: int) -> dict[str, Any] | None:
        if not self._modal_enabled_for_task(task_level):
            return None
        try:
            worker = self._get_modal_worker(task_level)
            self._last_backend_error = None
            call_kwargs: dict[str, Any] = {
                "task_key": self._task_spec(task_level)["worker_key"],
                "prompt_limit": self._modal_prompt_limit,
            }
            if self._modal_max_new_tokens is not None:
                call_kwargs["max_new_tokens"] = self._modal_max_new_tokens
            return worker.baseline.remote(
                **call_kwargs,
            )
        except Exception as exc:
            self._last_backend_error = str(exc)
            return None

    def _run_live_config(
        self,
        config: SpeculateConfig,
        task_level: int,
    ) -> dict[str, Any] | None:
        if not self._modal_enabled_for_task(task_level):
            return None
        try:
            worker = self._get_modal_worker(task_level)
            self._last_backend_error = None
            call_kwargs: dict[str, Any] = {
                "task_key": self._task_spec(task_level)["worker_key"],
                "prompt_limit": self._modal_prompt_limit,
            }
            if self._modal_max_new_tokens is not None:
                call_kwargs["max_new_tokens"] = self._modal_max_new_tokens
            if (
                self._current_state.reference_outputs is not None
                and self._current_state.baseline_throughput > 0
            ):
                call_kwargs["reference_outputs"] = self._current_state.reference_outputs
                call_kwargs["baseline_tok_s"] = self._current_state.baseline_throughput
            return worker.benchmark_config.remote(config.model_dump(), **call_kwargs)
        except Exception as exc:
            self._last_backend_error = str(exc)
            return None

    def _ensure_live_baseline(self, task_level: int) -> dict[str, Any] | None:
        if self._current_state.baseline_throughput > 0:
            return {
                "ready": True,
                "throughput_tok_s": self._current_state.baseline_throughput,
                "reference_outputs": self._current_state.reference_outputs,
            }
        baseline = self._fetch_live_baseline(task_level)
        if not baseline or not baseline.get("ready", False):
            return baseline
        self._current_state.baseline_throughput = float(
            baseline.get("throughput_tok_s", 0.0)
        )
        reference_outputs = baseline.get("reference_outputs")
        if isinstance(reference_outputs, list):
            self._current_state.reference_outputs = reference_outputs
        return baseline

    def _build_trial_result(
        self,
        *,
        config: SpeculateConfig,
        benchmark: dict[str, Any],
        reward: float,
        reward_info: dict[str, Any],
    ) -> TrialResult:
        return TrialResult(
            config=config,
            throughput_tok_s=float(benchmark.get("throughput_tok_s") or 0.0),
            quality_match_rate=float(benchmark.get("quality_match_rate") or 0.0),
            acceptance_rate=float(benchmark.get("acceptance_rate") or 0.0),
            speedup=float(benchmark.get("speedup") or 1.0),
            sol_score=reward,
            profiling={
                "exact_match_rate": float(benchmark.get("exact_match_rate") or 0.0),
                "generation_strategy": str(
                    benchmark.get("generation_strategy") or "assistant_model"
                ),
                "reward_pre_gate": float(reward_info.get("reward_pre_gate") or 0.0),
                "reward_post_gate": float(reward_info.get("reward_post_gate") or reward),
                "quality_gate_pass": bool(reward_info.get("quality_gate_pass", False)),
                "prompt_lookup_enabled": bool(
                    (benchmark.get("applied_generation") or {}).get(
                        "prompt_lookup_num_tokens"
                    )
                ),
            },
            elapsed_sec=float(benchmark.get("elapsed_sec_total") or 0.0),
            reward=reward,
            violation=reward_info.get("violation"),
        )

    def _build_observation(
        self,
        *,
        spec: dict[str, Any],
        evaluator_feedback: str,
        bottleneck: str,
        trial_results: list[TrialResult] | None = None,
        improvement_delta: float = 0.0,
    ) -> SpeculateObservation:
        return SpeculateObservation(
            trial_results=trial_results or [],
            baseline_throughput=self._current_state.baseline_throughput,
            iteration=self._current_state.iteration,
            phase=self._current_state.phase,
            turns_remaining=max(spec["max_steps"] - self._current_state.iteration, 0),
            task_level=self._current_state.task_level,
            task_slug=self._current_state.task_slug,
            evaluator_feedback=evaluator_feedback,
            bottleneck=bottleneck,
            gpu=self._current_state.gpu or spec["gpu"],
            trajectory=self._current_state.trajectory,
            trajectory_throughput=self._current_state.trajectory_throughput,
            best_config=self._current_state.best_config,
            best_quality=self._current_state.best_quality,
            best_reward=self._current_state.best_reward,
            best_throughput=self._current_state.best_throughput,
            improvement_delta=improvement_delta,
            current_speedup=(
                self._current_state.best_throughput / self._current_state.baseline_throughput
                if self._current_state.baseline_throughput
                else 1.0
            ),
            cost_so_far_usd=self._current_state.total_cost_usd,
            cost_model=self._current_state.cost_model,
            scenario_id=self._current_state.scenario_id,
            scenario_name=self._current_state.scenario_name,
            scenario_summary=self._current_state.scenario_summary,
            traffic_level=self._current_state.traffic_level,
            workload_profile=self._current_state.workload_profile,
            latency_sla_ms=self._current_state.latency_sla_ms,
            quality_sla=self._current_state.quality_sla,
            budget_cap_usd=self._current_state.budget_cap_usd,
            budget_remaining_usd=self._current_state.budget_remaining_usd,
            incident_status=self._current_state.incident_status,
            operating_mode=self._current_state.operating_mode,
            primary_objective=self._current_state.primary_objective,
            risk_level=self._current_state.risk_level,
            available_gpu_pool=self._current_state.available_gpu_pool,
            active_agents=self._current_state.active_agents,
            observed_queue_pressure=self._observed_queue_pressure(),
            oversight_status=self._oversight_status(),
        )

    def _observation_snapshot(self, spec: dict[str, Any]) -> SpeculateObservation:
        return self._build_observation(
            spec=spec,
            evaluator_feedback="Snapshot for planner.",
            bottleneck="planner_snapshot",
        )

    def _update_state_with_best_trial(self, trial: TrialResult) -> float:
        previous_reference = (
            self._current_state.best_throughput or self._current_state.baseline_throughput
        )
        self._current_state.trajectory.append(trial.reward)
        self._current_state.trajectory_throughput.append(trial.throughput_tok_s)

        should_promote = False
        if self._current_state.best_config is None:
            should_promote = True
        elif trial.reward > self._current_state.best_reward:
            should_promote = True
        elif (
            trial.reward == self._current_state.best_reward
            and trial.quality_match_rate > self._current_state.best_quality
        ):
            should_promote = True
        elif (
            trial.reward == self._current_state.best_reward
            and trial.quality_match_rate == self._current_state.best_quality
            and trial.throughput_tok_s > self._current_state.best_throughput
        ):
            should_promote = True

        if should_promote:
            self._current_state.best_reward = trial.reward
            self._current_state.best_throughput = trial.throughput_tok_s
            self._current_state.best_quality = trial.quality_match_rate
            self._current_state.best_config = trial.config

        return trial.throughput_tok_s - previous_reference

    def _recommended_action(
        self,
        spec: dict[str, Any],
        *,
        phase: str,
        note: str,
        hypothesis: str,
    ) -> dict[str, Any]:
        suggested_configs = generate_candidate_configs(
            self._observation_snapshot(spec),
            phase,
        )
        operation = "tune"
        target_mode = self._current_state.operating_mode
        allocation_policy = self._hidden.last_allocation_policy
        reroute_policy = self._hidden.last_reroute_policy
        risk_tolerance = "medium"
        rollback_to_best = False
        request_quality_lock = self._hidden.quality_lock_active
        request_oversight_review = False

        if self._current_state.incident_status in {
            "quality_watch",
            "sla_breach_risk",
            "queue_backlog",
            "sla_breach_active",
            "queue_overflow",
            "instability_incident",
        }:
            operation = "quality_lock"
            target_mode = "safe"
            risk_tolerance = "low"
            request_quality_lock = True
            request_oversight_review = self._hidden.oversight_required
        elif self._current_state.incident_status == "budget_pressure":
            operation = "allocate_budget"
            target_mode = "efficient"
            allocation_policy = "cost_save"
            reroute_policy = "cost_saver"
            risk_tolerance = "low"
        elif self._current_state.incident_status == "budget_exhausted":
            operation = "rollback"
            target_mode = "safe"
            allocation_policy = "cost_save"
            reroute_policy = "cost_saver"
            rollback_to_best = self._current_state.best_config is not None
            risk_tolerance = "low"
        elif self._current_state.best_config is not None and self._current_state.risk_level == "high":
            operation = "rollback"
            target_mode = "safe"
            rollback_to_best = True
            risk_tolerance = "low"
            request_quality_lock = True
        elif self._hidden.oversight_required:
            operation = "oversight_review"
            request_oversight_review = True
            target_mode = self._current_state.operating_mode
            risk_tolerance = "medium"
        elif phase == "exploitation":
            operation = "set_mode"
            target_mode = "aggressive" if self._current_state.risk_level != "high" else "balanced"
            allocation_policy = "performance"
            reroute_policy = "premium" if self._current_state.traffic_level != "steady" else "current"
            risk_tolerance = "medium"

        return {
            "candidate_configs": [config.model_dump() for config in suggested_configs],
            "operation": operation,
            "target_mode": target_mode,
            "allocation_policy": allocation_policy,
            "reroute_policy": reroute_policy,
            "risk_tolerance": risk_tolerance,
            "rollback_to_best": rollback_to_best,
            "request_quality_lock": request_quality_lock,
            "request_oversight_review": request_oversight_review,
            "phase": phase,
            "reasoning": note,
            "hypothesis": hypothesis,
        }

    def _should_override_with_planner(self, action: SpeculateAction) -> bool:
        reasoning = action.reasoning.lower()
        hypothesis = action.hypothesis.lower()
        return "pending" in reasoning or "pending" in hypothesis

    def _prepared_action(
        self,
        action: SpeculateAction,
        spec: dict[str, Any],
    ) -> tuple[SpeculateAction, str]:
        snapshot = self._observation_snapshot(spec)
        base_configs = self._candidate_configs_for_operation(action, spec, snapshot)
        if self._should_override_with_planner(action):
            planned = self._apply_operational_controls(
                generate_candidate_configs(snapshot, action.phase),
                action,
            )
            return (
                action.model_copy(
                    update={
                        "candidate_configs": planned,
                        "reasoning": (
                            "Server-side planner replaced stale placeholder configs "
                            "with trajectory-aware Phase C candidates."
                        ),
                        "hypothesis": (
                            "Prioritize quality recovery first, then reopen throughput."
                        ),
                    }
                ),
                "planner_generated",
            )

        if not action.candidate_configs:
            return (
                action.model_copy(
                    update={
                        "candidate_configs": self._apply_operational_controls(
                            base_configs[:3],
                            action,
                        )
                    }
                ),
                "planner_generated",
            )

        if len(base_configs) < 3:
            planned = generate_candidate_configs(snapshot, action.phase)
            materialized = list(base_configs)
            seen = {config.label for config in materialized}
            for candidate in planned:
                if candidate.label in seen:
                    continue
                materialized.append(candidate)
                seen.add(candidate.label)
                if len(materialized) >= 3:
                    break
            materialized = self._apply_operational_controls(materialized, action)
            return (
                action.model_copy(update={"candidate_configs": materialized}),
                "planner_filled",
            )

        return (
            action.model_copy(
                update={
                    "candidate_configs": self._apply_operational_controls(
                        base_configs[:3],
                        action,
                    )
                }
            ),
            "user_supplied",
        )

    def info(self) -> dict[str, Any]:
        backend = (
            "live_modal_multitask_staged"
            if self._modal_backend_enabled
            else "stubbed_pending_modal"
        )
        ui_mode = (
            "live_api_phase_e_multitask_backend"
            if self._modal_backend_enabled
            else "live_api_stubbed_backend"
        )
        return {
            "name": "SpeculateForge",
            "stage": "phase_f_command_center_foundation",
            "hardware_backend": backend,
            "environment_mode": "autonomous_inference_operations",
            "themes": [
                "Theme 1 - Multi-Agent Interactions",
                "Theme 2 - Long-Horizon Planning & Instruction Following",
                "Theme 3 - World Modeling (Professional Tasks)",
            ],
            "tasks": TASK_SPECS,
            "task_catalog": self._task_catalog(),
            "ui_mode": ui_mode,
            "worker_status": self._worker_status(),
            "last_backend_error": self._last_backend_error,
            "active_agents": COMMAND_CENTER_AGENTS,
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "project": "SpeculateForge",
            "tagline": "Autonomous inference-operations environment powered by hardware-verified speculative decoding",
            "deliverables": {
                "openenv_environment": "in_progress",
                "colab_training_script": "instrumented",
                "hf_blog": "drafted",
                "trackio": "long_horizon_instrumented",
                "reward_model": "implemented",
                "post_training_strategy": "documented",
                "modal_a100_worker": self._rollout_status(1),
                "modal_h100_workers": {
                    "task2": self._rollout_status(2),
                    "task3": self._rollout_status(3),
                },
                "modal_b200_worker": self._rollout_status(4),
            },
        }

    def schema(self) -> dict[str, Any]:
        return {
            "action": SpeculateAction.model_json_schema(),
            "config": SpeculateConfig.model_json_schema(),
            "observation": SpeculateObservation.model_json_schema(),
            "state": SpeculateState.model_json_schema(),
        }

    def state(self) -> SpeculateStateResponse:
        return SpeculateStateResponse(state=self._current_state)

    def reset(self, task_level: int) -> SpeculateResetResponse:
        spec = self._task_spec(task_level)
        self._last_task_level = task_level
        self._current_state = SpeculateState(
            iteration=0,
            phase="exploration",
            task_level=task_level,
            task_slug=spec["slug"],
            task_tier=spec["task_tier"],
            gpu=spec["gpu"],
            baseline_throughput=0.0,
            total_cost_usd=0.0,
            cost_model="estimated_modal_burn",
        )
        scenario = self._apply_scenario_profile(task_level)

        evaluator_feedback = (
            "Foundation scaffold is ready. Modal benchmarking is not yet connected, "
            "so no GPU measurements are returned."
        )
        bottleneck = "backend_pending"

        baseline = self._ensure_live_baseline(task_level)
        if baseline and baseline.get("ready", False):
            evaluator_feedback = (
                f"Measured live baseline on {spec['gpu']} via Modal: "
                f"{float(baseline.get('throughput_tok_s', 0.0)):.2f} tok/s. "
                f"Scenario '{scenario['scenario_name']}' is active with a "
                f"{scenario['latency_sla_ms']} ms latency SLA."
            )
            bottleneck = "ready_for_search"
        elif baseline and baseline.get("ready") is False:
            evaluator_feedback = (
                f"{spec['slug']} is wired into the worker interface, but the live "
                f"{spec['gpu']} path is still intentionally gated at the current rollout stage."
            )
            bottleneck = "worker_gated"
        elif self._last_backend_error:
            evaluator_feedback = (
                "A live baseline was attempted, but the Modal backend failed. "
                "Falling back to the honest stub path."
            )
            bottleneck = "backend_error"
        else:
            evaluator_feedback = (
                f"{spec['slug']} is planner-ready and catalogued, but live {spec['gpu']} "
                "execution is still gated until the current rollout stage advances."
            )
            bottleneck = "rollout_gated"

        observation = self._build_observation(
            spec=spec,
            evaluator_feedback=evaluator_feedback,
            bottleneck=bottleneck,
        )
        return SpeculateResetResponse(observation=observation, state=self._current_state)

    def step(self, action: SpeculateAction, task_level: int) -> SpeculateStepResponse:
        spec = self._task_spec(task_level)
        iteration = self._current_state.iteration + 1
        self._current_state.iteration = iteration

        strategic_action = self._apply_strategic_action(action, spec)
        prepared_action, planner_origin = self._prepared_action(action, spec)
        self._current_state.phase = prepared_action.phase

        if not self._modal_enabled_for_task(task_level):
            self._advance_operating_context(
                spec,
                bottleneck="backend_pending",
            )
            self._update_hidden_dynamics(
                spec,
                action=prepared_action,
                best_trial=None,
                bottleneck="backend_pending",
                quality_pass_count=0,
            )
            next_phase = choose_phase(
                iteration,
                trajectory=self._current_state.trajectory,
                best_reward=self._current_state.best_reward,
                current_bottleneck="backend_pending",
            )
            role_logs = self._role_logs(
                task_spec=spec,
                phase=prepared_action.phase,
                candidate_count=len(prepared_action.candidate_configs),
                backend_status="stub",
                planner_origin=planner_origin,
                recommended_phase=next_phase,
                strategic_action=strategic_action,
            )
            observation = self._build_observation(
                spec=spec,
                evaluator_feedback=role_logs[2]["message"],
                bottleneck="backend_pending",
            )
            terminal, terminal_reason = self._terminal_state(spec)
            done = iteration >= spec["max_steps"] or terminal
            return SpeculateStepResponse(
                observation=observation,
                state=self._current_state,
                reward=0.0,
                done=done,
                info={
                    "backend_status": "stub",
                    "message": "No simulated GPU values were returned.",
                    "candidate_configs": self._candidate_summary(
                        prepared_action.candidate_configs
                    ),
                    "strategic_action": strategic_action,
                    "role_logs": role_logs,
                    "planner_origin": planner_origin,
                    "next_phase": next_phase,
                    "recommended_action": self._recommended_action(
                        spec,
                        phase=next_phase,
                        note="Prepare the next search wave while Modal remains disabled.",
                        hypothesis="Lean toward safe exploration until live hardware is enabled.",
                    ),
                    "ops_diagnostics": self._ops_diagnostics(),
                    "long_horizon": {
                        "carryover_penalty": self._carryover_penalty(),
                        "post_step_penalty": self._post_step_penalty(),
                        "terminal_reason": terminal_reason,
                    },
                    "quality_diagnostics": {
                        "gate": 0.95,
                        "pass_count": 0,
                        "near_miss_count": 0,
                        "best_quality": self._current_state.best_quality,
                    },
                    "task": {
                        "level": task_level,
                        "slug": spec["slug"],
                        "name": spec["name"],
                        "difficulty": spec["difficulty"],
                        "gpu": spec["gpu"],
                        "baseline_expected_tok_s": spec["baseline_expected_tok_s"],
                        "target_speedup": spec["target_speedup"],
                    },
                },
            )

        baseline = self._ensure_live_baseline(task_level)
        if not baseline or not baseline.get("ready", False):
            backend_status = (
                "worker_gated"
                if baseline is not None and baseline.get("ready") is False
                else "backend_error"
            )
            current_bottleneck = (
                "worker_gated" if backend_status == "worker_gated" else "backend_error"
            )
            self._advance_operating_context(
                spec,
                bottleneck=current_bottleneck,
            )
            self._update_hidden_dynamics(
                spec,
                action=prepared_action,
                best_trial=None,
                bottleneck=current_bottleneck,
                quality_pass_count=0,
            )
            next_phase = choose_phase(
                iteration,
                trajectory=self._current_state.trajectory,
                best_reward=self._current_state.best_reward,
                current_bottleneck=current_bottleneck,
            )
            role_logs = self._role_logs(
                task_spec=spec,
                phase=prepared_action.phase,
                candidate_count=len(prepared_action.candidate_configs),
                backend_status=backend_status,
                planner_origin=planner_origin,
                recommended_phase=next_phase,
                strategic_action=strategic_action,
            )
            observation = self._build_observation(
                spec=spec,
                evaluator_feedback=role_logs[2]["message"],
                bottleneck=current_bottleneck,
            )
            terminal, terminal_reason = self._terminal_state(spec)
            done = iteration >= spec["max_steps"] or terminal
            return SpeculateStepResponse(
                observation=observation,
                state=self._current_state,
                reward=0.0,
                done=done,
                info={
                    "backend_status": backend_status,
                    "message": (
                        "The live worker is wired but still intentionally gated for this task."
                        if backend_status == "worker_gated"
                        else "Live baseline was unavailable, so the step could not benchmark candidate configs."
                    ),
                    "candidate_configs": self._candidate_summary(
                        prepared_action.candidate_configs
                    ),
                    "strategic_action": strategic_action,
                    "role_logs": role_logs,
                    "planner_origin": planner_origin,
                    "next_phase": next_phase,
                    "recommended_action": self._recommended_action(
                        spec,
                        phase=next_phase,
                        note="Recover from the backend error, then retry the live search.",
                        hypothesis="Keep a conservative candidate set ready for the next successful run.",
                    ),
                    "ops_diagnostics": self._ops_diagnostics(),
                    "long_horizon": {
                        "carryover_penalty": self._carryover_penalty(),
                        "post_step_penalty": self._post_step_penalty(),
                        "terminal_reason": terminal_reason,
                    },
                    "quality_diagnostics": {
                        "gate": 0.95,
                        "pass_count": 0,
                        "near_miss_count": 0,
                        "best_quality": self._current_state.best_quality,
                    },
                },
            )

        dispatch_records = dispatch_parallel(
            prepared_action.candidate_configs[:3],
            run_config=lambda config: self._run_live_config(config, task_level),
            max_workers=3,
        )

        trial_results: list[TrialResult] = []
        completed_count = 0
        for config, benchmark in dispatch_records:
            if not benchmark or not benchmark.get("ready", False):
                continue
            completed_count += 1
            reward, reward_info = compute_reward(
                benchmark,
                float(baseline.get("throughput_tok_s", 0.0)),
                spec["task_tier"],
                self._current_state.trajectory_throughput,
            )
            trial_results.append(
                self._build_trial_result(
                    config=config,
                    benchmark=benchmark,
                    reward=reward,
                    reward_info=reward_info,
                )
            )

        if not trial_results:
            self._advance_operating_context(
                spec,
                bottleneck="backend_error",
            )
            self._update_hidden_dynamics(
                spec,
                action=prepared_action,
                best_trial=None,
                bottleneck="backend_error",
                quality_pass_count=0,
            )
            next_phase = choose_phase(
                iteration,
                trajectory=self._current_state.trajectory,
                best_reward=self._current_state.best_reward,
                current_bottleneck="backend_error",
            )
            role_logs = self._role_logs(
                task_spec=spec,
                phase=prepared_action.phase,
                candidate_count=len(prepared_action.candidate_configs),
                backend_status="backend_error",
                planner_origin=planner_origin,
                recommended_phase=next_phase,
                strategic_action=strategic_action,
            )
            observation = self._build_observation(
                spec=spec,
                evaluator_feedback=role_logs[2]["message"],
                bottleneck="backend_error",
            )
            terminal, terminal_reason = self._terminal_state(spec)
            done = iteration >= spec["max_steps"] or terminal
            return SpeculateStepResponse(
                observation=observation,
                state=self._current_state,
                reward=0.0,
                done=done,
                info={
                    "backend_status": "backend_error",
                    "message": "Live worker returned no successful trial payloads.",
                    "candidate_configs": self._candidate_summary(
                        prepared_action.candidate_configs
                    ),
                    "strategic_action": strategic_action,
                    "role_logs": role_logs,
                    "planner_origin": planner_origin,
                    "next_phase": next_phase,
                    "recommended_action": self._recommended_action(
                        spec,
                        phase=next_phase,
                        note="No live trials completed, so keep the next search wave conservative.",
                        hypothesis="The next run should validate baseline connectivity before widening the search.",
                    ),
                    "ops_diagnostics": self._ops_diagnostics(),
                    "long_horizon": {
                        "carryover_penalty": self._carryover_penalty(),
                        "post_step_penalty": self._post_step_penalty(),
                        "terminal_reason": terminal_reason,
                    },
                    "quality_diagnostics": {
                        "gate": 0.95,
                        "pass_count": 0,
                        "near_miss_count": 0,
                        "best_quality": self._current_state.best_quality,
                    },
                },
            )

        evaluation = evaluate_results(trial_results)
        carryover_penalty = self._carryover_penalty()
        if carryover_penalty > 0:
            for trial in trial_results:
                self._apply_reward_penalty(
                    trial,
                    penalty=carryover_penalty,
                    tag="carryover",
                )
            evaluation = evaluate_results(trial_results)
        assert evaluation.best_result is not None
        self._accrue_cost_estimate(evaluation.ranked_results, spec)
        best_trial = evaluation.best_result
        self._update_hidden_dynamics(
            spec,
            action=prepared_action,
            best_trial=best_trial,
            bottleneck=evaluation.bottleneck,
            quality_pass_count=evaluation.quality_pass_count,
        )
        post_step_penalty = self._post_step_penalty()
        self._apply_reward_penalty(
            best_trial,
            penalty=post_step_penalty,
            tag="post_step",
        )
        improvement_delta = self._update_state_with_best_trial(best_trial)
        self._advance_operating_context(
            spec,
            bottleneck=evaluation.bottleneck,
            reward=best_trial.reward,
            quality_pass_count=evaluation.quality_pass_count,
        )
        next_phase = choose_phase(
            iteration,
            trajectory=self._current_state.trajectory,
            best_reward=self._current_state.best_reward,
            current_bottleneck=evaluation.bottleneck,
        )
        role_logs = self._role_logs(
            task_spec=spec,
            phase=prepared_action.phase,
            candidate_count=len(prepared_action.candidate_configs),
            backend_status="live_modal",
            planner_origin=planner_origin,
            completed_count=completed_count,
            evaluator_feedback=evaluation.feedback,
            bottleneck=evaluation.bottleneck,
            recommended_phase=next_phase,
            strategic_action=strategic_action,
        )
        observation = self._build_observation(
            spec=spec,
            evaluator_feedback=evaluation.feedback,
            bottleneck=evaluation.bottleneck,
            trial_results=evaluation.ranked_results,
            improvement_delta=improvement_delta,
        )
        quality_diagnostics = self._quality_diagnostics(evaluation.ranked_results)
        terminal, terminal_reason = self._terminal_state(spec)
        done = iteration >= spec["max_steps"] or best_trial.reward >= 1.0 or terminal
        return SpeculateStepResponse(
            observation=observation,
            state=self._current_state,
            reward=best_trial.reward,
            done=done,
            info={
                "backend_status": "live_modal",
                "message": "Real Modal measurements returned.",
                "candidate_configs": self._candidate_summary(
                    prepared_action.candidate_configs
                ),
                "strategic_action": strategic_action,
                "role_logs": role_logs,
                "planner_origin": planner_origin,
                "next_phase": next_phase,
                "recommended_action": self._recommended_action(
                    spec,
                    phase=next_phase,
                    note=evaluation.feedback,
                    hypothesis=(
                        "Push throughput only after quality clears the 95% gate."
                        if quality_diagnostics["pass_count"] == 0
                        else "Exploit the best live config neighborhood."
                    ),
                ),
                "ops_diagnostics": self._ops_diagnostics(),
                "long_horizon": {
                    "carryover_penalty": carryover_penalty,
                    "post_step_penalty": post_step_penalty,
                    "terminal_reason": terminal_reason,
                },
                "quality_diagnostics": quality_diagnostics,
                "scheduler": {
                    "mode": "parallel_thread_pool",
                    "submitted": len(prepared_action.candidate_configs),
                    "completed": completed_count,
                },
                "task": {
                    "level": task_level,
                    "slug": spec["slug"],
                    "name": spec["name"],
                    "difficulty": spec["difficulty"],
                    "gpu": spec["gpu"],
                    "baseline_expected_tok_s": spec["baseline_expected_tok_s"],
                    "target_speedup": spec["target_speedup"],
                },
            },
        )

    def manual_step(self, config: SpeculateConfig, task_level: int) -> dict[str, Any]:
        spec = self._task_spec(task_level)
        preflight_notes = self._manual_preflight_notes(config, spec, task_level)

        if self._modal_enabled_for_task(task_level):
            baseline = self._ensure_live_baseline(task_level)
            benchmark = self._run_live_config(config, task_level)
            if baseline and baseline.get("ready", False) and benchmark and benchmark.get(
                "ready", False
            ):
                reward, reward_info = compute_reward(
                    benchmark,
                    float(baseline.get("throughput_tok_s", 0.0)),
                    spec["task_tier"],
                    self._current_state.trajectory_throughput,
                )
                trial = self._build_trial_result(
                    config=config,
                    benchmark=benchmark,
                    reward=reward,
                    reward_info=reward_info,
                )
                self._accrue_cost_estimate([trial], spec)
                self._update_hidden_dynamics(
                    spec,
                    action=SpeculateAction(candidate_configs=[config]),
                    best_trial=trial,
                    bottleneck=(
                        "quality_regression"
                        if trial.quality_match_rate < 0.95
                        else "promising_candidate"
                    ),
                    quality_pass_count=1 if trial.quality_match_rate >= 0.95 else 0,
                )
                carryover_penalty = self._carryover_penalty()
                self._apply_reward_penalty(
                    trial,
                    penalty=carryover_penalty + self._post_step_penalty(),
                    tag="manual_long_horizon",
                )
                self._advance_operating_context(
                    spec,
                    bottleneck=(
                        "quality_regression"
                        if trial.quality_match_rate < 0.95
                        else "promising_candidate"
                    ),
                    reward=trial.reward,
                    quality_pass_count=1 if trial.quality_match_rate >= 0.95 else 0,
                )
                _, terminal_reason = self._terminal_state(spec)
                quality_diagnostics = self._quality_diagnostics([trial])
                observation = self._build_observation(
                    spec=spec,
                    evaluator_feedback=(
                        f"Manual run measured {trial.throughput_tok_s:.2f} tok/s on "
                        f"{spec['gpu']} with reward {trial.reward:.3f}."
                    ),
                    bottleneck=(
                        "quality_regression"
                        if trial.quality_match_rate < 0.95
                        else "manual_live_validation"
                    ),
                    trial_results=[trial],
                )
                return {
                    "reward": trial.reward,
                    "result": benchmark,
                    "observation": observation.model_dump(),
                    "state": self._current_state.model_dump(),
                    "info": {
                        "backend_status": "live_modal",
                        "message": "Real GPU validation completed on the live Modal worker.",
                        "submitted_config": config.model_dump(),
                        "task": {
                            "level": task_level,
                            "slug": spec["slug"],
                            "name": spec["name"],
                            "difficulty": spec["difficulty"],
                            "gpu": spec["gpu"],
                        },
                        "preflight_notes": preflight_notes,
                        "reward_diagnostics": {
                            "reward_pre_gate": float(
                                reward_info.get("reward_pre_gate") or 0.0
                            ),
                            "reward_post_gate": float(trial.reward),
                            "quality_gate_pass": bool(
                                reward_info.get("quality_gate_pass", False)
                            ),
                        },
                        "ops_diagnostics": self._ops_diagnostics(),
                        "long_horizon": {
                            "manual_penalty": float(
                                trial.profiling.get("manual_long_horizon_penalty") or 0.0
                            ),
                            "terminal_reason": terminal_reason,
                        },
                        "quality_diagnostics": quality_diagnostics,
                        "recommended_action": self._recommended_action(
                            spec,
                            phase=choose_phase(
                                self._current_state.iteration,
                                trajectory=self._current_state.trajectory,
                                best_reward=self._current_state.best_reward,
                                current_bottleneck=(
                                    "quality_regression"
                                    if trial.quality_match_rate < 0.95
                                    else None
                                ),
                            ),
                            note="Use the manual probe as a signal for the next automated search step.",
                            hypothesis="Recover quality first if the manual probe misses the gate.",
                        ),
                        "role_logs": [
                            {
                                "role": "Manual",
                                "color": "pink",
                                "message": (
                                    f"Judge-style manual config ran live on {spec['gpu']} "
                                    f"at {trial.throughput_tok_s:.2f} tok/s with "
                                    f"estimated burn ${self._current_state.total_cost_usd:.3f}."
                                ),
                            }
                        ],
                    },
                }
            if (baseline and baseline.get("ready") is False) or (
                benchmark and benchmark.get("ready") is False
            ):
                self._advance_operating_context(spec, bottleneck="worker_gated")
                _, terminal_reason = self._terminal_state(spec)
                return {
                    "reward": 0.0,
                    "result": benchmark,
                    "observation": self._build_observation(
                        spec=spec,
                        evaluator_feedback=(
                            f"{spec['slug']} is wired for live execution but intentionally gated at the current rollout stage."
                        ),
                        bottleneck="worker_gated",
                    ).model_dump(),
                    "state": self._current_state.model_dump(),
                    "info": {
                        "backend_status": "worker_gated",
                        "message": "The live worker is reachable but intentionally gated for this task.",
                        "submitted_config": config.model_dump(),
                        "task": {
                            "level": task_level,
                            "slug": spec["slug"],
                            "name": spec["name"],
                            "difficulty": spec["difficulty"],
                            "gpu": spec["gpu"],
                        },
                        "preflight_notes": preflight_notes,
                        "ops_diagnostics": self._ops_diagnostics(),
                        "long_horizon": {
                            "terminal_reason": terminal_reason,
                        },
                        "quality_diagnostics": {
                            "gate": 0.95,
                            "pass_count": 0,
                            "near_miss_count": 0,
                            "best_quality": self._current_state.best_quality,
                        },
                        "role_logs": [
                            {
                                "role": "Manual",
                                "color": "pink",
                                "message": f"{spec['slug']} is staged in the rollout but not live yet.",
                            }
                        ],
                    },
                }

        self._advance_operating_context(spec, bottleneck="backend_pending")
        _, terminal_reason = self._terminal_state(spec)
        return {
            "reward": 0.0,
            "result": None,
            "observation": self._build_observation(
                spec=spec,
                evaluator_feedback=(
                    "Manual mode is wired, but this workspace does not have live "
                    "Modal GPU execution connected yet."
                ),
                bottleneck="backend_pending",
            ).model_dump(),
            "state": self._current_state.model_dump(),
            "info": {
                "backend_status": "stub",
                "message": (
                    "Real GPU validation is still pending."
                    if not self._last_backend_error
                    else f"Live backend attempt failed: {self._last_backend_error}"
                ),
                "submitted_config": config.model_dump(),
                "task": {
                    "level": task_level,
                    "slug": spec["slug"],
                    "name": spec["name"],
                    "difficulty": spec["difficulty"],
                    "gpu": spec["gpu"],
                },
                "preflight_notes": preflight_notes,
                "ops_diagnostics": self._ops_diagnostics(),
                "long_horizon": {
                    "terminal_reason": terminal_reason,
                },
                "quality_diagnostics": {
                    "gate": 0.95,
                    "pass_count": 0,
                    "near_miss_count": 0,
                    "best_quality": self._current_state.best_quality,
                },
                "role_logs": [
                    {
                        "role": "Manual",
                        "color": "pink",
                        "message": "Judge-style manual config captured and queued for future live validation.",
                    }
                ],
            },
        }
