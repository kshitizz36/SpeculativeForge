from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

try:
    from openenv.core.models import (  # type: ignore
        Action as BaseAction,
        Observation as BaseObservation,
        State as BaseState,
    )
except ImportError:
    BaseAction = BaseModel
    BaseObservation = BaseModel
    BaseState = BaseModel


class SpeculateConfig(BaseModel):
    """Single candidate speculative decoding configuration."""

    num_speculative_tokens: int = Field(4, ge=1, le=16)
    acceptance_threshold: float = Field(0.5, ge=0.1, le=0.99)
    draft_temperature: float = Field(0.0, ge=0.0, le=1.0)
    tree_depth: int = Field(1, ge=1, le=6)
    tree_branching: int = Field(1, ge=1, le=4)
    ngram_cache_size: int = Field(0, ge=0, le=4096)
    adaptive_depth: bool = False
    label: str = "config"


class SpeculateAction(BaseAction):
    """Agent submits N candidate configs per iteration."""

    candidate_configs: List[SpeculateConfig] = Field(default_factory=list)
    operation: Literal[
        "tune",
        "set_mode",
        "rollback",
        "reroute",
        "quality_lock",
        "oversight_review",
        "allocate_budget",
    ] = "tune"
    target_mode: Optional[
        Literal["safe", "balanced", "aggressive", "efficient", "lock_in"]
    ] = None
    reroute_policy: Optional[
        Literal["current", "resilient", "premium", "cost_saver"]
    ] = None
    allocation_policy: Optional[Literal["performance", "balanced", "cost_save"]] = None
    risk_tolerance: Literal["low", "medium", "high"] = "medium"
    rollback_to_best: bool = False
    request_quality_lock: bool = False
    request_oversight_review: bool = False
    phase: Literal["exploration", "exploitation"] = "exploration"
    reasoning: str = ""
    hypothesis: str = ""


class TrialResult(BaseModel):
    """Result from one GPU trial."""

    config: SpeculateConfig
    throughput_tok_s: float = 0.0
    quality_match_rate: float = 0.0
    acceptance_rate: float = 0.0
    speedup: float = 1.0
    sol_score: float = 0.0
    profiling: Dict[str, float | str | bool] = Field(default_factory=dict)
    elapsed_sec: float = 0.0
    reward: float = 0.0
    violation: Optional[str] = None


class SpeculateObservation(BaseObservation):
    """Agent observation after each iteration."""

    trial_results: List[TrialResult] = Field(default_factory=list)
    best_config: Optional[SpeculateConfig] = None
    best_throughput: float = 0.0
    best_quality: float = 0.0
    best_reward: float = 0.0
    trajectory: List[float] = Field(default_factory=list)
    trajectory_throughput: List[float] = Field(default_factory=list)
    improvement_delta: float = 0.0
    baseline_throughput: float = 0.0
    current_speedup: float = 1.0
    iteration: int = 0
    phase: str = "exploration"
    turns_remaining: int = 10
    task_level: int = 1
    task_slug: str = "task1_easy_a100"
    evaluator_feedback: str = ""
    bottleneck: str = "unknown"
    gpu: str = "A100-80GB"
    cost_so_far_usd: float = 0.0
    cost_model: str = "estimated_modal_burn"
    scenario_id: str = "quality_guard_rollout"
    scenario_name: str = "Quality Guard Rollout"
    scenario_summary: str = ""
    traffic_level: str = "steady"
    workload_profile: str = "chat"
    latency_sla_ms: int = 1200
    quality_sla: float = 0.95
    budget_cap_usd: float = 5.0
    budget_remaining_usd: float = 5.0
    incident_status: str = "nominal"
    operating_mode: str = "balanced"
    primary_objective: str = "prove safe speedup"
    risk_level: str = "low"
    available_gpu_pool: List[str] = Field(default_factory=list)
    active_agents: List[str] = Field(default_factory=list)
    observed_queue_pressure: str = "low"
    oversight_status: str = "clear"


class SpeculateState(BaseState):
    """Episode state tracked across iterations."""

    iteration: int = 0
    phase: str = "exploration"
    trajectory: List[float] = Field(default_factory=list)
    trajectory_throughput: List[float] = Field(default_factory=list)
    best_config: Optional[SpeculateConfig] = None
    best_reward: float = 0.0
    best_throughput: float = 0.0
    best_quality: float = 0.0
    baseline_throughput: float = 0.0
    total_cost_usd: float = 0.0
    anti_cheat_violations: int = 0
    task_level: int = 1
    task_slug: str = "task1_easy_a100"
    task_tier: str = "easy"
    gpu: str = "A100-80GB"
    reference_outputs: Optional[List[str]] = None
    cost_model: str = "estimated_modal_burn"
    scenario_id: str = "quality_guard_rollout"
    scenario_name: str = "Quality Guard Rollout"
    scenario_summary: str = ""
    traffic_level: str = "steady"
    workload_profile: str = "chat"
    latency_sla_ms: int = 1200
    quality_sla: float = 0.95
    budget_cap_usd: float = 5.0
    budget_remaining_usd: float = 5.0
    incident_status: str = "nominal"
    operating_mode: str = "balanced"
    primary_objective: str = "prove safe speedup"
    risk_level: str = "low"
    available_gpu_pool: List[str] = Field(default_factory=list)
    active_agents: List[str] = Field(default_factory=list)


class SpeculateResetResponse(BaseModel):
    observation: SpeculateObservation
    state: SpeculateState


class SpeculateStepResponse(BaseModel):
    observation: SpeculateObservation
    state: SpeculateState
    reward: float
    done: bool
    info: dict


class SpeculateStateResponse(BaseModel):
    state: SpeculateState


Action = SpeculateAction
Observation = SpeculateObservation
State = SpeculateState
