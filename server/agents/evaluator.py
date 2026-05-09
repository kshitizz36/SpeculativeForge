from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from speculate_forge.models import TrialResult


@dataclass
class EvaluationSummary:
    ranked_results: list[TrialResult]
    best_result: TrialResult | None
    feedback: str
    bottleneck: str
    quality_pass_count: int
    near_miss_count: int


def _detect_bottleneck(ranked: list[TrialResult]) -> str:
    if not ranked:
        return "backend_error"

    best = ranked[0]
    quality_pass_count = sum(1 for item in ranked if item.quality_match_rate >= 0.95)
    if quality_pass_count == 0:
        return "quality_regression"
    if best.speedup <= 1.0:
        return "limited_speedup"
    if best.reward >= 1.0:
        return "target_exceeded"
    return "promising_candidate"


def _feedback(best: TrialResult, *, bottleneck: str, near_miss_count: int) -> str:
    intro = (
        f"Best config {best.config.label} reached {best.throughput_tok_s:.2f} tok/s "
        f"({best.speedup:.2f}x baseline) at quality {best.quality_match_rate:.3f}."
    )
    if bottleneck == "quality_regression":
        return (
            intro
            + " Quality is still below the 95% gate, so the next round should bias "
            + "toward fewer speculative tokens and higher acceptance thresholds."
        )
    if bottleneck == "limited_speedup":
        return (
            intro
            + " Quality is acceptable enough to keep exploring, but throughput has "
            + "not beaten baseline yet."
        )
    if bottleneck == "target_exceeded":
        return intro + " The current search has crossed the target regime."
    if near_miss_count > 0:
        return (
            intro
            + f" {near_miss_count} config(s) landed in the 90-95% quality near-miss band."
        )
    return intro + " This is the best live candidate so far."


def evaluate_results(results: Iterable[TrialResult]) -> EvaluationSummary:
    ranked = sorted(results, key=lambda item: item.reward, reverse=True)
    if not ranked:
        return EvaluationSummary(
            ranked_results=[],
            best_result=None,
            feedback="No live trial results yet.",
            bottleneck="backend_error",
            quality_pass_count=0,
            near_miss_count=0,
        )

    best = ranked[0]
    quality_pass_count = sum(1 for item in ranked if item.quality_match_rate >= 0.95)
    near_miss_count = sum(
        1 for item in ranked if 0.90 <= item.quality_match_rate < 0.95
    )
    bottleneck = _detect_bottleneck(ranked)
    return EvaluationSummary(
        ranked_results=ranked,
        best_result=best,
        feedback=_feedback(best, bottleneck=bottleneck, near_miss_count=near_miss_count),
        bottleneck=bottleneck,
        quality_pass_count=quality_pass_count,
        near_miss_count=near_miss_count,
    )
