"""SpeculateForge public package surface.

Keep imports lazy so worker-side modules can import `speculate_forge.models`
without pulling in the HTTP client stack.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "Action",
    "Observation",
    "SpeculateAction",
    "SpeculateConfig",
    "SpeculateForgeEnv",
    "SpeculateObservation",
    "SpeculateResetResponse",
    "SpeculateState",
    "SpeculateStateResponse",
    "SpeculateStepResponse",
    "State",
    "TrialResult",
]


def __getattr__(name: str) -> Any:
    if name == "SpeculateForgeEnv":
        return import_module("speculate_forge.client").SpeculateForgeEnv
    if name in {
        "Action",
        "Observation",
        "SpeculateAction",
        "SpeculateConfig",
        "SpeculateObservation",
        "SpeculateResetResponse",
        "SpeculateState",
        "SpeculateStateResponse",
        "SpeculateStepResponse",
        "State",
        "TrialResult",
    }:
        return getattr(import_module("speculate_forge.models"), name)
    raise AttributeError(f"module 'speculate_forge' has no attribute {name!r}")
