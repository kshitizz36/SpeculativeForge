from __future__ import annotations

from speculate_forge.models import SpeculateConfig, SpeculateObservation


def _clamp_int(value: int, *, low: int, high: int) -> int:
    return max(low, min(high, int(value)))


def _clamp_float(value: float, *, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _base_seed(observation: SpeculateObservation) -> SpeculateConfig:
    if observation.best_config is not None:
        return observation.best_config.model_copy()

    task_slug = observation.task_slug

    if observation.gpu == "A100-80GB":
        return SpeculateConfig(
            num_speculative_tokens=2,
            acceptance_threshold=0.95,
            tree_depth=1,
            tree_branching=1,
            ngram_cache_size=512,
            adaptive_depth=False,
            label="a100_lookup_anchor",
        )

    if task_slug == "task2_medium_h100":
        return SpeculateConfig(
            num_speculative_tokens=6,
            acceptance_threshold=0.88,
            tree_depth=1,
            tree_branching=1,
            ngram_cache_size=512,
            adaptive_depth=True,
            label="h100_fp8_anchor",
        )

    if task_slug == "task3_medium_hard_h100":
        return SpeculateConfig(
            num_speculative_tokens=6,
            acceptance_threshold=0.88,
            tree_depth=1,
            tree_branching=1,
            ngram_cache_size=768,
            adaptive_depth=True,
            label="h100_tree_validation_anchor",
        )

    if task_slug == "task4_hard_b200":
        return SpeculateConfig(
            num_speculative_tokens=6,
            acceptance_threshold=0.9,
            tree_depth=1,
            tree_branching=1,
            ngram_cache_size=512,
            adaptive_depth=True,
            label="b200_validation_anchor",
        )

    return SpeculateConfig(
        num_speculative_tokens=4,
        acceptance_threshold=0.72,
        tree_depth=2,
        tree_branching=1,
        ngram_cache_size=0,
        adaptive_depth=False,
        label="generic_seed",
    )


def _quality_recovery_variants(
    seed: SpeculateConfig,
    *,
    gpu: str,
    task_slug: str,
) -> list[SpeculateConfig]:
    if gpu == "A100-80GB":
        return [
            seed.model_copy(
                update={
                    "num_speculative_tokens": 2,
                    "acceptance_threshold": 0.95,
                    "adaptive_depth": False,
                    "tree_depth": 1,
                    "tree_branching": 1,
                    "ngram_cache_size": 512,
                    "label": "quality_recovery_lookup_anchor",
                }
            ),
            seed.model_copy(
                update={
                    "num_speculative_tokens": 1,
                    "acceptance_threshold": 0.95,
                    "adaptive_depth": False,
                    "tree_depth": 1,
                    "tree_branching": 1,
                    "ngram_cache_size": 512,
                    "label": "quality_recovery_lookup_safe",
                }
            ),
            seed.model_copy(
                update={
                    "num_speculative_tokens": 2,
                    "acceptance_threshold": 0.97,
                    "adaptive_depth": False,
                    "tree_depth": 1,
                    "tree_branching": 1,
                    "ngram_cache_size": 512,
                    "label": "quality_recovery_lookup_tight",
                }
            ),
        ]

    if task_slug == "task2_medium_h100":
        return [
            seed.model_copy(
                update={
                    "num_speculative_tokens": 4,
                    "acceptance_threshold": 0.9,
                    "tree_depth": 1,
                    "tree_branching": 1,
                    "ngram_cache_size": 512,
                    "adaptive_depth": False,
                    "label": "fp8_recovery_safe",
                }
            ),
            seed.model_copy(
                update={
                    "num_speculative_tokens": 5,
                    "acceptance_threshold": 0.92,
                    "tree_depth": 1,
                    "tree_branching": 1,
                    "ngram_cache_size": 256,
                    "adaptive_depth": False,
                    "label": "fp8_recovery_tight",
                }
            ),
            seed.model_copy(
                update={
                    "num_speculative_tokens": 6,
                    "acceptance_threshold": 0.9,
                    "tree_depth": 1,
                    "tree_branching": 1,
                    "ngram_cache_size": 768,
                    "adaptive_depth": True,
                    "label": "fp8_recovery_balanced",
                }
            ),
        ]

    if task_slug == "task3_medium_hard_h100":
        return [
            seed.model_copy(
                update={
                    "num_speculative_tokens": 5,
                    "acceptance_threshold": 0.9,
                    "tree_depth": 1,
                    "tree_branching": 1,
                    "ngram_cache_size": 512,
                    "adaptive_depth": False,
                    "label": "tree_validation_recovery_safe",
                }
            ),
            seed.model_copy(
                update={
                    "num_speculative_tokens": 6,
                    "acceptance_threshold": 0.9,
                    "tree_depth": 1,
                    "tree_branching": 1,
                    "ngram_cache_size": 512,
                    "adaptive_depth": True,
                    "label": "tree_validation_recovery_balanced",
                }
            ),
            seed.model_copy(
                update={
                    "num_speculative_tokens": 6,
                    "acceptance_threshold": 0.92,
                    "tree_depth": 1,
                    "tree_branching": 1,
                    "ngram_cache_size": 256,
                    "adaptive_depth": False,
                    "label": "tree_validation_recovery_tight",
                }
            ),
        ]

    if task_slug == "task4_hard_b200":
        return [
            seed.model_copy(
                update={
                    "num_speculative_tokens": 5,
                    "acceptance_threshold": 0.92,
                    "tree_depth": 1,
                    "tree_branching": 1,
                    "ngram_cache_size": 256,
                    "adaptive_depth": False,
                    "label": "frontier_validation_recovery_safe",
                }
            ),
            seed.model_copy(
                update={
                    "num_speculative_tokens": 6,
                    "acceptance_threshold": 0.92,
                    "tree_depth": 1,
                    "tree_branching": 1,
                    "ngram_cache_size": 512,
                    "adaptive_depth": True,
                    "label": "frontier_validation_recovery_balanced",
                }
            ),
            seed.model_copy(
                update={
                    "num_speculative_tokens": 6,
                    "acceptance_threshold": 0.9,
                    "tree_depth": 1,
                    "tree_branching": 1,
                    "ngram_cache_size": 256,
                    "adaptive_depth": False,
                    "label": "frontier_validation_recovery_tight",
                }
            ),
        ]

    return [
        seed.model_copy(
            update={
                "num_speculative_tokens": 1,
                "acceptance_threshold": max(seed.acceptance_threshold, 0.95),
                "adaptive_depth": False,
                "tree_depth": 1,
                "tree_branching": 1,
                "ngram_cache_size": 512 if gpu == "A100-80GB" else 0,
                "label": "quality_recovery_lookup",
            }
        ),
        seed.model_copy(
            update={
                "num_speculative_tokens": _clamp_int(
                    seed.num_speculative_tokens - 1,
                    low=1,
                    high=16,
                ),
                "acceptance_threshold": _clamp_float(
                    seed.acceptance_threshold + 0.08,
                    low=0.1,
                    high=0.99,
                ),
                "adaptive_depth": False,
                "tree_depth": 1 if gpu == "A100-80GB" else seed.tree_depth,
                "label": "quality_recovery_tight",
            }
        ),
        seed.model_copy(
            update={
                "num_speculative_tokens": max(1, min(seed.num_speculative_tokens, 2)),
                "acceptance_threshold": max(seed.acceptance_threshold, 0.9),
                "adaptive_depth": False,
                "tree_depth": 1,
                "tree_branching": 1,
                "ngram_cache_size": 0,
                "label": "quality_recovery_safe",
            }
        ),
        seed.model_copy(
            update={
                "num_speculative_tokens": _clamp_int(
                    seed.num_speculative_tokens,
                    low=1,
                    high=4,
                ),
                "acceptance_threshold": _clamp_float(
                    seed.acceptance_threshold + 0.04,
                    low=0.1,
                    high=0.99,
                ),
                "adaptive_depth": gpu != "A100-80GB",
                "tree_depth": 1 if gpu == "A100-80GB" else seed.tree_depth,
                "label": "quality_recovery_balanced",
            }
        ),
    ]


def _throughput_variants(
    seed: SpeculateConfig,
    *,
    gpu: str,
    task_slug: str,
) -> list[SpeculateConfig]:
    if gpu == "A100-80GB":
        return [
            seed.model_copy(
                update={
                    "num_speculative_tokens": 2,
                    "acceptance_threshold": 0.95,
                    "tree_depth": 1,
                    "tree_branching": 1,
                    "ngram_cache_size": 512,
                    "adaptive_depth": False,
                    "label": "lookup_anchor",
                }
            ),
            seed.model_copy(
                update={
                    "num_speculative_tokens": 3,
                    "acceptance_threshold": 0.95,
                    "tree_depth": 1,
                    "tree_branching": 1,
                    "ngram_cache_size": 768,
                    "adaptive_depth": False,
                    "label": "lookup_push_3",
                }
            ),
            seed.model_copy(
                update={
                    "num_speculative_tokens": 4,
                    "acceptance_threshold": 0.95,
                    "tree_depth": 1,
                    "tree_branching": 1,
                    "ngram_cache_size": 1024,
                    "adaptive_depth": False,
                    "label": "lookup_push_4",
                }
            ),
        ]

    if task_slug == "task2_medium_h100":
        return [
            seed.model_copy(
                update={
                    "num_speculative_tokens": 6,
                    "acceptance_threshold": 0.88,
                    "tree_depth": 1,
                    "tree_branching": 1,
                    "ngram_cache_size": 512,
                    "adaptive_depth": True,
                    "label": "fp8_anchor",
                }
            ),
            seed.model_copy(
                update={
                    "num_speculative_tokens": 8,
                    "acceptance_threshold": 0.86,
                    "tree_depth": 1,
                    "tree_branching": 1,
                    "ngram_cache_size": 1024,
                    "adaptive_depth": True,
                    "label": "fp8_push_8",
                }
            ),
            seed.model_copy(
                update={
                    "num_speculative_tokens": 10,
                    "acceptance_threshold": 0.84,
                    "tree_depth": 1,
                    "tree_branching": 1,
                    "ngram_cache_size": 1536,
                    "adaptive_depth": True,
                    "label": "fp8_push_10",
                }
            ),
        ]

    if task_slug == "task3_medium_hard_h100":
        return [
            seed.model_copy(
                update={
                    "num_speculative_tokens": 6,
                    "acceptance_threshold": 0.88,
                    "tree_depth": 1,
                    "tree_branching": 1,
                    "ngram_cache_size": 768,
                    "adaptive_depth": True,
                    "label": "tree_validation_mid",
                }
            ),
            seed.model_copy(
                update={
                    "num_speculative_tokens": 6,
                    "acceptance_threshold": 0.88,
                    "tree_depth": 1,
                    "tree_branching": 1,
                    "ngram_cache_size": 512,
                    "adaptive_depth": True,
                    "label": "tree_validation_safe",
                }
            ),
            seed.model_copy(
                update={
                    "num_speculative_tokens": 7,
                    "acceptance_threshold": 0.88,
                    "tree_depth": 1,
                    "tree_branching": 1,
                    "ngram_cache_size": 768,
                    "adaptive_depth": True,
                    "label": "tree_validation_push",
                }
            ),
        ]

    if task_slug == "task4_hard_b200":
        return [
            seed.model_copy(
                update={
                    "num_speculative_tokens": 6,
                    "acceptance_threshold": 0.9,
                    "tree_depth": 1,
                    "tree_branching": 1,
                    "ngram_cache_size": 512,
                    "adaptive_depth": True,
                    "label": "frontier_validation_safe",
                }
            ),
            seed.model_copy(
                update={
                    "num_speculative_tokens": 6,
                    "acceptance_threshold": 0.9,
                    "tree_depth": 1,
                    "tree_branching": 1,
                    "ngram_cache_size": 768,
                    "adaptive_depth": True,
                    "label": "frontier_validation_mid",
                }
            ),
            seed.model_copy(
                update={
                    "num_speculative_tokens": 7,
                    "acceptance_threshold": 0.9,
                    "tree_depth": 1,
                    "tree_branching": 1,
                    "ngram_cache_size": 512,
                    "adaptive_depth": True,
                    "label": "frontier_validation_push",
                }
            ),
        ]

    return [
        seed.model_copy(
            update={
                "num_speculative_tokens": _clamp_int(
                    seed.num_speculative_tokens + 1,
                    low=1,
                    high=16,
                ),
                "acceptance_threshold": _clamp_float(
                    seed.acceptance_threshold,
                    low=0.1,
                    high=0.99,
                ),
                "tree_depth": 1 if gpu == "A100-80GB" else seed.tree_depth,
                "label": "throughput_probe",
            }
        ),
        seed.model_copy(
            update={
                "num_speculative_tokens": _clamp_int(
                    seed.num_speculative_tokens + 2,
                    low=1,
                    high=16,
                ),
                "acceptance_threshold": _clamp_float(
                    seed.acceptance_threshold - 0.03,
                    low=0.1,
                    high=0.99,
                ),
                "adaptive_depth": True,
                "tree_depth": 1 if gpu == "A100-80GB" else max(seed.tree_depth, 2),
                "label": "throughput_push",
            }
        ),
        seed.model_copy(
            update={
                "num_speculative_tokens": _clamp_int(
                    max(seed.num_speculative_tokens, 2),
                    low=1,
                    high=16,
                ),
                "acceptance_threshold": _clamp_float(
                    seed.acceptance_threshold + 0.02,
                    low=0.1,
                    high=0.99,
                ),
                "adaptive_depth": gpu != "A100-80GB",
                "ngram_cache_size": 0 if gpu == "A100-80GB" else 512,
                "label": "throughput_balanced",
            }
        ),
    ]


def generate_candidate_configs(
    observation: SpeculateObservation,
    phase: str,
) -> list[SpeculateConfig]:
    """Generate three configs with simple but Phase-C-usable heuristics."""

    seed = _base_seed(observation)
    gpu = observation.gpu
    task_slug = observation.task_slug

    if observation.best_quality and observation.best_quality < 0.95:
        return _quality_recovery_variants(seed, gpu=gpu, task_slug=task_slug)

    if phase == "exploitation" and observation.best_config is not None:
        return _throughput_variants(seed, gpu=gpu, task_slug=task_slug)

    if observation.iteration == 0:
        if task_slug == "task2_medium_h100":
            return [
                SpeculateConfig(
                    num_speculative_tokens=6,
                    acceptance_threshold=0.88,
                    tree_depth=1,
                    tree_branching=1,
                    ngram_cache_size=512,
                    adaptive_depth=True,
                    label="bootstrap_fp8_anchor",
                ),
                SpeculateConfig(
                    num_speculative_tokens=8,
                    acceptance_threshold=0.86,
                    tree_depth=1,
                    tree_branching=1,
                    ngram_cache_size=1024,
                    adaptive_depth=True,
                    label="bootstrap_fp8_push",
                ),
                SpeculateConfig(
                    num_speculative_tokens=4,
                    acceptance_threshold=0.9,
                    tree_depth=1,
                    tree_branching=1,
                    ngram_cache_size=256,
                    adaptive_depth=False,
                    label="bootstrap_fp8_safe",
                ),
            ]

        if task_slug == "task3_medium_hard_h100":
            return [
                SpeculateConfig(
                    num_speculative_tokens=6,
                    acceptance_threshold=0.88,
                    tree_depth=1,
                    tree_branching=1,
                    ngram_cache_size=768,
                    adaptive_depth=True,
                    label="bootstrap_tree_validation_mid",
                ),
                SpeculateConfig(
                    num_speculative_tokens=7,
                    acceptance_threshold=0.88,
                    tree_depth=1,
                    tree_branching=1,
                    ngram_cache_size=768,
                    adaptive_depth=True,
                    label="bootstrap_tree_validation_push",
                ),
                SpeculateConfig(
                    num_speculative_tokens=6,
                    acceptance_threshold=0.88,
                    tree_depth=1,
                    tree_branching=1,
                    ngram_cache_size=512,
                    adaptive_depth=True,
                    label="bootstrap_tree_validation_safe",
                ),
            ]

        if task_slug == "task4_hard_b200":
            return [
                SpeculateConfig(
                    num_speculative_tokens=6,
                    acceptance_threshold=0.9,
                    tree_depth=1,
                    tree_branching=1,
                    ngram_cache_size=512,
                    adaptive_depth=True,
                    label="bootstrap_frontier_validation_safe",
                ),
                SpeculateConfig(
                    num_speculative_tokens=6,
                    acceptance_threshold=0.9,
                    tree_depth=1,
                    tree_branching=1,
                    ngram_cache_size=768,
                    adaptive_depth=True,
                    label="bootstrap_frontier_validation_mid",
                ),
                SpeculateConfig(
                    num_speculative_tokens=5,
                    acceptance_threshold=0.92,
                    tree_depth=1,
                    tree_branching=1,
                    ngram_cache_size=256,
                    adaptive_depth=False,
                    label="bootstrap_frontier_validation_recovery",
                ),
            ]

        return [
            SpeculateConfig(
                num_speculative_tokens=2,
                acceptance_threshold=0.95,
                tree_depth=1,
                tree_branching=1,
                ngram_cache_size=512 if gpu == "A100-80GB" else 0,
                adaptive_depth=False,
                label="bootstrap_lookup_anchor",
            ),
            SpeculateConfig(
                num_speculative_tokens=3 if gpu == "A100-80GB" else 1,
                acceptance_threshold=0.95 if gpu == "A100-80GB" else 0.9,
                tree_depth=1,
                tree_branching=1,
                ngram_cache_size=768 if gpu == "A100-80GB" else 0,
                adaptive_depth=False,
                label="bootstrap_lookup_push",
            ),
            SpeculateConfig(
                num_speculative_tokens=1 if gpu == "A100-80GB" else 2,
                acceptance_threshold=0.95 if gpu == "A100-80GB" else 0.84,
                tree_depth=1 if gpu == "A100-80GB" else 2,
                tree_branching=1,
                ngram_cache_size=512 if gpu == "A100-80GB" else 0,
                adaptive_depth=False,
                label="bootstrap_lookup_safe",
            ),
        ]

    return _throughput_variants(seed, gpu=gpu, task_slug=task_slug)
