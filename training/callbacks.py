from __future__ import annotations

try:
    from transformers import TrainerCallback
except ImportError:  # pragma: no cover - optional during local scaffolding
    class TrainerCallback:  # type: ignore[override]
        pass

from training.trackio_integration import log_step_metrics


class TrackioCallback(TrainerCallback):
    """Logs TRL training metrics to Trackio when available."""

    def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[override]
        if not logs:
            return
        log_step_metrics(
            {
                "train/loss": logs.get("loss", 0.0),
                "train/kl": logs.get("kl", 0.0),
                "train/clip_fraction": logs.get("clip_fraction", 0.0),
                "train/reward_mean": logs.get("reward", 0.0),
                "train/learning_rate": logs.get("learning_rate", 0.0),
                "train/grad_norm": logs.get("grad_norm", 0.0),
                "train/epoch": logs.get("epoch", 0.0),
                "train/step": getattr(state, "global_step", 0),
            }
        )
