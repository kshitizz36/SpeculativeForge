---
title: SpeculativeForge
emoji: "⚡"
colorFrom: yellow
colorTo: orange
sdk: docker
app_port: 8000
tags:
  - openenv
  - speculative-decoding
  - gpu-optimization
  - modal
  - rl-environment
---

# SpeculativeForge

SpeculativeForge is a local-first playground for benchmarking and optimizing
speculative decoding methods.

The core idea is simple: speculative decoding can make LLM inference cheaper
and faster, but methods are hard to compare fairly because each one often lives
in a different codepath, metric format, and runtime assumption. SpeculativeForge
puts multiple proposer strategies behind one shared evaluation contract so the
speed, throughput, acceptance, and correctness tradeoffs are visible.

This is not a chatbot wrapper. It is a control surface for reasoning about when
each speculative decoding strategy actually wins.

## What it does

- Exposes a FastAPI environment with `/reset`, `/step`, `/manual_step`,
  `/state`, `/schema`, `/metadata`, and WebSocket updates.
- Implements and compares six speculative decoding strategies:
  EAGLE-3, PARD, Medusa-1, draft model, n-gram prompt lookup, and suffix
  decoding.
- Covers both vLLM and non-vLLM PyTorch inference paths.
- Scores configs using speedup, throughput, acceptance, quality/correctness, and
  divergence checks.
- Supports a manual judge/demo path from the browser UI.
- Includes staged task profiles for A100, H100, and B200 style inference
  optimization.
- Can connect to Modal workers for real GPU-backed benchmarking.
- Falls back honestly when the live worker is unavailable instead of faking
  results.

## Measured results

These are the best measured results from the current benchmark runs:

| Strategy | Best observed result |
| --- | --- |
| PARD | Up to `1.83x` serial vLLM speedup and `1.31x` batched throughput speedup |
| EAGLE-3 | Up to `1.65x` serial vLLM speedup and `1.35x` batched throughput speedup |
| Medusa-1 | Up to `2.30x` non-vLLM throughput speedup |
| n-gram prompt lookup | Up to `1.89x` serial vLLM speedup and `2.07x` non-vLLM speedup |
| suffix decoding | Up to `1.83x` serial vLLM speedup |

Acceptance rates reached roughly:

| Strategy | Acceptance rate |
| --- | --- |
| PARD | `~61%` |
| EAGLE-3 | `~56-60%` |
| n-gram prompt lookup | `~84%` |
| suffix decoding | `~81%` |

The main takeaway is that speculative decoding is not one trick. Different
methods win under different runtime settings, so a useful system needs to make
the tradeoff between speedup, latency, throughput, acceptance, and correctness
explicit.

## Why I built it

Most AI demos optimize the model response. I wanted to optimize the system that
serves the model.

Speculative decoding is valuable because it can reduce latency and serving cost,
but the best settings depend on the target model, draft model, GPU, workload,
quality threshold, and traffic pattern. That makes it a good fit for a
benchmark/control surface: run the same methods under the same contract, then
measure which one actually helps.

The smallest useful version is not a full inference platform. It is a working
measurement loop:

```text
choose method/config -> run benchmark -> measure speed + acceptance -> compare
```

## Demo flow

1. Start the server.
2. Open the UI.
3. Select a method/task scenario.
4. Run a manual config or benchmark step.
5. Watch speedup, acceptance, quality status, and trajectory update.

The demo is designed to show the core loop quickly: speed only matters when the
quality/correctness gate holds.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
```

Then open:

```text
http://localhost:8000
```

Health check:

```bash
curl http://localhost:8000/health
```

## Optional Modal GPU backend

The local app can run without Modal for UI and environment inspection. For real
GPU measurements, deploy the Modal worker and enable the backend:

```bash
modal secret create hf-token HF_TOKEN=hf_xxx
modal deploy server/modal_workers.py

export SPECFORGE_ENABLE_MODAL_BACKEND=1
export SPECFORGE_MODAL_APP_NAME=speculate-forge-workers
export SPECFORGE_MODAL_MAX_NEW_TOKENS=24
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
```

Run a baseline:

```bash
modal run server/modal_workers.py --mode baseline --task a100 --prompt-limit 5 --max-new-tokens 24
```

Run a benchmark config:

```bash
modal run server/modal_workers.py --mode benchmark --task a100 --prompt-limit 5 --max-new-tokens 24 \
  --num-speculative-tokens 6 --acceptance-threshold 0.7 --label aggressive_probe
```

## API surface

| Endpoint | Purpose |
| --- | --- |
| `GET /` | Browser UI |
| `GET /health` | Service health |
| `GET /info` | Environment information |
| `GET /schema` | Action/observation schema |
| `GET /metadata` | Task and reward metadata |
| `GET /state` | Current environment state |
| `POST /reset` | Reset an episode |
| `POST /step` | Agent proposes candidate configs |
| `POST /manual_step` | Run one hand-picked config |
| `GET /ws` | Live UI updates |

## Example client

```python
from speculate_forge import SpeculateForgeEnv, SpeculateConfig

env = SpeculateForgeEnv(base_url="http://localhost:8000")
print(env.health())
print(env.manual_step(SpeculateConfig(label="smoke")))
```

## What I intentionally cut

For the challenge version, I kept the scope narrow.

- No full production leaderboard.
- No automatic multi-GPU rollout from day one.
- No complex auth or team dashboard.
- No fake benchmark numbers when GPU workers are offline.
- No overbuilt training UI.

The important part is the benchmark loop: comparable methods, measurable
speedups, acceptance/correctness checks, and a quality gate that keeps speed
from becoming the only objective.

## Repo shape

```text
server/             FastAPI app, environment, reward, Modal workers
speculate_forge/    Python client and shared models
ui/                 Single-page browser demo
corpus/             Prompt/task corpora
training/           Trackio + notebook scaffold for future GRPO runs
graders/            Deterministic grading helpers
blog/               Longer project write-up
```
