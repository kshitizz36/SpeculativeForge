# SpeculateForge Final Week Execution Plan

## Goal

Maximize win probability by **keeping the validated real-GPU backend** and spending the remaining time on the pieces judges will score hardest:

1. environment framing
2. visible learning signal
3. demo reliability
4. public deliverables

This plan assumes the current state is:

- Task 1: live validated
- Task 2: live validated
- Task 3: live validated
- Task 4: live validated B200 path
- command-center state layer: in progress

## Executive Decision

Do **not** pivot away from SpeculateForge.

Do **not** spend the next 4-5 days rewriting the core backend.

Do **not** ship mocked performance claims as part of the main demo.

The winning move is:

> present SpeculateForge as an autonomous inference-operations environment built on top of a real speculative-decoding truth engine.

## Current Strongest Story

The backend already proves something rare:

- real A100 / H100 / B200 measurement paths
- real reward
- real anti-cheat
- real multi-step search
- real environment API

The weakness is not backend legitimacy.

The weakness is that judges could still summarize it as:

> "config tuning on GPUs"

The fix is to raise the layer to:

> "multi-agent AI system managing a live inference platform under latency, quality, cost, and incident constraints"

## What To Integrate vs Reference

### EAGLE

Useful:

- as a benchmark comparison
- as a future backend lane
- as a README / pitch credibility reference

Not useful right now:

- full EAGLE training integration
- fake or mocked EAGLE throughput in the main demo

Decision:

- add `inference_backend` support only if it can be **clearly marked as staged**
- do **not** claim live EAGLE performance unless it is actually measured
- use EAGLE in docs/pitch as:
  - "SpeculateForge can optimize different speculative methods, not only HF assisted decoding"

### Atropos

Useful:

- as a devex pattern
- for local debug tooling
- for trainer-agnostic framing

Not useful right now:

- full Atropos microservice integration
- re-architecting the environment around async queues this week

Decision:

- borrow the **process / view-run** idea
- add a simple local rollout-debug path
- mention Atropos compatibility in docs as future integration

## Priority Stack

### P0: Must Ship

These are the highest-value items for judging.

1. Command-center environment layer
2. Trackio reward curve from real training
3. Hugging Face Space deployment
4. Manual mode demo reliability
5. Blog + pitch rewrite
6. Backup demo videos

### P1: Strong Differentiators

1. Local debug mode (`process` / `view-run` style)
2. `inference_backend` field with staged support metadata
3. Oversight / guardrail explanations in UI

### P2: Nice, But Only If Time Remains

1. Real EAGLE backend lane
2. Atropos-style trajectory export contract
3. Full trainer abstraction layer

## What We Should Build

## 1. Command-Center Layer

### Objective

Make the environment obviously align with:

- Multi-Agent Interactions
- World Modeling
- Long-Horizon Planning

### Required State

In `state` and `observation`, carry:

- scenario name
- scenario summary
- traffic level
- workload profile
- latency SLA
- quality SLA
- budget cap
- budget remaining
- incident status
- operating mode
- risk level
- available GPU pool
- active agents

### Required Agent Framing

Judge-facing roles:

- `Latency Agent`
- `Quality Agent`
- `Cost Agent`
- `Orchestrator`
- optional `Oversight Agent`

Internal implementation can still reuse existing planner/evaluator/optimizer logic.

### Required UI

Show:

- active scenario
- objective
- traffic pressure
- SLA targets
- budget pressure
- incident state
- mode
- risk
- GPU pool

### Success Condition

A judge can understand the environment in 10 seconds without reading code.

## 2. Training Proof

### Objective

Remove the strongest possible judge criticism:

> "Interesting system, but where is the actual learning?"

### Required

Run a real Task 1 GRPO training session with:

- `bf16=True`
- Trackio logging enabled
- at least `50-100` steps if time is tight
- ideally `200` steps

### Required Trackio Metrics

- `reward/mean`
- `reward/best_so_far`
- `reward/pre_gate`
- `reward/post_gate`
- `reward/quality_blocked`
- `quality/pass`
- `quality/near_miss`
- `throughput/speedup`
- `config/n_spec`
- `config/threshold`
- `train/loss`
- `train/kl`
- `train/clip_fraction`
- command-center metrics:
  - `ops/scenario`
  - `ops/traffic_level`
  - `ops/operating_mode`
  - `ops/incident_status`
  - `ops/risk_level`

### Success Condition

We have a public Trackio URL with an upward reward trend.

## 3. Hugging Face Space

### Objective

Make the project public, demoable, and judge-accessible.

### Required

- deploy the FastAPI app to a HF Space
- make `/health`, `/info`, `/reset`, `/step`, `/manual_step` work
- ensure the UI loads cleanly on first visit
- show Task 1 as the main stable demo lane
- show Tasks 2-4 as validated tiers in the scenario list

### Success Condition

A judge can open one URL and immediately use the environment.

## 4. Manual Mode

### Objective

Create the most trust-building demo moment.

### Required

- judge enters config
- real hardware run happens
- quality gate or reward is explained clearly
- if rejected, the reason is obvious

### Success Condition

A bad config visibly fails for a principled reason, not because of vague text.

## 5. Public Narrative

### Objective

Ensure docs and pitch match the upgraded environment.

### Required Changes

README, blog, and pitch should say:

- this is an autonomous inference-operations environment
- it trains agents to manage speed, quality, and cost tradeoffs
- it uses real A100/H100/B200 evaluation
- it supports multiple speculative-decoding methods conceptually
- current validated lane is HF-assisted decoding
- EAGLE / broader backends are staged extensions

### Success Condition

No one leaves thinking it is "just a benchmark harness."

## What Not To Build This Week

Do not spend the week on:

- full EAGLE training integration
- mock EAGLE performance claims in the main demo
- complete Atropos runtime integration
- final 70B NVFP4 production stack polish
- deep infra refactors
- brand-new RL stack rewrite

Those are real future directions, but not the best use of finale-week time.

## Recommended 5-Day Schedule

## Day 1: Lock The Environment Layer

### Backend

- finish scenario state and transitions
- finish agent role reframing
- expose operational context in API responses

### UI

- wire scenario / SLA / budget / incident panels
- make logs use the new role names

### End-of-day output

- environment looks like a command center, not a tuner

## Day 2: Training + Trackio

### Training

- run Task 1 GRPO smoke test
- confirm reward is non-flat
- log to Trackio

### UI / story

- connect training result references into README/blog copy

### End-of-day output

- public or near-public reward curve exists

## Day 3: HF Space + Reliability

### Deployment

- deploy to Hugging Face Space
- verify live endpoints
- verify browser rendering

### Reliability

- prepare fallback mode
- add clearer manual diagnostics

### End-of-day output

- stable public demo URL

## Day 4: Docs + Pitch + Backup

### Docs

- finalize README
- publish mini-blog

### Demo

- record backup videos:
  - manual mode
  - auto-run mode
  - Trackio dashboard

### End-of-day output

- complete narrative package

## Day 5: Final Polish

### Team run-through

- rehearse 3-minute pitch
- rehearse 2-minute Q&A
- tighten weak transitions
- remove confusing UI text

### End-of-day output

- pitch-ready project

## Exact Team Split

### Kshitiz

- backend state / API / deployment / Modal / Trackio / training

### Anshuman

- UI polish
- scenario cards
- clarity of logs
- dashboard feel

### Aditya

- README
- blog
- pitch script
- judge Q&A
- backup video orchestration

## Best One-Two Day Stretch If Time Gets Tight

If the team only has 1-2 strong build days, do these in order:

1. command-center environment layer
2. Trackio training proof
3. HF Space deployment

That combination is the highest score-per-day path.

## Judge-Defense Checklist

To make the project hard to dismiss, we need clear answers to:

### "Is this real?"

Answer with:

- live Modal runs
- real throughput
- manual mode

### "Is this just hyperparameter tuning?"

Answer with:

- dynamic scenarios
- SLA / budget / incident state
- multi-agent tradeoff framing
- long-horizon episodes

### "Does it learn?"

Answer with:

- Trackio reward curve
- trained policy improving over steps

### "Can it fail safely?"

Answer with:

- 95% quality gate
- anti-cheat
- explicit rejection reasons

### "Can we trust the demo?"

Answer with:

- manual mode
- public Space
- backup videos

## Final Recommendation

The best-of-best path is:

1. keep the validated backend
2. finish the command-center layer
3. get real Trackio proof
4. deploy the HF Space
5. tell the story as autonomous inference operations

EAGLE is useful as a **benchmark/reference and staged extension**.

Atropos is useful as a **developer-experience and compatibility reference**.

Neither should displace the core plan this week.
