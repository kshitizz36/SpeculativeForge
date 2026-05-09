# SpeculateForge: Teaching Agents to Optimize LLM Inference on Real Hardware

*Draft for Hugging Face publication by Kshitiz Yadav | April 2026*

## The Problem With Speculative Decoding

Speculative decoding is one of the most promising inference optimizations in
the modern LLM stack. A fast draft model proposes tokens and a larger target
model verifies them in parallel, which can reduce serving cost and latency
substantially.

The hard part is tuning it. Teams still spend days or weeks exploring the
right speculative depth, acceptance threshold, tree depth, branching, and
caching strategy for a specific model pair and hardware target.

## What We Built

SpeculateForge is an OpenEnv-style environment for autonomous speculative
decoding optimization. The system is designed around four cooperating roles:

- Planner proposes candidate configs.
- Scheduler dispatches configs in parallel to GPU workers.
- Evaluator ranks results and diagnoses the bottleneck.
- Optimizer decides when to explore and when to exploit.

The goal is simple: train an agent that becomes measurably better at proposing
high-throughput configurations while respecting a hard quality floor.

## Why This Environment Matters

This project sits at the overlap of multi-agent coordination and real-world
world modeling. The agent does not optimize a toy simulator. It has to reason
about genuine hardware regimes, changing bottlenecks, and measurable tradeoffs
between speed, quality, and cost.

## Reward Design

The reward combines speedup and output quality, with three anti-cheat guards:

1. Quality must remain at or above 95 percent.
2. Acceptance-rate exploits are rejected.
3. Throughput beyond hardware ceilings is invalidated.

This keeps the optimization grounded in physics instead of score hacking.

## Training Story

Our training plan uses GRPO with `Qwen/Qwen2.5-0.5B-Instruct` and a reward
signal coming from the environment itself. The Colab demo includes Trackio
logging so judges can open a public dashboard and see reward curves improve
over time.

One implementation detail matters a lot: GRPO must run with `bf16=True` to
match inference precision. The training notebook is scaffolded around that
setting from the start.

## Try It

The repo includes:

- A FastAPI environment shell
- OpenEnv metadata and packaging files
- A Trackio-ready training scaffold
- A draft Colab notebook
- A draft UI for manual and agent-driven interaction

Next steps are wiring the real Modal workers, validating live GPU runs, and
publishing the dashboard and Space URLs before the finale.
