"""
inference.py - speculate_forge
Baseline agent using an LLM to propose speculative decoding configs.
Logs in OpenEnv [START]/[STEP]/[END] format.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import requests
from openai import OpenAI

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:8000")
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-ai/DeepSeek-V3-0324")
BENCHMARK = "speculate_forge"

client = OpenAI(
    base_url=os.getenv("API_BASE_URL", "https://router.huggingface.co/v1"),
    api_key=os.getenv("HF_TOKEN", ""),
)

SYSTEM_PROMPT = """You are an expert in LLM inference optimization.
Your task: find speculative decoding configurations that maximize throughput
while maintaining output quality >= 95% reference match rate.

Respond ONLY with valid JSON:
{
  "candidate_configs": [
    {"num_speculative_tokens": 4, "acceptance_threshold": 0.65,
     "tree_depth": 1, "ngram_cache_size": 0, "adaptive_depth": false,
     "label": "strategy_name"},
    ...
  ],
  "phase": "exploration",
  "reasoning": "why these configs",
  "hypothesis": "what you expect"
}"""

TASK_NAMES = {
    1: "task1_easy_a100",
    2: "task2_medium_h100",
    3: "task3_medhard_h100",
    4: "task4_hard_b200",
}


def call_llm(obs: dict, history: list[dict[str, str]]) -> dict:
    prompt = f"""Observation:
baseline_throughput: {obs.get('baseline_throughput', 0):.1f} tok/s
best_throughput: {obs.get('best_throughput', 0):.1f} tok/s
trajectory: {obs.get('trajectory', [])}
feedback: {obs.get('evaluator_feedback', 'No feedback yet')}
bottleneck: {obs.get('bottleneck', 'unknown')}
iteration: {obs.get('iteration', 0)}
phase: {obs.get('phase', 'exploration')}
gpu: {obs.get('gpu', 'A100-80GB')}

Propose 3 candidate configs to try in parallel."""

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            *history[-6:],
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=800,
    )
    raw = resp.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def _fmt_error(val: object) -> str:
    if val in (None, ""):
        return "null"
    return str(val).replace("\n", " ")[:100]


def run_episode(task_level: int, base_url: str) -> float:
    task_name = TASK_NAMES[task_level]
    rewards: list[float] = []
    step_count = 0
    success = False
    history: list[dict[str, str]] = []

    print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    try:
        reset_response = requests.post(
            f"{base_url}/reset",
            params={"task_level": task_level},
            timeout=60,
        )
        reset_response.raise_for_status()
        obs = reset_response.json()["observation"]

        for i in range(10):
            action = call_llm(obs, history)
            history.append({"role": "assistant", "content": json.dumps(action)})

            step_response = requests.post(
                f"{base_url}/step",
                json=action,
                params={"task_level": task_level},
                timeout=180,
            )
            step_response.raise_for_status()
            payload = step_response.json()

            reward = round(float(payload["reward"]), 4)
            rewards.append(reward)
            step_count = i + 1
            obs = payload["observation"]
            done = payload["done"]
            err = payload.get("info", {}).get("violation")

            action_summary = json.dumps(
                {
                    "phase": action.get("phase"),
                    "n_configs": len(action.get("candidate_configs", [])),
                    "best_label": action.get("candidate_configs", [{}])[0].get(
                        "label", ""
                    ),
                }
            )

            print(
                f"[STEP] step={step_count} action={action_summary} "
                f"reward={reward:.4f} done={str(done).lower()} "
                f"error={_fmt_error(err)}",
                flush=True,
            )

            history.append(
                {
                    "role": "user",
                    "content": (
                        "Result: "
                        f"best={obs.get('best_throughput', 0):.1f}tok/s "
                        f"reward={reward:.4f} "
                        f"feedback={obs.get('evaluator_feedback', '')[:100]}"
                    ),
                }
            )

            if done or reward >= 1.0:
                break

        final_score = round(sum(rewards), 4)
        success = final_score > 0.0
        return final_score

    except Exception as exc:
        print(f"Error in episode: {exc}", file=sys.stderr)
        return 0.0
    finally:
        final_score = round(sum(rewards), 4)
        rewards_str = ",".join(f"{reward:.4f}" for reward in rewards)
        print(
            f"[END] success={str(success).lower()} steps={step_count} "
            f"score={final_score:.4f} rewards={rewards_str}",
            flush=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-level", type=int, choices=[1, 2, 3, 4])
    parser.add_argument("--base-url", default=ENV_BASE_URL)
    args = parser.parse_args()

    task_levels = [args.task_level] if args.task_level else [1, 2, 3, 4]
    scores = [run_episode(task_level, args.base_url) for task_level in task_levels]
    return 0 if all(score > 0 for score in scores) else 1


if __name__ == "__main__":
    raise SystemExit(main())
