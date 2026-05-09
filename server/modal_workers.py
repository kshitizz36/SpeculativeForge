from __future__ import annotations

import copy
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import modal

from graders.speculation_grader import quality_match_rate, score_pairs
from speculate_forge.models import SpeculateConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_CORPUS_DIR = PROJECT_ROOT / "corpus"
REMOTE_ASSET_ROOT = Path("/root/speculate_forge_assets")
REMOTE_CORPUS_DIR = REMOTE_ASSET_ROOT / "corpus"
REMOTE_CACHE_DIR = Path("/cache")

APP_NAME = "speculate-forge-workers"
HF_SECRET_NAME = os.getenv("SPECFORGE_MODAL_HF_SECRET", "hf-token")
MODEL_CACHE_VOLUME_NAME = os.getenv(
    "SPECFORGE_MODAL_VOLUME",
    "speculate-forge-model-cache",
)

MODEL_CACHE_VOLUME = modal.Volume.from_name(
    MODEL_CACHE_VOLUME_NAME,
    create_if_missing=True,
)
HF_SECRET = modal.Secret.from_name(HF_SECRET_NAME)

A100_MIN_CONTAINERS = int(os.getenv("SPECFORGE_A100_MIN_CONTAINERS", "2"))
H100_MIN_CONTAINERS = int(os.getenv("SPECFORGE_H100_MIN_CONTAINERS", "0"))
B200_MIN_CONTAINERS = int(os.getenv("SPECFORGE_B200_MIN_CONTAINERS", "0"))

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "transformers>=4.49,<5",
        "accelerate>=1.0,<2",
        "pydantic>=2.6,<3",
        "sentencepiece>=0.2,<1",
        "safetensors>=0.4,<1",
        "huggingface-hub>=0.29,<1",
        "hf_transfer>=0.1.9,<1",
    )
    .run_commands(
        "python -m pip install 'torch==2.10.0' --index-url https://download.pytorch.org/whl/cu128"
    )
    .env(
        {
            "HF_HOME": str(REMOTE_CACHE_DIR / "hf"),
            "HUGGINGFACE_HUB_CACHE": str(REMOTE_CACHE_DIR / "hf" / "hub"),
            "TRANSFORMERS_CACHE": str(REMOTE_CACHE_DIR / "hf" / "transformers"),
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .add_local_python_source("graders", "speculate_forge")
    .add_local_dir(LOCAL_CORPUS_DIR, remote_path=str(REMOTE_CORPUS_DIR))
)

app = modal.App(APP_NAME)


class ModalWorkerUnavailableError(RuntimeError):
    """Raised when a worker is called for a task tier we have not enabled yet."""


@dataclass(frozen=True)
class WorkerTaskSpec:
    task_key: str
    slug: str
    gpu_name: str
    target_model: str
    draft_model: str
    attn_implementation: str | None
    default_max_new_tokens: int
    baseline_expected_range: tuple[float, float]
    supports_transformers_assisted: bool
    prompt_file: str
    rollout_status: str
    notes: tuple[str, ...] = ()


TASK_SPECS: dict[str, WorkerTaskSpec] = {
    "task1": WorkerTaskSpec(
        task_key="task1",
        slug="task1_easy_a100",
        gpu_name="A100-80GB",
        target_model=os.getenv(
            "SPECFORGE_TASK1_TARGET_MODEL",
            "Qwen/Qwen2.5-3B-Instruct",
        ),
        draft_model=os.getenv(
            "SPECFORGE_TASK1_DRAFT_MODEL",
            "Qwen/Qwen2.5-0.5B-Instruct",
        ),
        attn_implementation=os.getenv(
            "SPECFORGE_TASK1_ATTN_IMPLEMENTATION",
            "sdpa",
        ),
        default_max_new_tokens=24,
        baseline_expected_range=(40.0, 60.0),
        supports_transformers_assisted=True,
        prompt_file="reference_prompts.jsonl",
        rollout_status="live_validated",
        notes=(
            "Phase B is focused on a real A100 baseline first.",
            "This worker uses Transformers assisted generation with Universal Assisted Decoding.",
            "The default Phase B A100 path uses open-access Qwen models so benchmarking is not blocked on gated-repo approval.",
            "A100 defaults to Transformers SDPA so PyTorch can choose the fastest compatible Ampere kernel automatically.",
            "A100 is an Ampere GPU, so FlashAttention-2 is the relevant attention-kernel family here rather than FlashAttention-3.",
            "The Phase C A100 corpus is input-grounded and deterministic so reward noise comes from hardware choices rather than open-ended prose variation.",
        ),
    ),
    "task2": WorkerTaskSpec(
        task_key="task2",
        slug="task2_medium_h100",
        gpu_name="H100-80GB",
        target_model=os.getenv(
            "SPECFORGE_TASK2_TARGET_MODEL",
            "Qwen/Qwen2.5-7B-Instruct",
        ),
        draft_model=os.getenv(
            "SPECFORGE_TASK2_DRAFT_MODEL",
            "Qwen/Qwen2.5-1.5B-Instruct",
        ),
        attn_implementation=os.getenv(
            "SPECFORGE_TASK2_ATTN_IMPLEMENTATION",
            "sdpa",
        ),
        default_max_new_tokens=24,
        baseline_expected_range=(60.0, 80.0),
        supports_transformers_assisted=True,
        prompt_file="task2_reference_prompts.jsonl",
        rollout_status="live_validated",
        notes=(
            "Task 2 is the FP8/H100 regime and now has a real H100 validation path using an open Qwen pair.",
            "Use H100! on Modal to avoid automatic H200 upgrades during benchmarking.",
            "H100 defaults to SDPA for stability in the current build; FlashAttention-3 can be re-enabled via env override once the stack is pinned.",
        ),
    ),
    "task3": WorkerTaskSpec(
        task_key="task3",
        slug="task3_medium_hard_h100",
        gpu_name="H100-80GB",
        target_model=os.getenv(
            "SPECFORGE_TASK3_TARGET_MODEL",
            "Qwen/Qwen2.5-7B-Instruct",
        ),
        draft_model=os.getenv(
            "SPECFORGE_TASK3_DRAFT_MODEL",
            "Qwen/Qwen2.5-1.5B-Instruct",
        ),
        attn_implementation=os.getenv(
            "SPECFORGE_TASK3_ATTN_IMPLEMENTATION",
            "sdpa",
        ),
        default_max_new_tokens=16,
        baseline_expected_range=(60.0, 80.0),
        supports_transformers_assisted=True,
        prompt_file="task3_reference_prompts.jsonl",
        rollout_status="live_validated",
        notes=(
            "Task 3 is the tree-speculation H100 regime and now has a real H100 validation path using an open Qwen pair.",
            "Use H100! on Modal to avoid automatic H200 upgrades during benchmarking.",
            "This task is expected to emphasize deeper trees, branching, and adaptive speculative depth.",
            "H100 defaults to SDPA for stability in the current build; FlashAttention-3 can be re-enabled via env override once the stack is pinned.",
        ),
    ),
    "task4": WorkerTaskSpec(
        task_key="task4",
        slug="task4_hard_b200",
        gpu_name="B200-180GB",
        target_model=os.getenv(
            "SPECFORGE_TASK4_TARGET_MODEL",
            "Qwen/Qwen2.5-32B-Instruct",
        ),
        draft_model=os.getenv(
            "SPECFORGE_TASK4_DRAFT_MODEL",
            "Qwen/Qwen2.5-3B-Instruct",
        ),
        attn_implementation=os.getenv(
            "SPECFORGE_TASK4_ATTN_IMPLEMENTATION",
            "sdpa",
        ),
        default_max_new_tokens=16,
        baseline_expected_range=(35.0, 80.0),
        supports_transformers_assisted=True,
        prompt_file="task4_reference_prompts.jsonl",
        rollout_status="live_validated",
        notes=(
            "Task 4 now has a real B200 validation path using an open Qwen pair so backend bring-up is not blocked on gated-repo approval.",
            "This is a validation stack for the B200 lane, not the final Llama-70B NVFP4 frontier serving path.",
            "FlashAttention-4 is promising on Blackwell, but this validation worker stays on the current Transformers-native path until a dedicated B200 serving stack is wired.",
            "The final 70B/NVFP4 frontier path is still expected to graduate to a dedicated B200 serving stack after this validation phase.",
        ),
    ),
}


def _prompt_path(prompt_file: str) -> Path:
    return REMOTE_CORPUS_DIR / prompt_file


def _load_prompts(prompt_limit: int, *, prompt_file: str) -> list[str]:
    prompts: list[str] = []
    prompt_path = _prompt_path(prompt_file)
    with prompt_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if len(prompts) >= prompt_limit:
                break
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            prompt = str(record.get("prompt", "")).strip()
            if prompt:
                prompts.append(prompt)
    if not prompts:
        raise RuntimeError(f"No prompts found in {prompt_path}")
    return prompts


def _decode_new_text(tokenizer, sequences, prompt_length: int) -> str:
    new_tokens = sequences[0][prompt_length:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def _generated_token_count(sequences, prompt_length: int) -> int:
    return int(sequences.shape[-1] - prompt_length)


def _preview_text(text: str, *, limit: int = 120) -> str:
    normalized = " ".join(text.strip().split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3] + "..."


class _TransformersAssistedWorker:
    default_task_key = "task1"
    supported_task_keys = ("task1",)

    @property
    def spec(self) -> WorkerTaskSpec:
        return self._resolve_spec(self.default_task_key)

    def _resolve_spec(self, task_key: str | None = None) -> WorkerTaskSpec:
        selected_key = task_key or self.default_task_key
        if selected_key not in self.supported_task_keys:
            raise ModalWorkerUnavailableError(
                f"Worker does not support task '{selected_key}'. "
                f"Supported tasks: {', '.join(self.supported_task_keys)}."
            )
        return TASK_SPECS[selected_key]

    def _runtime_matches_task_spec(self, task_key: str | None = None) -> bool:
        requested = self._resolve_spec(task_key)
        loaded = self.spec
        return (
            requested.target_model == loaded.target_model
            and requested.draft_model == loaded.draft_model
            and requested.attn_implementation == loaded.attn_implementation
        )

    @modal.enter()
    def load_models(self) -> None:
        self._baseline_cache: dict[tuple[int, int], dict[str, Any]] = {}
        if not self.spec.supports_transformers_assisted:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError(
                "HF_TOKEN was not found in the Modal container. "
                f"Create the '{HF_SECRET_NAME}' secret before deploying."
            )

        self.torch = torch
        self.device = "cuda"
        self.target_tokenizer = AutoTokenizer.from_pretrained(
            self.spec.target_model,
            token=token,
            cache_dir=str(REMOTE_CACHE_DIR / "hf"),
        )
        self.assistant_tokenizer = AutoTokenizer.from_pretrained(
            self.spec.draft_model,
            token=token,
            cache_dir=str(REMOTE_CACHE_DIR / "hf"),
        )

        self.target_model = AutoModelForCausalLM.from_pretrained(
            self.spec.target_model,
            token=token,
            cache_dir=str(REMOTE_CACHE_DIR / "hf"),
            dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
            attn_implementation=self.spec.attn_implementation,
        )
        self.assistant_model = AutoModelForCausalLM.from_pretrained(
            self.spec.draft_model,
            token=token,
            cache_dir=str(REMOTE_CACHE_DIR / "hf"),
            dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
            attn_implementation=self.spec.attn_implementation,
        )

        self._ensure_pad_tokens()
        self.target_generation_config = self._benchmark_generation_config(
            GenerationConfig.from_model_config(self.target_model.config),
            pad_token_id=self.target_tokenizer.pad_token_id,
            eos_token_id=self.target_tokenizer.eos_token_id,
        )
        self.assistant_generation_config = self._benchmark_generation_config(
            GenerationConfig.from_model_config(self.assistant_model.config),
            pad_token_id=self.assistant_tokenizer.pad_token_id,
            eos_token_id=self.assistant_tokenizer.eos_token_id,
        )
        self._shared_tokenizer = self._tokenizers_match()
        self._assistant_tokenizer_mode = (
            "tokenizer_only" if self._shared_tokenizer else "uad"
        )
        self._warmup()
        MODEL_CACHE_VOLUME.commit()

    def _ensure_pad_tokens(self) -> None:
        for tokenizer in (self.target_tokenizer, self.assistant_tokenizer):
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token

    def _benchmark_generation_config(
        self,
        generation_config,
        *,
        pad_token_id: int | None,
        eos_token_id: int | list[int] | None,
    ):
        sanitized = copy.deepcopy(generation_config)
        sanitized.do_sample = False
        sanitized.num_beams = 1
        sanitized.pad_token_id = pad_token_id
        sanitized.eos_token_id = eos_token_id
        for attr in (
            "temperature",
            "top_p",
            "min_p",
            "typical_p",
            "penalty_alpha",
            "top_k",
        ):
            if hasattr(sanitized, attr):
                setattr(sanitized, attr, None)
        if hasattr(sanitized, "repetition_penalty"):
            sanitized.repetition_penalty = 1.0
        return sanitized

    def _warmup(self) -> None:
        prompt = "Benchmark warmup prompt for speculative decoding."
        self._generate_baseline_text(prompt, max_new_tokens=8)
        self._generate_speculative_text(
            prompt,
            SpeculateConfig(
                num_speculative_tokens=4,
                acceptance_threshold=0.5,
                label="warmup",
            ),
            max_new_tokens=8,
        )

    def _tokenizers_match(self) -> bool:
        if self.target_tokenizer.__class__ is not self.assistant_tokenizer.__class__:
            return False
        if self.target_tokenizer.vocab_size != self.assistant_tokenizer.vocab_size:
            return False
        return self.target_tokenizer.get_vocab() == self.assistant_tokenizer.get_vocab()

    def _unsupported_config_dimensions(
        self,
        config: SpeculateConfig,
        *,
        task_key: str | None = None,
    ) -> list[str]:
        _ = self._resolve_spec(task_key)
        unsupported: list[str] = []
        if config.tree_depth > 1:
            unsupported.append("tree_depth")
        if config.tree_branching > 1:
            unsupported.append("tree_branching")
        return unsupported

    def _prompt_lookup_kwargs(self, config: SpeculateConfig) -> dict[str, int] | None:
        if config.ngram_cache_size <= 0:
            return None

        prompt_lookup_num_tokens = max(
            1,
            min(
                8,
                config.num_speculative_tokens + 1,
                max(2, config.ngram_cache_size // 256),
            ),
        )
        max_matching_ngram_size = max(2, min(6, config.tree_depth + 1))
        return {
            "prompt_lookup_num_tokens": prompt_lookup_num_tokens,
            "max_matching_ngram_size": max_matching_ngram_size,
        }

    def _generate_baseline_text(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
    ) -> dict[str, Any]:
        inputs = self.target_tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_length = int(inputs["input_ids"].shape[-1])

        with self.torch.inference_mode():
            self.torch.cuda.synchronize()
            started = time.perf_counter()
            outputs = self.target_model.generate(
                **inputs,
                generation_config=copy.deepcopy(self.target_generation_config),
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.target_tokenizer.pad_token_id,
                eos_token_id=self.target_tokenizer.eos_token_id,
                return_dict_in_generate=True,
            )
            self.torch.cuda.synchronize()
            elapsed = time.perf_counter() - started

        text = _decode_new_text(self.target_tokenizer, outputs.sequences, prompt_length)
        token_count = _generated_token_count(outputs.sequences, prompt_length)
        return {
            "text": text,
            "generated_tokens": token_count,
            "elapsed_sec": elapsed,
            "throughput_tok_s": (token_count / elapsed) if elapsed else 0.0,
        }

    def _generate_with_assistant(
        self,
        inputs: dict[str, Any],
        generation_kwargs: dict[str, Any],
    ):
        tried_modes: set[str] = set()
        modes = [self._assistant_tokenizer_mode]
        if self._assistant_tokenizer_mode == "tokenizer_only":
            modes.append("uad")
        else:
            modes.append("tokenizer_only")

        last_error: Exception | None = None
        for mode in modes:
            if mode in tried_modes:
                continue
            tried_modes.add(mode)

            call_kwargs = dict(generation_kwargs)
            call_kwargs["tokenizer"] = self.target_tokenizer
            if mode == "uad":
                call_kwargs["assistant_tokenizer"] = self.assistant_tokenizer

            try:
                outputs = self.target_model.generate(**inputs, **call_kwargs)
                self._assistant_tokenizer_mode = mode
                return outputs
            except ValueError as exc:
                message = str(exc)
                last_error = exc
                needs_uad = "Please provide `tokenizer` and `assistant_tokenizer`" in message
                tokenizer_only = "assistant_tokenizer" in message and "not required" in message
                if (mode == "tokenizer_only" and needs_uad) or (
                    mode == "uad" and tokenizer_only
                ):
                    continue
                raise

        if last_error is not None:
            raise last_error
        raise RuntimeError("Assisted generation failed before any backend attempt completed.")

    def _generate_speculative_text(
        self,
        prompt: str,
        config: SpeculateConfig,
        *,
        max_new_tokens: int,
    ) -> dict[str, Any]:
        inputs = self.target_tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_length = int(inputs["input_ids"].shape[-1])
        prompt_lookup = self._prompt_lookup_kwargs(config)
        generation_strategy = "assistant_model"
        generation_kwargs: dict[str, Any] = {
            "do_sample": config.draft_temperature > 0,
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.target_tokenizer.pad_token_id,
            "eos_token_id": self.target_tokenizer.eos_token_id,
            "return_dict_in_generate": True,
        }

        if prompt_lookup is not None:
            generation_strategy = "prompt_lookup"
            generation_kwargs.update(prompt_lookup)
        else:
            # This is the closest honest mapping from the project config to current
            # Transformers assisted decoding knobs on A100.
            assistant_generation = copy.deepcopy(self.assistant_generation_config)
            assistant_generation.num_assistant_tokens = config.num_speculative_tokens
            assistant_generation.assistant_confidence_threshold = (
                config.acceptance_threshold
            )
            assistant_generation.num_assistant_tokens_schedule = (
                "heuristic_transient" if config.adaptive_depth else "constant"
            )
            self.assistant_model.generation_config = assistant_generation
            generation_kwargs["assistant_model"] = self.assistant_model

        if config.draft_temperature > 0:
            generation_kwargs["temperature"] = config.draft_temperature
        else:
            generation_kwargs["generation_config"] = copy.deepcopy(
                self.target_generation_config
            )

        with self.torch.inference_mode():
            self.torch.cuda.synchronize()
            started = time.perf_counter()
            if prompt_lookup is not None:
                outputs = self.target_model.generate(
                    **inputs,
                    **generation_kwargs,
                )
            else:
                outputs = self._generate_with_assistant(inputs, generation_kwargs)
            self.torch.cuda.synchronize()
            elapsed = time.perf_counter() - started

        text = _decode_new_text(self.target_tokenizer, outputs.sequences, prompt_length)
        token_count = _generated_token_count(outputs.sequences, prompt_length)
        return {
            "text": text,
            "generated_tokens": token_count,
            "elapsed_sec": elapsed,
            "throughput_tok_s": (token_count / elapsed) if elapsed else 0.0,
            "generation_strategy": generation_strategy,
            "applied_generation": {
                "strategy": generation_strategy,
                "num_assistant_tokens": config.num_speculative_tokens,
                "assistant_confidence_threshold": config.acceptance_threshold,
                "num_assistant_tokens_schedule": (
                    "heuristic_transient" if config.adaptive_depth else "constant"
                ),
                "prompt_lookup_num_tokens": (
                    prompt_lookup["prompt_lookup_num_tokens"]
                    if prompt_lookup is not None
                    else None
                ),
                "max_matching_ngram_size": (
                    prompt_lookup["max_matching_ngram_size"]
                    if prompt_lookup is not None
                    else None
                ),
                "temperature": config.draft_temperature,
            },
        }

    def _ensure_baseline_cache(
        self,
        task_key: str,
        prompt_limit: int,
        max_new_tokens: int,
    ) -> dict[str, Any]:
        spec = self._resolve_spec(task_key)
        key = (task_key, prompt_limit, max_new_tokens)
        cached = self._baseline_cache.get(key)
        if cached is not None:
            return cached

        prompts = _load_prompts(prompt_limit, prompt_file=spec.prompt_file)
        runs = [
            self._generate_baseline_text(prompt, max_new_tokens=max_new_tokens)
            for prompt in prompts
        ]
        total_elapsed = sum(run["elapsed_sec"] for run in runs)
        total_tokens = sum(run["generated_tokens"] for run in runs)
        average_throughput = (total_tokens / total_elapsed) if total_elapsed else 0.0
        average_output_length = (
            total_tokens / len(runs) if runs else 0.0
        )

        baseline = {
            "task": spec.slug,
            "gpu": spec.gpu_name,
            "ready": True,
            "model": spec.target_model,
            "draft_model": spec.draft_model,
            "attention_backend": spec.attn_implementation,
            "mode": "vanilla_target_only",
            "prompt_count": len(prompts),
            "max_new_tokens": max_new_tokens,
            "generated_tokens_total": total_tokens,
            "elapsed_sec_total": total_elapsed,
            "throughput_tok_s": average_throughput,
            "average_output_tokens": average_output_length,
            "expected_range_tok_s": list(spec.baseline_expected_range),
            "reference_outputs": [run["text"] for run in runs],
            "prompt_file": spec.prompt_file,
            "rollout_status": spec.rollout_status,
            "notes": list(spec.notes),
        }
        self._baseline_cache[key] = baseline
        return baseline

    def _benchmark_prompts(
        self,
        prompts: Iterable[str],
        config: SpeculateConfig,
        *,
        max_new_tokens: int,
    ) -> dict[str, Any]:
        runs = [
            self._generate_speculative_text(prompt, config, max_new_tokens=max_new_tokens)
            for prompt in prompts
        ]
        total_elapsed = sum(run["elapsed_sec"] for run in runs)
        total_tokens = sum(run["generated_tokens"] for run in runs)
        throughput = (total_tokens / total_elapsed) if total_elapsed else 0.0
        outputs = [run["text"] for run in runs]
        return {
            "runs": runs,
            "outputs": outputs,
            "generated_tokens_total": total_tokens,
            "elapsed_sec_total": total_elapsed,
            "throughput_tok_s": throughput,
        }

    def _not_ready_payload(self, task_key: str | None = None) -> dict[str, Any]:
        spec = self._resolve_spec(task_key)
        return {
            "task": spec.slug,
            "gpu": spec.gpu_name,
            "ready": False,
            "prompt_file": spec.prompt_file,
            "rollout_status": spec.rollout_status,
            "target_model": spec.target_model,
            "draft_model": spec.draft_model,
            "attention_backend": spec.attn_implementation,
            "notes": list(spec.notes),
        }


@app.cls(
    image=image,
    gpu="A100-80GB",
    cpu=8,
    memory=65536,
    min_containers=A100_MIN_CONTAINERS,
    scaledown_window=600,
    secrets=[HF_SECRET],
    volumes={str(REMOTE_CACHE_DIR): MODEL_CACHE_VOLUME},
    timeout=60 * 30,
)
class SpeculationA100(_TransformersAssistedWorker):
    default_task_key = "task1"
    supported_task_keys = ("task1",)

    @modal.method()
    def baseline(
        self,
        task_key: str = "task1",
        prompt_limit: int = 5,
        max_new_tokens: int | None = None,
    ) -> dict[str, Any]:
        spec = self._resolve_spec(task_key)
        max_new_tokens = max_new_tokens or spec.default_max_new_tokens
        return self._ensure_baseline_cache(task_key, prompt_limit, max_new_tokens)

    @modal.method()
    def benchmark_config(
        self,
        config: dict[str, Any],
        task_key: str = "task1",
        prompt_limit: int = 5,
        max_new_tokens: int | None = None,
        reference_outputs: list[str] | None = None,
        baseline_tok_s: float | None = None,
    ) -> dict[str, Any]:
        spec = self._resolve_spec(task_key)
        max_new_tokens = max_new_tokens or spec.default_max_new_tokens
        parsed = SpeculateConfig.model_validate(config)
        if reference_outputs is not None and baseline_tok_s is not None:
            baseline = {
                "throughput_tok_s": float(baseline_tok_s),
                "reference_outputs": reference_outputs,
            }
        else:
            baseline = self._ensure_baseline_cache(task_key, prompt_limit, max_new_tokens)
        prompts = _load_prompts(prompt_limit, prompt_file=spec.prompt_file)
        benchmark = self._benchmark_prompts(prompts, parsed, max_new_tokens=max_new_tokens)

        quality = quality_match_rate(
            baseline["reference_outputs"],
            benchmark["outputs"],
        )
        quality_scores = score_pairs(
            baseline["reference_outputs"],
            benchmark["outputs"],
        )
        speedup = (
            benchmark["throughput_tok_s"] / baseline["throughput_tok_s"]
            if baseline["throughput_tok_s"]
            else 0.0
        )
        unsupported = self._unsupported_config_dimensions(parsed)
        exact_match_count = sum(
            1 for item in quality_scores if bool(item["exact_match"])
        )
        quality_breakdown = [
            {
                "prompt_index": index,
                "token_overlap": round(float(score["token_overlap"]), 4),
                "exact_match": bool(score["exact_match"]),
                "reference_preview": _preview_text(reference),
                "candidate_preview": _preview_text(candidate),
            }
            for index, (score, reference, candidate) in enumerate(
                zip(
                    quality_scores,
                    baseline["reference_outputs"],
                    benchmark["outputs"],
                    strict=False,
                )
            )
        ]

        return {
            "task": spec.slug,
            "gpu": spec.gpu_name,
            "ready": True,
            "target_model": spec.target_model,
            "draft_model": spec.draft_model,
            "config": parsed.model_dump(),
            "throughput_tok_s": benchmark["throughput_tok_s"],
            "baseline_tok_s": baseline["throughput_tok_s"],
            "speedup": speedup,
            "quality_match_rate": quality,
            "exact_match_rate": (
                exact_match_count / len(quality_scores) if quality_scores else 0.0
            ),
            "exact_match_count": exact_match_count,
            "quality_breakdown": quality_breakdown,
            "acceptance_rate": None,
            "acceptance_rate_note": (
                "Transformers assisted generation does not expose accepted-token "
                "counts directly, so acceptance_rate is unavailable in Phase B."
            ),
            "average_output_tokens": (
                benchmark["generated_tokens_total"] / prompt_limit if prompt_limit else 0.0
            ),
            "elapsed_sec_total": benchmark["elapsed_sec_total"],
            "generated_tokens_total": benchmark["generated_tokens_total"],
            "attention_backend": spec.attn_implementation,
            "prompt_file": spec.prompt_file,
            "rollout_status": spec.rollout_status,
            "generation_strategy": (
                benchmark["runs"][0]["generation_strategy"]
                if benchmark["runs"]
                else "assistant_model"
            ),
            "applied_generation": benchmark["runs"][0]["applied_generation"] if benchmark["runs"] else {},
            "unsupported_config_dimensions": unsupported,
            "notes": list(spec.notes)
            + (
                [
                    "For the A100 worker, ngram_cache_size maps to Transformers "
                    "prompt-lookup assisted decoding rather than a literal KV n-gram cache."
                ]
                if parsed.ngram_cache_size > 0
                else []
            )
            + (
                [
                    "Unsupported config dimensions were ignored for the A100 worker: "
                    + ", ".join(unsupported)
                ]
                if unsupported
                else []
            ),
        }


@app.cls(
    image=image,
    gpu="H100!",
    cpu=8,
    memory=65536,
    min_containers=H100_MIN_CONTAINERS,
    scaledown_window=600,
    secrets=[HF_SECRET],
    volumes={str(REMOTE_CACHE_DIR): MODEL_CACHE_VOLUME},
    timeout=60 * 30,
)
class SpeculationH100(_TransformersAssistedWorker):
    default_task_key = "task2"
    supported_task_keys = ("task2", "task3")

    @modal.method()
    def baseline(
        self,
        task_key: str = "task2",
        prompt_limit: int = 5,
        max_new_tokens: int | None = None,
    ) -> dict[str, Any]:
        spec = self._resolve_spec(task_key)
        if not self._runtime_matches_task_spec(task_key):
            payload = self._not_ready_payload(task_key)
            payload["notes"] = list(payload["notes"]) + [
                "Task 2 and Task 3 currently share one H100 runtime. "
                "If you override their models independently, deploy separate workers "
                "or align the task-specific model env vars first."
            ]
            return payload
        max_new_tokens = max_new_tokens or spec.default_max_new_tokens
        return self._ensure_baseline_cache(task_key, prompt_limit, max_new_tokens)

    @modal.method()
    def benchmark_config(
        self,
        config: dict[str, Any],
        task_key: str = "task2",
        prompt_limit: int = 5,
        max_new_tokens: int | None = None,
        reference_outputs: list[str] | None = None,
        baseline_tok_s: float | None = None,
    ) -> dict[str, Any]:
        spec = self._resolve_spec(task_key)
        if not self._runtime_matches_task_spec(task_key):
            payload = self._not_ready_payload(task_key)
            payload["notes"] = list(payload["notes"]) + [
                "Task 2 and Task 3 currently share one H100 runtime. "
                "If you override their models independently, deploy separate workers "
                "or align the task-specific model env vars first."
            ]
            return payload

        max_new_tokens = max_new_tokens or spec.default_max_new_tokens
        parsed = SpeculateConfig.model_validate(config)
        if reference_outputs is not None and baseline_tok_s is not None:
            baseline = {
                "throughput_tok_s": float(baseline_tok_s),
                "reference_outputs": reference_outputs,
            }
        else:
            baseline = self._ensure_baseline_cache(task_key, prompt_limit, max_new_tokens)
        prompts = _load_prompts(prompt_limit, prompt_file=spec.prompt_file)
        benchmark = self._benchmark_prompts(prompts, parsed, max_new_tokens=max_new_tokens)

        quality = quality_match_rate(
            baseline["reference_outputs"],
            benchmark["outputs"],
        )
        quality_scores = score_pairs(
            baseline["reference_outputs"],
            benchmark["outputs"],
        )
        speedup = (
            benchmark["throughput_tok_s"] / baseline["throughput_tok_s"]
            if baseline["throughput_tok_s"]
            else 0.0
        )
        unsupported = self._unsupported_config_dimensions(parsed, task_key=task_key)
        exact_match_count = sum(
            1 for item in quality_scores if bool(item["exact_match"])
        )
        quality_breakdown = [
            {
                "prompt_index": index,
                "token_overlap": round(float(score["token_overlap"]), 4),
                "exact_match": bool(score["exact_match"]),
                "reference_preview": _preview_text(reference),
                "candidate_preview": _preview_text(candidate),
            }
            for index, (score, reference, candidate) in enumerate(
                zip(
                    quality_scores,
                    baseline["reference_outputs"],
                    benchmark["outputs"],
                    strict=False,
                )
            )
        ]

        notes = list(spec.notes)
        if parsed.ngram_cache_size > 0:
            notes.append(
                "On the current H100 validation path, ngram_cache_size maps to Transformers "
                "prompt-lookup assisted decoding rather than a literal KV n-gram cache."
            )
        if unsupported:
            notes.append(
                "Unsupported config dimensions were ignored for the H100 validation path: "
                + ", ".join(unsupported)
            )
        if task_key == "task3":
            notes.append(
                "Task 3 currently validates the H100 search loop on real hardware, "
                "but explicit tree branching is still staged for a later serving stack."
            )

        return {
            "task": spec.slug,
            "gpu": spec.gpu_name,
            "ready": True,
            "target_model": spec.target_model,
            "draft_model": spec.draft_model,
            "config": parsed.model_dump(),
            "throughput_tok_s": benchmark["throughput_tok_s"],
            "baseline_tok_s": baseline["throughput_tok_s"],
            "speedup": speedup,
            "quality_match_rate": quality,
            "exact_match_rate": (
                exact_match_count / len(quality_scores) if quality_scores else 0.0
            ),
            "exact_match_count": exact_match_count,
            "quality_breakdown": quality_breakdown,
            "acceptance_rate": None,
            "acceptance_rate_note": (
                "Transformers assisted generation does not expose accepted-token "
                "counts directly, so acceptance_rate is unavailable in the current H100 path."
            ),
            "average_output_tokens": (
                benchmark["generated_tokens_total"] / prompt_limit if prompt_limit else 0.0
            ),
            "elapsed_sec_total": benchmark["elapsed_sec_total"],
            "generated_tokens_total": benchmark["generated_tokens_total"],
            "attention_backend": spec.attn_implementation,
            "prompt_file": spec.prompt_file,
            "rollout_status": spec.rollout_status,
            "generation_strategy": (
                benchmark["runs"][0]["generation_strategy"]
                if benchmark["runs"]
                else "assistant_model"
            ),
            "applied_generation": benchmark["runs"][0]["applied_generation"] if benchmark["runs"] else {},
            "unsupported_config_dimensions": unsupported,
            "notes": notes,
        }


@app.cls(
    image=image,
    gpu="B200",
    cpu=8,
    memory=65536,
    min_containers=B200_MIN_CONTAINERS,
    scaledown_window=600,
    secrets=[HF_SECRET],
    volumes={str(REMOTE_CACHE_DIR): MODEL_CACHE_VOLUME},
    timeout=60 * 45,
)
class SpeculationB200(_TransformersAssistedWorker):
    default_task_key = "task4"
    supported_task_keys = ("task4",)

    @modal.method()
    def baseline(
        self,
        task_key: str = "task4",
        prompt_limit: int = 5,
        max_new_tokens: int | None = None,
    ) -> dict[str, Any]:
        spec = self._resolve_spec(task_key)
        if not self._runtime_matches_task_spec(task_key):
            payload = self._not_ready_payload(task_key)
            payload["notes"] = list(payload["notes"]) + [
                "The current B200 runtime does not match the requested Task 4 model configuration. "
                "Align the Task 4 env overrides and redeploy before running live benchmarks."
            ]
            return payload
        max_new_tokens = max_new_tokens or spec.default_max_new_tokens
        return self._ensure_baseline_cache(task_key, prompt_limit, max_new_tokens)

    @modal.method()
    def benchmark_config(
        self,
        config: dict[str, Any],
        task_key: str = "task4",
        prompt_limit: int = 5,
        max_new_tokens: int | None = None,
        reference_outputs: list[str] | None = None,
        baseline_tok_s: float | None = None,
    ) -> dict[str, Any]:
        spec = self._resolve_spec(task_key)
        if not self._runtime_matches_task_spec(task_key):
            payload = self._not_ready_payload(task_key)
            payload["notes"] = list(payload["notes"]) + [
                "The current B200 runtime does not match the requested Task 4 model configuration. "
                "Align the Task 4 env overrides and redeploy before running live benchmarks."
            ]
            return payload

        max_new_tokens = max_new_tokens or spec.default_max_new_tokens
        parsed = SpeculateConfig.model_validate(config)
        if reference_outputs is not None and baseline_tok_s is not None:
            baseline = {
                "throughput_tok_s": float(baseline_tok_s),
                "reference_outputs": reference_outputs,
            }
        else:
            baseline = self._ensure_baseline_cache(task_key, prompt_limit, max_new_tokens)
        prompts = _load_prompts(prompt_limit, prompt_file=spec.prompt_file)
        benchmark = self._benchmark_prompts(prompts, parsed, max_new_tokens=max_new_tokens)

        quality = quality_match_rate(
            baseline["reference_outputs"],
            benchmark["outputs"],
        )
        quality_scores = score_pairs(
            baseline["reference_outputs"],
            benchmark["outputs"],
        )
        speedup = (
            benchmark["throughput_tok_s"] / baseline["throughput_tok_s"]
            if baseline["throughput_tok_s"]
            else 0.0
        )
        unsupported = self._unsupported_config_dimensions(parsed, task_key=task_key)
        exact_match_count = sum(
            1 for item in quality_scores if bool(item["exact_match"])
        )
        quality_breakdown = [
            {
                "prompt_index": index,
                "token_overlap": round(float(score["token_overlap"]), 4),
                "exact_match": bool(score["exact_match"]),
                "reference_preview": _preview_text(reference),
                "candidate_preview": _preview_text(candidate),
            }
            for index, (score, reference, candidate) in enumerate(
                zip(
                    quality_scores,
                    baseline["reference_outputs"],
                    benchmark["outputs"],
                    strict=False,
                )
            )
        ]

        notes = list(spec.notes)
        if parsed.ngram_cache_size > 0:
            notes.append(
                "On the current B200 validation path, ngram_cache_size maps to Transformers "
                "prompt-lookup assisted decoding rather than a literal KV n-gram cache."
            )
        if unsupported:
            notes.append(
                "Unsupported config dimensions were ignored for the B200 validation path: "
                + ", ".join(unsupported)
            )
        notes.append(
            "Task 4 currently validates the B200 backend on real hardware, but explicit wide-tree frontier execution remains staged for a later serving stack."
        )

        return {
            "task": spec.slug,
            "gpu": spec.gpu_name,
            "ready": True,
            "target_model": spec.target_model,
            "draft_model": spec.draft_model,
            "config": parsed.model_dump(),
            "throughput_tok_s": benchmark["throughput_tok_s"],
            "baseline_tok_s": baseline["throughput_tok_s"],
            "speedup": speedup,
            "quality_match_rate": quality,
            "exact_match_rate": (
                exact_match_count / len(quality_scores) if quality_scores else 0.0
            ),
            "exact_match_count": exact_match_count,
            "quality_breakdown": quality_breakdown,
            "acceptance_rate": None,
            "acceptance_rate_note": (
                "Transformers assisted generation does not expose accepted-token "
                "counts directly, so acceptance_rate is unavailable in the current B200 path."
            ),
            "average_output_tokens": (
                benchmark["generated_tokens_total"] / prompt_limit if prompt_limit else 0.0
            ),
            "elapsed_sec_total": benchmark["elapsed_sec_total"],
            "generated_tokens_total": benchmark["generated_tokens_total"],
            "attention_backend": spec.attn_implementation,
            "prompt_file": spec.prompt_file,
            "rollout_status": spec.rollout_status,
            "generation_strategy": (
                benchmark["runs"][0]["generation_strategy"]
                if benchmark["runs"]
                else "assistant_model"
            ),
            "applied_generation": benchmark["runs"][0]["applied_generation"] if benchmark["runs"] else {},
            "unsupported_config_dimensions": unsupported,
            "notes": notes,
        }


@app.local_entrypoint()
def main(
    mode: str = "baseline",
    task: str = "task1",
    prompt_limit: int = 5,
    max_new_tokens: int = 0,
    num_speculative_tokens: int = 4,
    acceptance_threshold: float = 0.5,
    draft_temperature: float = 0.0,
    tree_depth: int = 1,
    tree_branching: int = 1,
    ngram_cache_size: int = 0,
    adaptive_depth: bool = False,
    label: str = "manual_probe",
) -> None:
    task = task.lower()
    if task in {"a100", "task1"}:
        worker = SpeculationA100()
        task_key = "task1"
    elif task in {"h100", "task2", "task3"}:
        worker = SpeculationH100()
        task_key = "task2" if task == "h100" else task
    elif task in {"b200", "task4"}:
        worker = SpeculationB200()
        task_key = "task4"
    else:
        raise ValueError(
            f"Unsupported task '{task}'. Use task1, task2, task3, task4, a100, h100, or b200."
        )

    if mode == "baseline":
        result = worker.baseline.remote(
            task_key=task_key,
            prompt_limit=prompt_limit,
            max_new_tokens=max_new_tokens or None,
        )
    elif mode == "benchmark":
        result = worker.benchmark_config.remote(
            {
                "num_speculative_tokens": num_speculative_tokens,
                "acceptance_threshold": acceptance_threshold,
                "draft_temperature": draft_temperature,
                "tree_depth": tree_depth,
                "tree_branching": tree_branching,
                "ngram_cache_size": ngram_cache_size,
                "adaptive_depth": adaptive_depth,
                "label": label,
            },
            task_key=task_key,
            prompt_limit=prompt_limit,
            max_new_tokens=max_new_tokens or None,
        )
    else:
        raise ValueError(f"Unsupported mode '{mode}'. Use baseline or benchmark.")

    print(json.dumps(result, indent=2))
