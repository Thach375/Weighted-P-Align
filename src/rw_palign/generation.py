from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Iterable

from .io import load_continuation_samples
from .prompts import build_continuation_prompt
from .schemas import ContinuationSample, PrefixRecord


Generator = Callable[[PrefixRecord, str, int, dict[str, object]], str]


def build_generation_config(
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    batch_size: int,
    seed: int,
    backend: str,
) -> dict[str, object]:
    if not model:
        raise ValueError("model is required")
    if temperature < 0 or not math.isfinite(temperature):
        raise ValueError("temperature must be a non-negative finite value")
    if top_p <= 0 or top_p > 1 or not math.isfinite(top_p):
        raise ValueError("top_p must be in (0, 1]")
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return {
        "model": model,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "batch_size": batch_size,
        "seed": seed,
        "backend": backend,
    }


def generate_continuation_samples(
    records: Iterable[PrefixRecord],
    generator: Generator,
    generation_config: dict[str, object],
    k: int,
    resume_keys: set[tuple[str, int]] | None = None,
    limit: int | None = None,
) -> list[ContinuationSample]:
    if k <= 0:
        raise ValueError("k must be positive")
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")

    samples: list[ContinuationSample] = []
    skipped = resume_keys or set()
    selected_records = list(records)
    if limit is not None:
        selected_records = selected_records[:limit]

    for record in selected_records:
        prompt = build_continuation_prompt(record.question, record.sufficient_reasoning)
        group_id = f"{record.id}:v1"
        for sample_index in range(k):
            if (record.id, sample_index) in skipped:
                continue
            continuation = generator(record, prompt, sample_index, generation_config)
            samples.append(
                ContinuationSample(
                    group_id=group_id,
                    record_id=record.id,
                    sample_index=sample_index,
                    question=record.question,
                    answer=record.answer,
                    prefix=record.sufficient_reasoning,
                    prompt=prompt,
                    continuation=continuation,
                    generation_config=dict(generation_config),
                )
            )
    return samples


def load_resume_keys(path: str | Path) -> set[tuple[str, int]]:
    output_path = Path(path)
    if not output_path.exists():
        return set()
    result = load_continuation_samples(output_path)
    return {(sample.record_id, sample.sample_index) for sample in result.records}


def mock_generator(
    record: PrefixRecord,
    prompt: str,
    sample_index: int,
    generation_config: dict[str, object],
) -> str:
    del prompt, generation_config
    return f"Mock sample {sample_index}. The final answer is \\boxed{{{record.answer}}}."


class VLLMGenerator:
    def __init__(self, model: str):
        try:
            from vllm import LLM
        except ImportError as exc:
            raise RuntimeError("vLLM is not installed; use --backend mock for fixture tests") from exc
        self._llm = LLM(model=model)

    def __call__(
        self,
        record: PrefixRecord,
        prompt: str,
        sample_index: int,
        generation_config: dict[str, object],
    ) -> str:
        del record
        from vllm import SamplingParams

        seed = int(generation_config["seed"]) + sample_index
        params = SamplingParams(
            temperature=float(generation_config["temperature"]),
            top_p=float(generation_config["top_p"]),
            max_tokens=int(generation_config["max_tokens"]),
            seed=seed,
        )
        outputs = self._llm.generate([prompt], params)
        if not outputs or not outputs[0].outputs:
            return ""
        return outputs[0].outputs[0].text
