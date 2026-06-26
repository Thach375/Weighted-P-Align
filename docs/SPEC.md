# Specification: Reward-Weighted P-ALIGN MVP

Source: `docs/PROJECT_DESCRIPTION.md`

Status: draft for review before implementation. This document intentionally defines requirements and contracts only; no product code has been written for this spec.

## Objective

Build the smallest working extension of the existing P-ALIGN offline pipeline that replaces binary keep/drop filtering with reward-weighted data selection.

The MVP takes existing prefix-truncation output, samples `K` student continuations per prefix, verifies each continuation against the ground-truth answer, computes group-relative rewards and weights, then exports:

- weighted SFT JSONL examples for downstream supervised fine-tuning;
- group statistics for analysis and ablations;
- optional DPO preference-pair JSONL for later contrastive refinement.

The primary user is a researcher running math-reasoning distillation experiments on local GPU machines. Success means they can reproduce a baseline-compatible data pipeline and run controlled ablations for `K`, weighting mode, prefix feedback, and optional DPO pair creation.

## Assumptions

- The domain for the MVP is math QA with verifiable final answers, usually inside `\boxed{}`.
- The input training data is JSONL and contains at least `question`, `answer`, and either `sufficient_reasoning` or `Long-CoT`.
- Existing P-ALIGN prefix truncation remains the source of prefixes for the first implementation.
- Teacher traces are text only. No teacher logits, shared tokenizer, online RL, or on-policy rollout is required.
- Training itself is delegated to LLaMA-Factory or another existing trainer. The MVP produces data artifacts and training metadata rather than reimplementing a trainer.
- DPO is optional in the MVP and starts as preference-pair export. Full DPO training integration can be added after the SFT data path is stable.
- The existing repo currently has placeholder paths and some filename mismatches in shell scripts. Those are treated as implementation tasks, not solved in this planning pass.
- `oat_math_grader` is imported by existing `src/evaluation.py` but is not visible in `requirements.txt`; the MVP should either document it as an external optional dependency or keep the verifier functional without it.

## Non-Goals

- No web UI or service backend.
- No database. JSONL artifacts are the durable interface.
- No online RL, GRPO training loop, or policy-gradient implementation.
- No teacher-logit steering.
- No broad refactor of the original P-ALIGN scripts unless needed for compatibility.
- No dependency changes without explicit review.

## User Flows

### Flow 1: Build Weighted SFT Data

1. User prepares raw or prefix-truncated JSONL.
2. User runs the existing prefix truncation step if only `Long-CoT` is available.
3. User samples `K` continuations for each `(question, prefix)` group.
4. User scores every continuation with the verifier.
5. User computes group stats, advantage, and normalized SFT weights.
6. User exports weighted SFT JSONL.
7. User passes the weighted dataset to the training stack.

### Flow 2: Inspect Prefix Quality

1. User runs K-continuation generation and scoring.
2. User reads group statistics such as `pass_rate`, `status`, and `prefix_action`.
3. User identifies all-wrong or low-pass-rate groups as candidates for prefix extension.
4. User reruns only flagged groups with adjusted prefixes, within a bounded retry limit.

### Flow 3: Export DPO Pairs

1. User enables DPO export after scoring.
2. For each mixed group, the pipeline selects a chosen continuation and rejected continuation.
3. The pipeline writes preference pairs with rewards and metadata.
4. User trains DPO later with an external trainer.

### Flow 4: Run Ablations

1. User selects one experiment configuration: baseline, L1, L2, prefix feedback, or DPO export.
2. Pipeline writes config, data artifacts, and metrics to a run directory.
3. User compares accuracy, pass rate, sample coverage, length, and cost across runs.

## Functional Requirements

### F1. Prefix Input Compatibility

- Accept prefix-truncation JSONL records containing `question`, `answer`, and `sufficient_reasoning`.
- Accept raw trace JSONL containing `question`, `answer`, and `Long-CoT` only if the prefix-truncation step is explicitly requested.
- Preserve source IDs when present; otherwise generate stable deterministic IDs from row order and input path.
- Reject or quarantine records missing `question` or `answer`.

### F2. K-Continuation Generation

- Generate `K >= 1` continuations per `(question, prefix)` group.
- Use configurable sampling settings: `temperature`, `top_p`, `max_tokens`, `seed`, `batch_size`, and `model`.
- Default MVP values: `K=4`, `temperature=0.8`, `top_p=0.95`, `max_tokens=32768` for training data generation, subject to GPU memory.
- Store each continuation with prompt, sample index, generation config, and raw model output.
- Support resume behavior that skips already completed `(record_id, sample_index)` outputs.

### F3. Outcome Reward Verification

- Extract final answers from continuations, prioritizing `\boxed{...}`.
- Score each continuation against the ground-truth answer with `math_verify` or a compatible local verifier.
- Return a binary correctness label for MVP.
- Optionally apply a small length penalty after correctness scoring.
- Persist verifier name, parsed answer, reward, correctness, and errors.

### F4. Group-Relative Advantage

- For each group of `K` samples, compute:
  - reward mean;
  - reward standard deviation;
  - pass rate;
  - advantage `(reward - mean) / (std + epsilon)`;
  - group status: `mixed`, `all_correct`, `all_wrong`, or `singleton`.
- Use `epsilon=1e-6` by default.
- Never divide by zero or emit NaN/Infinity.

### F5. SFT Weighting

- Support L1 filtering: keep samples with `reward >= group_mean`; assign equal normalized weights among kept samples.
- Support L2 weighting: `weight = clip(advantage, 0, c)`, default `c=3.0`; normalize weights within each group.
- For `all_correct` with zero reward variance, assign equal normalized weights unless length shaping creates a rank.
- For `all_wrong`, skip SFT examples and flag the group for review or prefix feedback.
- For `K=1`, preserve baseline behavior: keep the sample only if it verifies correct.

### F6. Prefix Feedback

- Compute group pass rate `p`.
- If `p < tau_low`, flag the prefix for extension.
- If `p == 1` and the configured mode allows compression, flag the prefix for optional shortening review.
- Bound feedback attempts by `max_feedback_rounds`, default `1` for MVP.
- Record the prefix action and reason in group stats.
- Do not run unbounded feedback loops.

### F7. Weighted SFT Export

- Export JSONL records containing prompt/question, prefix, continuation, combined supervision text, normalized weight, raw reward, and metadata.
- Include enough metadata to trace each example back to input record, group, sample index, and config.
- Keep output compatible with a downstream trainer that can consume weighted examples. If the selected trainer cannot consume weights directly, export a documented fallback format for oversampling or postprocessing.

### F8. DPO Pair Export

- Export pairs only from `mixed` groups unless a later config explicitly enables correct-vs-correct length preferences.
- Select chosen as the highest-reward sample and rejected as the lowest-reward sample.
- Require a minimum reward gap, default `> 0`.
- Write prompt/question, prefix, chosen continuation, rejected continuation, rewards, parsed answers, and group metadata.
- Skip all-correct and all-wrong groups by default.

### F9. Metrics and Reports

- Emit a run summary with:
  - input records;
  - generated samples;
  - verified samples;
  - skipped records;
  - all-correct, all-wrong, mixed, singleton counts;
  - average pass rate;
  - average selected weight count per group;
  - average continuation length;
  - DPO pair count when enabled.
- Write metrics as JSON for scripts and Markdown or plain text for human inspection.

### F10. Experiment Configuration and Reproducibility

- Use a run config file or explicit CLI flags for all non-default behavior.
- Copy the resolved config into the output run directory.
- Record model path/name, seed, sampling parameters, verifier mode, weighting mode, and git commit when available.
- Fail fast on invalid configs.

## Data Model

All durable artifacts are JSONL unless noted.

### RawTrace

Input record before prefix truncation.

```json
{
  "id": "optional stable id",
  "question": "problem text",
  "answer": "ground truth final answer",
  "Long-CoT": "teacher reasoning trace"
}
```

### PrefixRecord

Record after adaptive prefix truncation.

```json
{
  "id": "record id",
  "question": "problem text",
  "answer": "ground truth final answer",
  "sufficient_reasoning": "minimal sufficient prefix",
  "sufficient_sentences": 5,
  "total_sentences": 18,
  "prefix_ratio": 0.2778,
  "is_sufficient": true,
  "evaluator_response": "[ENOUGH]"
}
```

### ContinuationSample

One generated sample within a group.

```json
{
  "group_id": "record id plus prefix version",
  "record_id": "record id",
  "sample_index": 0,
  "question": "problem text",
  "answer": "ground truth final answer",
  "prefix": "sufficient reasoning prefix",
  "prompt": "full model prompt",
  "continuation": "student continuation",
  "generation_config": {
    "model": "path or name",
    "temperature": 0.8,
    "top_p": 0.95,
    "max_tokens": 32768,
    "seed": 42
  }
}
```

### RewardedSample

Continuation sample after verification.

```json
{
  "group_id": "record id plus prefix version",
  "record_id": "record id",
  "sample_index": 0,
  "correct": true,
  "reward": 1.0,
  "parsed_answer": "204",
  "length_tokens": 512,
  "length_penalty": 0.0,
  "verifier": "math_verify",
  "verifier_error": null
}
```

### GroupStats

Aggregate stats for a single `(question, prefix)` group.

```json
{
  "group_id": "record id plus prefix version",
  "record_id": "record id",
  "k": 4,
  "pass_rate": 0.25,
  "reward_mean": 0.25,
  "reward_std": 0.4330127,
  "status": "mixed",
  "prefix_action": "keep",
  "prefix_action_reason": "pass_rate within configured bounds"
}
```

### WeightedSFTExample

Exported training example.

```json
{
  "id": "record id/sample index",
  "prompt": "question plus prefix prompt",
  "question": "problem text",
  "prefix": "sufficient reasoning prefix",
  "continuation": "student continuation",
  "text": "prefix plus continuation",
  "weight": 1.0,
  "normalized_weight": 1.0,
  "reward": 1.0,
  "advantage": 1.732,
  "metadata": {
    "group_id": "record id plus prefix version",
    "sample_index": 0,
    "weighting_mode": "L2"
  }
}
```

### DPOPair

Optional preference pair.

```json
{
  "id": "group id/pair index",
  "prompt": "question plus prefix prompt",
  "question": "problem text",
  "prefix": "sufficient reasoning prefix",
  "chosen": "best continuation",
  "rejected": "worst continuation",
  "chosen_reward": 1.0,
  "rejected_reward": 0.0,
  "reward_gap": 1.0,
  "metadata": {
    "group_id": "record id plus prefix version",
    "chosen_sample_index": 1,
    "rejected_sample_index": 3
  }
}
```

## CLI API Contracts

These are proposed command contracts for implementation. Exact filenames may change during implementation, but the data contracts should remain stable.

### Generate Continuations

```bash
python src/generate_continuations.py \
  --input data/prefix/train.jsonl \
  --output runs/example/continuations.jsonl \
  --model /path/to/student \
  --k 4 \
  --temperature 0.8 \
  --top_p 0.95 \
  --max_tokens 32768 \
  --batch_size 128 \
  --seed 42 \
  --resume
```

### Score Rewards

```bash
python src/score_rewards.py \
  --input runs/example/continuations.jsonl \
  --output runs/example/rewarded_samples.jsonl \
  --group_stats runs/example/group_stats.jsonl \
  --verifier math_verify \
  --length_penalty 0.0
```

### Build Weighted SFT Data

```bash
python src/build_weighted_sft.py \
  --samples runs/example/rewarded_samples.jsonl \
  --groups runs/example/group_stats.jsonl \
  --output runs/example/weighted_sft.jsonl \
  --mode L2 \
  --clip 3.0 \
  --epsilon 1e-6
```

### Build DPO Pairs

```bash
python src/build_dpo_pairs.py \
  --samples runs/example/rewarded_samples.jsonl \
  --groups runs/example/group_stats.jsonl \
  --output runs/example/dpo_pairs.jsonl \
  --min_reward_gap 1.0
```

### Write Run Report

```bash
python src/report_run.py \
  --run_dir runs/example \
  --output runs/example/report.md
```

## Python Module API Contracts

The implementation should keep core logic testable without loading a model.

```python
def extract_answer(text: str) -> str | None:
    """Return the final answer candidate, preferring boxed answers."""

def verify_answer(prediction: str, golden: str) -> tuple[bool, str | None, str | None]:
    """Return correctness, parsed answer, and optional error."""

def compute_group_stats(rewards: list[float], epsilon: float) -> dict:
    """Return mean, std, pass rate, advantages, and status."""

def compute_sft_weights(
    rewards: list[float],
    advantages: list[float],
    mode: str,
    clip: float,
) -> list[float]:
    """Return normalized per-group weights."""

def select_dpo_pair(samples: list[dict], min_reward_gap: float) -> dict | None:
    """Return one chosen/rejected pair or None."""

def decide_prefix_action(pass_rate: float, tau_low: float, allow_shorten: bool) -> str:
    """Return keep, extend, shorten_review, or skip."""
```

## Edge Cases

- `K=1`: behaves like baseline binary filtering.
- All samples wrong: skip SFT and flag for feedback.
- All samples correct with identical reward: use equal weights and no DPO pair.
- All samples correct with length penalty: allow ranked SFT weights but no negative reward-only examples unless configured.
- Reward standard deviation is zero: advantages become zero and no NaN is emitted.
- Duplicate continuations: keep for raw generation accounting, but optionally deduplicate before SFT export if configured.
- Missing `\boxed{}`: verifier attempts fallback parsing; if parsing fails, correctness is false and error is recorded.
- Multiple boxed answers: use the final boxed answer by default and record extraction mode.
- Numeric answer type mismatch, such as `142` vs `142.0`: verifier normalizes where possible.
- Malformed JSONL line: quarantine line with error and continue unless `--strict` is set.
- Generation crash mid-run: resume skips completed sample indexes and continues incomplete groups.
- Very long continuations: truncate only at generation limit; record length and do not silently clip output files.
- Prefix judged sufficient but group pass rate low: flag for prefix extension.
- Prefix extension still all wrong after maximum feedback rounds: skip from SFT and report.
- No mixed groups: DPO export writes an empty file plus a report explaining why.
- Verifier timeout: mark sample incorrect, record timeout, and continue.
- GPU unavailable: generation command fails with an actionable error; non-generation tests still run.

## Success Criteria

- The pipeline can process a small fixture dataset end to end without loading a real model by using generated fixture continuations.
- Weighted SFT output contains only eligible examples and all normalized weights sum to `1.0` per group within tolerance.
- DPO pair output contains pairs only from valid mixed groups.
- Metrics report identifies all group statuses and skipped records.
- All core reward, weighting, pair-selection, and JSONL validation logic is covered by automated tests.
- Existing baseline P-ALIGN scripts remain untouched unless a later implementation task explicitly changes them.

## Open Questions

- Which downstream trainer will consume `normalized_weight` directly? If none, should MVP implement oversampling as a fallback export?
- Should length penalty use token count, character count, or generated-token count from the model output?
- What exact threshold should define "prefix too easy" for optional shortening: `p == 1.0`, `p >= 0.95`, or a dataset-specific rule?
- Should DPO pairs include the shared prefix inside `chosen/rejected`, or should prefix stay only in the prompt field for the selected trainer?
- Which datasets are in scope for first verification: s1K-1.1 only, or also AIME24, AIME25, AMC12, and MATH500?
