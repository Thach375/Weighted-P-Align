# Architecture: Reward-Weighted P-ALIGN MVP

Status: draft for review before implementation.

## Stack

- Language: Python 3.10, matching the existing environment guidance.
- Model inference: Hugging Face `transformers` and `vllm`.
- Training integration: external LLaMA-Factory workflow, not reimplemented in MVP.
- Verification: `math_verify` as the required local verifier, with optional OAT-style fallback only if the dependency is available.
- Data format: JSONL for all pipeline artifacts.
- Utilities: `jsonlines`, `tqdm`, standard-library `json`, `argparse`, `dataclasses`, and `pathlib`.
- Tests: standard-library `unittest` for MVP to avoid adding dependencies. A later phase may add `pytest` with approval.

## Existing Repository Shape

```text
data/
  raw/                 # Dataset acquisition notes
  result/              # Existing inference and evaluation results
docs/
  PROJECT_DESCRIPTION.md
figs/
  pipeline-final.png
scripts/
  Prefix_truncation.sh
  Prefix_alignment.sh
  Inference.sh
  evaluation.sh
  train.sh
src/
  binary_search.py
  prefix-alignment.py
  evaluation.py
  test.py
task/
  todo.md
```

## Proposed MVP Additions

Keep the original baseline files stable and add the reward-weighted pipeline beside them.

```text
src/
  generate_continuations.py      # CLI: K samples per prefix
  score_rewards.py               # CLI: verifier + reward records
  build_weighted_sft.py          # CLI: L1/L2 SFT export
  build_dpo_pairs.py             # CLI: optional DPO pair export
  report_run.py                  # CLI: metrics summary
  rw_palign/
    __init__.py
    schemas.py                   # Typed record/dataclass definitions
    io.py                        # JSONL load/write/resume helpers
    prompts.py                   # Prompt construction
    verifier.py                  # Answer extraction and verification
    weighting.py                 # Group stats and SFT weights
    dpo.py                       # Pair selection
    prefix_feedback.py           # Prefix action decisions
tests/
  fixtures/
    prefix_records.jsonl
    continuations.jsonl
  test_verifier.py
  test_weighting.py
  test_dpo.py
  test_io_contracts.py
runs/
  .gitkeep                       # Optional local run parent; outputs ignored if added
```

## Runtime Artifact Layout

Each experiment should write to a dedicated run directory.

```text
runs/<run_name>/
  config.resolved.json
  continuations.jsonl
  rewarded_samples.jsonl
  group_stats.jsonl
  weighted_sft.jsonl
  dpo_pairs.jsonl
  metrics.json
  report.md
  logs/
```

Assumption: `runs/` artifacts are local experiment outputs and should not be committed unless explicitly selected as small fixtures.

## Component Responsibilities

### CLI Scripts

CLI scripts parse arguments, validate file paths, call library functions, and write artifacts. They should stay thin so tests can target pure Python modules.

### `rw_palign.schemas`

Defines stable record shapes for:

- `PrefixRecord`
- `ContinuationSample`
- `RewardedSample`
- `GroupStats`
- `WeightedSFTExample`
- `DPOPair`

The schemas should tolerate extra fields from existing P-ALIGN JSONL records.

### `rw_palign.io`

Handles JSONL streaming, malformed-line reporting, output directory creation, atomic-ish writes where practical, and resume checks by stable IDs.

### `rw_palign.prompts`

Builds the prompt used for continuation sampling. It should preserve the existing P-ALIGN prompt intent:

```text
Please continue from the draft and solve the problem step by step, and put your final answer within \boxed{}.
Question: ...
Prefix: ...
```

### `rw_palign.verifier`

Extracts final answers, runs symbolic or exact verification, applies optional length shaping, and records verifier errors without killing the whole run.

### `rw_palign.weighting`

Computes group stats, advantages, L1/L2 weights, and group statuses. This module must be fully testable without model or GPU access.

### `rw_palign.prefix_feedback`

Decides whether to keep, extend, or mark a prefix for shortening review based on group pass rate. The MVP records actions and supports at most one feedback retry unless configured otherwise.

### `rw_palign.dpo`

Selects one best-vs-worst pair per mixed group by default and skips groups without a clear reward gap.

## Design Decisions

### D1. Use JSONL Artifacts Instead of a Database

Research runs are file-oriented, batchable, and easy to inspect. JSONL also matches the current repository style and Hugging Face style data workflows. A database would add operational cost without improving the MVP.

### D2. Preserve Baseline P-ALIGN Scripts

The existing implementation is the baseline. The reward-weighted extension should be additive so baseline results remain reproducible and differences are attributable to the new data-selection step.

### D3. Separate Model Generation From Scoring and Weighting

Generation is GPU-heavy and slow; scoring and weighting are cheaper and need fast iteration. Persisting intermediate continuations allows verifier and weighting changes without regenerating samples.

### D4. Keep Core Logic Model-Free

Verifier, weighting, DPO selection, IO validation, and prefix-action logic should run in unit tests without loading a model. This reduces feedback time and isolates math/data bugs from GPU infrastructure.

### D5. Make DPO Export Optional

Weighted SFT is the primary MVP path. DPO adds value but also introduces trainer-specific format choices and overoptimization risk, so it starts as a clean pair-export artifact.

### D6. Normalize SFT Weights Within Each Group

Per-group normalization stabilizes loss scale across different `K` values and keeps hard-question emphasis local to each question.

### D7. Bound Prefix Feedback

Prefix feedback addresses false sufficiency judgments, but unbounded retries can hide data problems and waste GPU time. The MVP records decisions and permits a small configured retry count.

### D8. Prefer Standard Library Tests Initially

The current requirements do not list a test framework. `unittest` is enough for the deterministic core logic and avoids introducing dependency churn during MVP planning.

## Dependency Notes

Required from existing environment:

- `torch`
- `transformers`
- `vllm`
- `jsonlines`
- `tqdm`
- `math_verify`
- `sympy`
- `latex2sympy2-extended`

External or optional:

- LLaMA-Factory for training.
- `oat_math_grader` if the OAT fallback remains enabled.
- DPO trainer such as LLaMA-Factory DPO support or TRL, to be selected later.

No new dependency should be added for the MVP without documenting the reason and updating `requirements.txt`.

## Commands

Planning-level command targets:

```bash
python -m unittest discover tests
python src/generate_continuations.py --help
python src/score_rewards.py --help
python src/build_weighted_sft.py --help
python src/build_dpo_pairs.py --help
python src/report_run.py --help
```

GPU smoke command shape:

```bash
python src/generate_continuations.py \
  --input tests/fixtures/prefix_records.jsonl \
  --output runs/smoke/continuations.jsonl \
  --model /path/to/student \
  --k 2 \
  --limit 2
```

## Risks and Mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Verifier rejects correct equivalent answers | False negatives reduce useful data | Keep parsed answer and verifier error fields; add fixture cases for numeric, LaTeX, and boxed answers |
| L2 overweights lucky correct samples | Noisy training signal | Clip advantage and report high-weight samples |
| Sampling collapse produces duplicates | Little contrast despite high K | Report duplicate rate and distinct final-answer count |
| Trainer cannot consume weights | Weighted SFT output unusable | Provide documented fallback export or oversampling phase after trainer choice |
| GPU generation interrupted | Lost work | Resume by `(record_id, sample_index)` |
| Existing scripts have placeholder paths | Repro friction | New CLI scripts use explicit arguments and `--help` |

## Boundaries

- Always: validate JSONL fields, record configs, write resumable artifacts, and test core math/data logic.
- Ask first: adding dependencies, changing baseline scripts, changing training framework, committing generated run outputs.
- Never: commit secrets or model paths, silently drop malformed records without a report, run unbounded feedback loops, or overwrite run outputs unless `--overwrite` is explicit.
