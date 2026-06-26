# Reward-Weighted P-ALIGN MVP Usage

This MVP adds a file-based reward-weighted data pipeline beside the original
P-ALIGN scripts. It does not change the baseline prefix truncation, alignment,
inference, or evaluation scripts.

## Artifacts

A run directory can contain:

```text
runs/<name>/
  config.resolved.json
  continuations.jsonl
  rewarded_samples.jsonl
  group_stats.jsonl
  weighted_sft.jsonl
  dpo_pairs.jsonl
  metrics.json
  report.md
```

All CLIs refuse to replace existing output files unless `--overwrite` is
passed. Generation can also use `--resume` to keep existing samples and add only
missing `(record_id, sample_index)` rows.

## 1. Generate K Continuations

Use prefix-truncation output containing `question`, `answer`, and
`sufficient_reasoning`.

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

For a no-GPU fixture smoke test:

```bash
python src/generate_continuations.py \
  --input tests/fixtures/prefix_records.jsonl \
  --output runs/fixture/continuations.jsonl \
  --model fixture-model \
  --backend mock \
  --k 2 \
  --limit 2
```

## 2. Score Rewards

```bash
python src/score_rewards.py \
  --input runs/example/continuations.jsonl \
  --output runs/example/rewarded_samples.jsonl \
  --group_stats runs/example/group_stats.jsonl \
  --verifier math_verify \
  --length_penalty 0.0
```

Use `--verifier local` for exact/numeric fixture tests that should not depend on
`math_verify`.

## 3. Export Weighted SFT Data

```bash
python src/build_weighted_sft.py \
  --samples runs/example/rewarded_samples.jsonl \
  --groups runs/example/group_stats.jsonl \
  --output runs/example/weighted_sft.jsonl \
  --mode L2 \
  --clip 3.0 \
  --epsilon 1e-6
```

The output rows include `prompt`, `question`, `prefix`, `continuation`, `text`,
`reward`, `advantage`, `weight`, `normalized_weight`, and trace metadata.

## 4. Export DPO Pairs

```bash
python src/build_dpo_pairs.py \
  --samples runs/example/rewarded_samples.jsonl \
  --groups runs/example/group_stats.jsonl \
  --output runs/example/dpo_pairs.jsonl \
  --min_reward_gap 0.0
```

Pairs are exported only for mixed groups by default. All-correct and all-wrong
groups are skipped.

## 5. Write Metrics and Report

```bash
python src/report_run.py \
  --run_dir runs/example \
  --output runs/example/report.md
```

This writes `runs/example/metrics.json` unless `--metrics_output` is supplied.

## Fixture Verification

```bash
python -m unittest discover tests
python src/generate_continuations.py --help
python src/score_rewards.py --help
python src/build_weighted_sft.py --help
python src/build_dpo_pairs.py --help
python src/report_run.py --help
```

## Training Integration Status

`weighted_sft.jsonl` is the durable MVP handoff. Trainers that understand a
per-example `normalized_weight` field can consume it directly. The upstream
LLaMA-Factory path is not modified here, and stock trainer behavior should not
be assumed to apply these weights automatically.

If the selected trainer cannot read `normalized_weight`, use this artifact for
an external weighted sampler or an explicit oversampling conversion step before
training. That conversion is intentionally outside this MVP implementation.
