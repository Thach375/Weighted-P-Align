# Acceptance Criteria: Reward-Weighted P-ALIGN MVP

Status: draft for review before implementation.

All features below map to `docs/SPEC.md`. Automated tests should use small fixtures and avoid real GPU/model loads unless marked as manual smoke tests.

## Project-Level Definition of Done

- All requested artifacts are produced under documented paths.
- Each CLI supports `--help` and exits with code `0`.
- Unit tests pass with `python -m unittest discover tests`.
- A fixture end-to-end run produces `weighted_sft.jsonl`, `group_stats.jsonl`, `metrics.json`, and optionally `dpo_pairs.jsonl`.
- No existing baseline P-ALIGN behavior is changed without explicit approval.

## F1. Prefix Input Compatibility

Criteria:

- Valid prefix records with `question`, `answer`, and `sufficient_reasoning` are loaded.
- Extra fields are preserved or ignored without failure.
- Missing `question` or `answer` records are reported and skipped unless strict mode is enabled.
- Records without `id` receive deterministic IDs.

Test cases:

- `test_load_valid_prefix_record`: one valid record loads with expected fields.
- `test_missing_answer_is_quarantined`: missing answer increments skipped count.
- `test_extra_fields_do_not_fail`: unknown fields do not break loading.
- `test_generated_id_stable`: same fixture produces same generated IDs across runs.

## F2. K-Continuation Generation

Criteria:

- For `N` input records and `K` samples, output contains `N * K` samples when no generation errors occur.
- Each sample has `group_id`, `record_id`, `sample_index`, `prompt`, `continuation`, and `generation_config`.
- Resume mode does not duplicate completed sample indexes.
- Generation config values in output match resolved CLI/config values.

Test cases:

- `test_mock_generation_outputs_k_samples`: mock generator emits exactly `K` samples per group.
- `test_resume_skips_existing_samples`: existing `(record_id, sample_index)` is not rewritten.
- `test_generation_config_persisted`: output config contains model, temperature, top_p, max_tokens, and seed.
- Manual smoke: run `--limit 2 --k 2` on a real local model and verify four samples are written.

## F3. Outcome Reward Verification

Criteria:

- Final boxed answers are extracted from continuations.
- Equivalent simple numeric answers such as `142` and `142.0` verify as equal when supported by verifier.
- Unparseable outputs receive `correct=false`, an error field, and no crash.
- Verifier timeout or exception is recorded per sample.

Test cases:

- `test_extracts_final_boxed_answer`: text with multiple boxes uses the final answer.
- `test_numeric_equivalence`: `142` verifies against `142.0`.
- `test_unboxed_fallback`: plain final answer is attempted when no boxed answer exists.
- `test_verifier_exception_records_error`: simulated verifier failure does not stop processing.

## F4. Group-Relative Advantage

Criteria:

- Group stats include mean, standard deviation, pass rate, status, and per-sample advantage.
- No NaN or Infinity is emitted for all-correct or all-wrong groups.
- Status classification is correct for `mixed`, `all_correct`, `all_wrong`, and `singleton`.

Test cases:

- `test_mixed_group_stats`: rewards `[1, 0, 0, 0]` produce pass rate `0.25` and status `mixed`.
- `test_all_correct_zero_std`: rewards `[1, 1, 1, 1]` produce finite advantages.
- `test_all_wrong_zero_std`: rewards `[0, 0, 0, 0]` produce status `all_wrong`.
- `test_singleton_status`: one reward produces status `singleton`.

## F5. SFT Weighting

Criteria:

- L1 keeps samples with `reward >= group_mean`.
- L2 applies clipped positive advantage and normalizes kept weights.
- Per-group normalized weights sum to `1.0` within `1e-6` for groups with eligible samples.
- All-wrong groups produce no SFT examples.
- `K=1` keeps only correct samples.

Test cases:

- `test_l1_keeps_above_mean`: rewards `[1, 0, 0, 0]` keep only the correct sample.
- `test_l2_clips_and_normalizes`: large positive advantage is clipped at configured cap.
- `test_equal_all_correct_weights`: four all-correct samples each receive `0.25`.
- `test_all_wrong_skipped`: all-wrong group exports zero SFT examples.
- `test_k1_baseline_behavior`: singleton wrong sample is skipped, singleton correct sample is kept.

## F6. Prefix Feedback

Criteria:

- Groups with pass rate below `tau_low` are marked `extend`.
- Groups with acceptable pass rate are marked `keep`.
- Optional shortening review is only marked when enabled.
- Feedback attempts cannot exceed `max_feedback_rounds`.

Test cases:

- `test_low_pass_rate_extends`: pass rate `0.0` with `tau_low=0.2` returns `extend`.
- `test_mixed_group_kept`: pass rate `0.5` returns `keep`.
- `test_all_correct_shorten_disabled`: pass rate `1.0` returns `keep` when shortening is disabled.
- `test_feedback_round_cap`: group at max rounds is not scheduled again.

## F7. Weighted SFT Export

Criteria:

- Output JSONL contains one line per eligible weighted example.
- Each line includes prompt/question, prefix, continuation, text, reward, advantage, normalized weight, and metadata.
- Output can be traced back to source `record_id`, `group_id`, and `sample_index`.
- Export report counts selected and skipped groups.

Test cases:

- `test_weighted_sft_schema`: exported fixture line has required keys.
- `test_weight_sum_by_group`: weights sum to `1.0` for each selected group.
- `test_trace_metadata_present`: metadata links back to source IDs.
- `test_skipped_group_reported`: all-wrong group appears in metrics as skipped.

## F8. DPO Pair Export

Criteria:

- Mixed groups with reward gap above threshold produce exactly one pair by default.
- Chosen sample has reward greater than rejected sample.
- All-correct and all-wrong groups are skipped by default.
- Empty DPO output is allowed and accompanied by metrics explaining skipped groups.

Test cases:

- `test_mixed_group_pair_selection`: rewards `[1, 0, 0]` select reward `1` as chosen and reward `0` as rejected.
- `test_min_gap_enforced`: group with reward gap below threshold produces no pair.
- `test_all_correct_no_pair`: all-correct group is skipped.
- `test_dpo_pair_schema`: pair contains prompt, prefix, chosen, rejected, rewards, and metadata.

## F9. Metrics and Reports

Criteria:

- `metrics.json` contains input count, generated count, verified count, skipped count, group status counts, average pass rate, average continuation length, and DPO pair count.
- `report.md` summarizes the same values in human-readable form.
- Metrics match artifact line counts.

Test cases:

- `test_metrics_match_counts`: fixture metrics equal JSONL line counts.
- `test_group_status_counts`: mixed/all-correct/all-wrong counts are correct.
- `test_report_contains_key_metrics`: report includes pass rate and DPO pair count.

## F10. Experiment Configuration and Reproducibility

Criteria:

- Resolved config is written to the run directory.
- Invalid config values fail before processing starts.
- Seed, model, sampling config, verifier mode, weighting mode, and git commit when available are recorded.
- Existing run output is not overwritten unless `--overwrite` is explicit.

Test cases:

- `test_resolved_config_written`: config file exists and includes defaults plus overrides.
- `test_invalid_k_rejected`: `K=0` fails validation.
- `test_invalid_weighting_mode_rejected`: unknown mode fails validation.
- `test_no_overwrite_by_default`: command refuses to replace existing output.

## End-to-End Fixture Test

Criteria:

- Given two prefix records and hand-authored continuation samples:
  - one mixed group;
  - one all-wrong group.
- The scoring and export path produces:
  - one or more weighted SFT examples from the mixed group;
  - zero weighted SFT examples from the all-wrong group;
  - one DPO pair from the mixed group when DPO export is enabled;
  - metrics with `mixed=1` and `all_wrong=1`.

Command shape:

```bash
python -m unittest discover tests
```

Manual GPU smoke, after code exists:

```bash
python src/generate_continuations.py \
  --input tests/fixtures/prefix_records.jsonl \
  --output runs/smoke/continuations.jsonl \
  --model /path/to/student \
  --k 2 \
  --limit 2
```
