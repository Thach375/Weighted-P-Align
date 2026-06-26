# Implementation Todo: Reward-Weighted P-ALIGN MVP

Status: draft checklist. Do not begin product-code implementation until the spec and architecture docs are reviewed.

## Phase 0: Planning Approval

- [x] Review `docs/SPEC.md`.
  - Acceptance: assumptions and open questions are either accepted or corrected.
  - Verify: human confirms the MVP scope.
  - Files likely touched: docs only.

- [x] Review `docs/ARCHITECTURE.md`.
  - Acceptance: stack, folder structure, and dependency decisions are accepted.
  - Verify: no unresolved architecture blocker remains.
  - Files likely touched: docs only.

- [x] Review `docs/ACCEPTANCE_CRITERIA.md`.
  - Acceptance: every planned feature has measurable criteria and tests.
  - Verify: human confirms criteria are sufficient for implementation.
  - Files likely touched: docs only.

## Phase 1: Test Fixtures and Data Contracts

- [x] Add minimal JSONL fixtures for prefix records and continuations.
  - Acceptance: fixtures cover mixed, all-correct, all-wrong, singleton, malformed, and missing-field cases.
  - Verify: fixture files can be parsed with a simple JSONL reader.
  - Files likely touched: `tests/fixtures/*`.

- [x] Add schema/dataclass definitions for pipeline records.
  - Acceptance: record types cover PrefixRecord, ContinuationSample, RewardedSample, GroupStats, WeightedSFTExample, and DPOPair.
  - Verify: `python -m unittest discover tests`.
  - Files likely touched: `src/rw_palign/schemas.py`, `tests/test_io_contracts.py`.

- [x] Add JSONL IO helpers with validation and quarantine reporting.
  - Acceptance: valid records load, malformed records are reported, deterministic IDs are generated when needed.
  - Verify: `python -m unittest discover tests`.
  - Files likely touched: `src/rw_palign/io.py`, `tests/test_io_contracts.py`.

## Phase 2: Verifier and Reward Core

- [ ] Implement final-answer extraction.
  - Acceptance: final boxed answers, unboxed fallbacks, and multiple-box cases are handled as specified.
  - Verify: `python -m unittest discover tests`.
  - Files likely touched: `src/rw_palign/verifier.py`, `tests/test_verifier.py`.

- [ ] Implement answer verification wrapper.
  - Acceptance: verifier returns correctness, parsed answer, and per-sample error without crashing the run.
  - Verify: `python -m unittest discover tests`.
  - Files likely touched: `src/rw_palign/verifier.py`, `tests/test_verifier.py`.

- [ ] Implement optional length penalty calculation.
  - Acceptance: reward equals binary correctness minus configured penalty, with penalty disabled by default.
  - Verify: unit tests for zero and nonzero penalty.
  - Files likely touched: `src/rw_palign/verifier.py`, `tests/test_verifier.py`.

## Phase 3: Group Stats and Weighting

- [ ] Implement group status and advantage computation.
  - Acceptance: mixed, all-correct, all-wrong, and singleton groups produce finite stats.
  - Verify: `python -m unittest discover tests`.
  - Files likely touched: `src/rw_palign/weighting.py`, `tests/test_weighting.py`.

- [ ] Implement L1 SFT weighting.
  - Acceptance: samples with `reward >= group_mean` are kept and normalized per group.
  - Verify: L1 fixture tests pass.
  - Files likely touched: `src/rw_palign/weighting.py`, `tests/test_weighting.py`.

- [ ] Implement L2 clipped advantage weighting.
  - Acceptance: positive advantages are clipped, negative advantages become zero, and weights normalize per group.
  - Verify: L2 fixture tests pass.
  - Files likely touched: `src/rw_palign/weighting.py`, `tests/test_weighting.py`.

## Phase 4: Artifact Export

- [ ] Implement weighted SFT export.
  - Acceptance: eligible examples are written with prompt, prefix, continuation, text, reward, advantage, normalized weight, and metadata.
  - Verify: export schema and per-group weight-sum tests pass.
  - Files likely touched: `src/build_weighted_sft.py`, `src/rw_palign/weighting.py`, `src/rw_palign/io.py`, tests.

- [ ] Implement DPO pair selection.
  - Acceptance: one best-vs-worst pair is selected from each valid mixed group by default.
  - Verify: `python -m unittest discover tests`.
  - Files likely touched: `src/rw_palign/dpo.py`, `tests/test_dpo.py`.

- [ ] Implement DPO pair export CLI.
  - Acceptance: pair JSONL follows the DPOPair contract and skips invalid groups with metrics.
  - Verify: fixture export test passes.
  - Files likely touched: `src/build_dpo_pairs.py`, `src/rw_palign/dpo.py`, tests.

## Phase 5: Prefix Feedback and Reporting

- [ ] Implement prefix-feedback decision logic.
  - Acceptance: pass-rate thresholds map to keep, extend, or shorten-review actions with retry caps.
  - Verify: unit tests for threshold and cap cases.
  - Files likely touched: `src/rw_palign/prefix_feedback.py`, tests.

- [ ] Implement metrics aggregation.
  - Acceptance: metrics count records, samples, verified outputs, group statuses, skipped groups, average pass rate, and DPO pairs.
  - Verify: metrics match fixture artifact line counts.
  - Files likely touched: `src/report_run.py`, `src/rw_palign/io.py`, tests.

- [ ] Implement human-readable run report.
  - Acceptance: `report.md` summarizes run config and key metrics.
  - Verify: report fixture contains required metric names.
  - Files likely touched: `src/report_run.py`, tests.

## Phase 6: K-Continuation Generation

- [ ] Implement prompt builder.
  - Acceptance: prompt includes question, prefix, step-by-step continuation instruction, and boxed-answer instruction.
  - Verify: prompt unit test matches expected fixture text.
  - Files likely touched: `src/rw_palign/prompts.py`, tests.

- [ ] Implement mockable generation interface.
  - Acceptance: generation core can be tested with a fake model client.
  - Verify: mock generation test produces `N * K` samples.
  - Files likely touched: `src/generate_continuations.py`, `src/rw_palign/prompts.py`, tests.

- [ ] Implement vLLM-backed continuation CLI.
  - Acceptance: CLI writes ContinuationSample JSONL and supports resume.
  - Verify: `python src/generate_continuations.py --help`; manual GPU smoke when model path is available.
  - Files likely touched: `src/generate_continuations.py`.

## Phase 7: End-to-End Fixture Pipeline

- [ ] Wire fixture scoring to weighted SFT export.
  - Acceptance: fixture mixed group creates weighted SFT examples; all-wrong group is skipped.
  - Verify: `python -m unittest discover tests`.
  - Files likely touched: tests and CLI glue only.

- [ ] Wire fixture DPO export.
  - Acceptance: fixture mixed group creates one DPO pair when enabled.
  - Verify: `python -m unittest discover tests`.
  - Files likely touched: tests and CLI glue only.

- [ ] Run complete local verification.
  - Acceptance: all unit tests pass and CLI help commands succeed.
  - Verify: `python -m unittest discover tests` plus CLI `--help` checks.
  - Files likely touched: none unless failures require fixes.

## Phase 8: Documentation Update

- [ ] Update README or add a short usage doc for the MVP pipeline.
  - Acceptance: commands describe input, output, and expected artifacts.
  - Verify: commands match implemented CLI names and flags.
  - Files likely touched: `README.md` or `docs/USAGE.md`.

- [ ] Document unresolved training integration details.
  - Acceptance: downstream weighted-SFT consumption path is explicit, or a fallback export is documented.
  - Verify: no ambiguity remains for the next implementation phase.
  - Files likely touched: `docs/SPEC.md`, `docs/ARCHITECTURE.md`, or `docs/USAGE.md`.
