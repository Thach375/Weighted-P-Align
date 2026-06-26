import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rw_palign.io import (  # noqa: E402
    JsonlReadError,
    load_continuation_samples,
    load_prefix_records,
    read_jsonl,
    write_jsonl,
)
from rw_palign.schemas import (  # noqa: E402
    ContinuationSample,
    DPOPair,
    GroupStats,
    PrefixRecord,
    RewardedSample,
    SchemaValidationError,
    WeightedSFTExample,
)


FIXTURES = ROOT / "tests" / "fixtures"


class FixtureContractsTest(unittest.TestCase):
    def test_valid_fixture_files_are_parseable_jsonl(self):
        for name in ("prefix_records.jsonl", "continuations.jsonl"):
            with self.subTest(name=name):
                rows = list(read_jsonl(FIXTURES / name))
                self.assertGreater(len(rows), 0)
                self.assertTrue(all(isinstance(row, dict) for row in rows))

    def test_invalid_fixture_contains_missing_field_and_malformed_cases(self):
        result = load_prefix_records(FIXTURES / "invalid_records.jsonl")

        self.assertEqual(result.records, [])
        self.assertEqual(len(result.errors), 2)
        self.assertEqual(result.errors[0].line_number, 1)
        self.assertEqual(result.errors[0].code, "validation_error")
        self.assertEqual(result.errors[1].line_number, 2)
        self.assertEqual(result.errors[1].code, "json_decode_error")


class SchemaContractsTest(unittest.TestCase):
    def test_prefix_record_preserves_extra_fields(self):
        record = PrefixRecord.from_dict(
            {
                "id": "rec-extra",
                "question": "What is 1 + 1?",
                "answer": "2",
                "sufficient_reasoning": "One plus one makes two.",
                "source": "fixture",
            }
        )

        self.assertEqual(record.id, "rec-extra")
        self.assertEqual(record.extra, {"source": "fixture"})
        self.assertEqual(record.to_dict()["source"], "fixture")

    def test_prefix_record_requires_core_fields(self):
        with self.assertRaises(SchemaValidationError):
            PrefixRecord.from_dict(
                {
                    "id": "rec-invalid",
                    "question": "Missing an answer",
                    "sufficient_reasoning": "Cannot validate without answer.",
                }
            )

    def test_all_phase_one_record_shapes_round_trip(self):
        records = [
            PrefixRecord(
                id="rec-1",
                question="Q",
                answer="A",
                sufficient_reasoning="prefix",
            ),
            ContinuationSample(
                group_id="rec-1:v1",
                record_id="rec-1",
                sample_index=0,
                question="Q",
                answer="A",
                prefix="prefix",
                prompt="prompt",
                continuation="continuation",
                generation_config={"model": "fixture"},
            ),
            RewardedSample(
                group_id="rec-1:v1",
                record_id="rec-1",
                sample_index=0,
                correct=True,
                reward=1.0,
            ),
            GroupStats(
                group_id="rec-1:v1",
                record_id="rec-1",
                k=1,
                pass_rate=1.0,
                reward_mean=1.0,
                reward_std=0.0,
                status="singleton",
            ),
            WeightedSFTExample(
                id="rec-1/0",
                prompt="prompt",
                question="Q",
                prefix="prefix",
                continuation="continuation",
                text="prefix continuation",
                weight=1.0,
                normalized_weight=1.0,
                reward=1.0,
                advantage=0.0,
            ),
            DPOPair(
                id="rec-1:v1/0",
                prompt="prompt",
                question="Q",
                prefix="prefix",
                chosen="good",
                rejected="bad",
                chosen_reward=1.0,
                rejected_reward=0.0,
                reward_gap=1.0,
            ),
        ]

        for record in records:
            with self.subTest(record=type(record).__name__):
                raw = record.to_dict()
                restored = type(record).from_dict(raw)
                self.assertEqual(restored.to_dict(), raw)


class JsonlIOContractsTest(unittest.TestCase):
    def test_load_prefix_records_generates_stable_ids(self):
        first = load_prefix_records(FIXTURES / "prefix_records.jsonl")
        second = load_prefix_records(FIXTURES / "prefix_records.jsonl")

        self.assertEqual(first.errors, [])
        self.assertEqual(second.errors, [])
        self.assertEqual(len(first.records), 5)
        self.assertEqual([record.id for record in first.records], [record.id for record in second.records])
        self.assertEqual(first.records[-1].id, "prefix_records:5")

    def test_load_continuation_samples_preserves_fixture_metadata(self):
        result = load_continuation_samples(FIXTURES / "continuations.jsonl")

        self.assertEqual(result.errors, [])
        self.assertEqual(len(result.records), 7)
        self.assertEqual(result.records[0].extra["expected_group_status"], "mixed")
        self.assertIs(result.records[0].extra["expected_correct"], True)

    def test_strict_mode_raises_on_invalid_jsonl(self):
        with self.assertRaises(JsonlReadError):
            load_prefix_records(FIXTURES / "invalid_records.jsonl", strict=True)

    def test_schema_conversion_error_is_quarantined(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "bad_continuation.jsonl"
            input_path.write_text(
                json.dumps(
                    {
                        "group_id": "rec-1:v1",
                        "record_id": "rec-1",
                        "sample_index": "not-an-int",
                        "question": "Q",
                        "answer": "A",
                        "prefix": "prefix",
                        "prompt": "prompt",
                        "continuation": "continuation",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            result = load_continuation_samples(input_path)

            self.assertEqual(result.records, [])
            self.assertEqual(len(result.errors), 1)
            self.assertEqual(result.errors[0].code, "validation_error")

    def test_write_jsonl_creates_parent_and_round_trips_dicts(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "nested" / "records.jsonl"
            rows = [{"id": "one", "value": 1}, {"id": "two", "value": 2}]

            write_jsonl(output, rows)

            self.assertEqual(list(read_jsonl(output)), rows)
            with output.open("r", encoding="utf-8") as handle:
                raw_lines = [json.loads(line) for line in handle if line.strip()]
            self.assertEqual(raw_lines, rows)


if __name__ == "__main__":
    unittest.main()
