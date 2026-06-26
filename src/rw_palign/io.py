from __future__ import annotations

import json
from dataclasses import dataclass, field
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Callable, Iterable, TypeVar

from .schemas import ContinuationSample, PrefixRecord, SchemaValidationError


T = TypeVar("T")


class JsonlReadError(ValueError):
    """Raised when JSONL input cannot be read in strict mode."""


@dataclass(frozen=True)
class JsonlIssue:
    line_number: int
    code: str
    message: str
    raw_line: str


@dataclass
class JsonlLoadResult:
    records: list[Any] = field(default_factory=list)
    errors: list[JsonlIssue] = field(default_factory=list)


def read_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    input_path = Path(path)
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw_line = line.rstrip("\n")
            if not raw_line.strip():
                continue
            try:
                value = json.loads(raw_line)
            except JSONDecodeError as exc:
                raise JsonlReadError(f"{input_path}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(value, dict):
                raise JsonlReadError(f"{input_path}:{line_number}: expected JSON object")
            yield value


def write_jsonl(path: str | Path, rows: Iterable[Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            if hasattr(row, "to_dict"):
                row = row.to_dict()
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_prefix_records(path: str | Path, strict: bool = False) -> JsonlLoadResult:
    return _load_records(
        path=Path(path),
        factory=PrefixRecord.from_dict,
        strict=strict,
        missing_id_factory=lambda input_path, line_number: f"{input_path.stem}:{line_number}",
    )


def load_continuation_samples(path: str | Path, strict: bool = False) -> JsonlLoadResult:
    return _load_records(
        path=Path(path),
        factory=ContinuationSample.from_dict,
        strict=strict,
        missing_id_factory=None,
    )


def _load_records(
    path: Path,
    factory: Callable[[dict[str, Any]], T],
    strict: bool,
    missing_id_factory: Callable[[Path, int], str] | None,
) -> JsonlLoadResult:
    result: JsonlLoadResult = JsonlLoadResult()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw_line = line.rstrip("\n")
            if not raw_line.strip():
                continue
            try:
                data = json.loads(raw_line)
                if not isinstance(data, dict):
                    raise SchemaValidationError("Expected JSON object")
                if missing_id_factory is not None and "id" not in data:
                    data = dict(data)
                    data["id"] = missing_id_factory(path, line_number)
                result.records.append(factory(data))
            except JSONDecodeError as exc:
                issue = JsonlIssue(
                    line_number=line_number,
                    code="json_decode_error",
                    message=exc.msg,
                    raw_line=raw_line,
                )
                _handle_issue(path, issue, strict)
                result.errors.append(issue)
            except (SchemaValidationError, TypeError, ValueError) as exc:
                issue = JsonlIssue(
                    line_number=line_number,
                    code="validation_error",
                    message=str(exc),
                    raw_line=raw_line,
                )
                _handle_issue(path, issue, strict)
                result.errors.append(issue)
    return result


def _handle_issue(path: Path, issue: JsonlIssue, strict: bool) -> None:
    if strict:
        raise JsonlReadError(f"{path}:{issue.line_number}: {issue.code}: {issue.message}")
