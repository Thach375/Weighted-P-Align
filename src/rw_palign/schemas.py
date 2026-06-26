from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar


class SchemaValidationError(ValueError):
    """Raised when a JSONL record does not satisfy a pipeline schema."""


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value == "")


def _require(data: dict[str, Any], field_name: str) -> Any:
    if field_name not in data or _is_missing(data[field_name]):
        raise SchemaValidationError(f"Missing required field: {field_name}")
    return data[field_name]


def _split_extra(data: dict[str, Any], known_fields: set[str]) -> dict[str, Any]:
    return {key: value for key, value in data.items() if key not in known_fields}


def _base_dict(values: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    result = {key: value for key, value in values.items() if value is not None}
    result.update(extra)
    return result


@dataclass
class PrefixRecord:
    id: str
    question: str
    answer: Any
    sufficient_reasoning: str
    sufficient_sentences: int | None = None
    total_sentences: int | None = None
    prefix_ratio: float | None = None
    is_sufficient: bool | None = None
    evaluator_response: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    _KNOWN_FIELDS: ClassVar[set[str]] = {
        "id",
        "question",
        "answer",
        "sufficient_reasoning",
        "sufficient_sentences",
        "total_sentences",
        "prefix_ratio",
        "is_sufficient",
        "evaluator_response",
    }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PrefixRecord":
        return cls(
            id=str(_require(data, "id")),
            question=str(_require(data, "question")),
            answer=_require(data, "answer"),
            sufficient_reasoning=str(_require(data, "sufficient_reasoning")),
            sufficient_sentences=data.get("sufficient_sentences"),
            total_sentences=data.get("total_sentences"),
            prefix_ratio=data.get("prefix_ratio"),
            is_sufficient=data.get("is_sufficient"),
            evaluator_response=data.get("evaluator_response"),
            extra=_split_extra(data, cls._KNOWN_FIELDS),
        )

    def to_dict(self) -> dict[str, Any]:
        return _base_dict(
            {
                "id": self.id,
                "question": self.question,
                "answer": self.answer,
                "sufficient_reasoning": self.sufficient_reasoning,
                "sufficient_sentences": self.sufficient_sentences,
                "total_sentences": self.total_sentences,
                "prefix_ratio": self.prefix_ratio,
                "is_sufficient": self.is_sufficient,
                "evaluator_response": self.evaluator_response,
            },
            self.extra,
        )


@dataclass
class ContinuationSample:
    group_id: str
    record_id: str
    sample_index: int
    question: str
    answer: Any
    prefix: str
    prompt: str
    continuation: str
    generation_config: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    _KNOWN_FIELDS: ClassVar[set[str]] = {
        "group_id",
        "record_id",
        "sample_index",
        "question",
        "answer",
        "prefix",
        "prompt",
        "continuation",
        "generation_config",
    }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ContinuationSample":
        return cls(
            group_id=str(_require(data, "group_id")),
            record_id=str(_require(data, "record_id")),
            sample_index=int(_require(data, "sample_index")),
            question=str(_require(data, "question")),
            answer=_require(data, "answer"),
            prefix=str(_require(data, "prefix")),
            prompt=str(_require(data, "prompt")),
            continuation=str(_require(data, "continuation")),
            generation_config=dict(data.get("generation_config") or {}),
            extra=_split_extra(data, cls._KNOWN_FIELDS),
        )

    def to_dict(self) -> dict[str, Any]:
        return _base_dict(
            {
                "group_id": self.group_id,
                "record_id": self.record_id,
                "sample_index": self.sample_index,
                "question": self.question,
                "answer": self.answer,
                "prefix": self.prefix,
                "prompt": self.prompt,
                "continuation": self.continuation,
                "generation_config": self.generation_config,
            },
            self.extra,
        )


@dataclass
class RewardedSample:
    group_id: str
    record_id: str
    sample_index: int
    correct: bool
    reward: float
    parsed_answer: str | None = None
    length_tokens: int | None = None
    length_penalty: float | None = None
    verifier: str | None = None
    verifier_error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    _KNOWN_FIELDS: ClassVar[set[str]] = {
        "group_id",
        "record_id",
        "sample_index",
        "correct",
        "reward",
        "parsed_answer",
        "length_tokens",
        "length_penalty",
        "verifier",
        "verifier_error",
    }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RewardedSample":
        return cls(
            group_id=str(_require(data, "group_id")),
            record_id=str(_require(data, "record_id")),
            sample_index=int(_require(data, "sample_index")),
            correct=bool(_require(data, "correct")),
            reward=float(_require(data, "reward")),
            parsed_answer=data.get("parsed_answer"),
            length_tokens=data.get("length_tokens"),
            length_penalty=data.get("length_penalty"),
            verifier=data.get("verifier"),
            verifier_error=data.get("verifier_error"),
            extra=_split_extra(data, cls._KNOWN_FIELDS),
        )

    def to_dict(self) -> dict[str, Any]:
        return _base_dict(
            {
                "group_id": self.group_id,
                "record_id": self.record_id,
                "sample_index": self.sample_index,
                "correct": self.correct,
                "reward": self.reward,
                "parsed_answer": self.parsed_answer,
                "length_tokens": self.length_tokens,
                "length_penalty": self.length_penalty,
                "verifier": self.verifier,
                "verifier_error": self.verifier_error,
            },
            self.extra,
        )


@dataclass
class GroupStats:
    group_id: str
    record_id: str
    k: int
    pass_rate: float
    reward_mean: float
    reward_std: float
    status: str
    prefix_action: str | None = None
    prefix_action_reason: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    _KNOWN_FIELDS: ClassVar[set[str]] = {
        "group_id",
        "record_id",
        "k",
        "pass_rate",
        "reward_mean",
        "reward_std",
        "status",
        "prefix_action",
        "prefix_action_reason",
    }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GroupStats":
        return cls(
            group_id=str(_require(data, "group_id")),
            record_id=str(_require(data, "record_id")),
            k=int(_require(data, "k")),
            pass_rate=float(_require(data, "pass_rate")),
            reward_mean=float(_require(data, "reward_mean")),
            reward_std=float(_require(data, "reward_std")),
            status=str(_require(data, "status")),
            prefix_action=data.get("prefix_action"),
            prefix_action_reason=data.get("prefix_action_reason"),
            extra=_split_extra(data, cls._KNOWN_FIELDS),
        )

    def to_dict(self) -> dict[str, Any]:
        return _base_dict(
            {
                "group_id": self.group_id,
                "record_id": self.record_id,
                "k": self.k,
                "pass_rate": self.pass_rate,
                "reward_mean": self.reward_mean,
                "reward_std": self.reward_std,
                "status": self.status,
                "prefix_action": self.prefix_action,
                "prefix_action_reason": self.prefix_action_reason,
            },
            self.extra,
        )


@dataclass
class WeightedSFTExample:
    id: str
    prompt: str
    question: str
    prefix: str
    continuation: str
    text: str
    weight: float
    normalized_weight: float
    reward: float
    advantage: float
    metadata: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    _KNOWN_FIELDS: ClassVar[set[str]] = {
        "id",
        "prompt",
        "question",
        "prefix",
        "continuation",
        "text",
        "weight",
        "normalized_weight",
        "reward",
        "advantage",
        "metadata",
    }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WeightedSFTExample":
        return cls(
            id=str(_require(data, "id")),
            prompt=str(_require(data, "prompt")),
            question=str(_require(data, "question")),
            prefix=str(_require(data, "prefix")),
            continuation=str(_require(data, "continuation")),
            text=str(_require(data, "text")),
            weight=float(_require(data, "weight")),
            normalized_weight=float(_require(data, "normalized_weight")),
            reward=float(_require(data, "reward")),
            advantage=float(_require(data, "advantage")),
            metadata=dict(data.get("metadata") or {}),
            extra=_split_extra(data, cls._KNOWN_FIELDS),
        )

    def to_dict(self) -> dict[str, Any]:
        return _base_dict(
            {
                "id": self.id,
                "prompt": self.prompt,
                "question": self.question,
                "prefix": self.prefix,
                "continuation": self.continuation,
                "text": self.text,
                "weight": self.weight,
                "normalized_weight": self.normalized_weight,
                "reward": self.reward,
                "advantage": self.advantage,
                "metadata": self.metadata,
            },
            self.extra,
        )


@dataclass
class DPOPair:
    id: str
    prompt: str
    question: str
    prefix: str
    chosen: str
    rejected: str
    chosen_reward: float
    rejected_reward: float
    reward_gap: float
    metadata: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    _KNOWN_FIELDS: ClassVar[set[str]] = {
        "id",
        "prompt",
        "question",
        "prefix",
        "chosen",
        "rejected",
        "chosen_reward",
        "rejected_reward",
        "reward_gap",
        "metadata",
    }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DPOPair":
        return cls(
            id=str(_require(data, "id")),
            prompt=str(_require(data, "prompt")),
            question=str(_require(data, "question")),
            prefix=str(_require(data, "prefix")),
            chosen=str(_require(data, "chosen")),
            rejected=str(_require(data, "rejected")),
            chosen_reward=float(_require(data, "chosen_reward")),
            rejected_reward=float(_require(data, "rejected_reward")),
            reward_gap=float(_require(data, "reward_gap")),
            metadata=dict(data.get("metadata") or {}),
            extra=_split_extra(data, cls._KNOWN_FIELDS),
        )

    def to_dict(self) -> dict[str, Any]:
        return _base_dict(
            {
                "id": self.id,
                "prompt": self.prompt,
                "question": self.question,
                "prefix": self.prefix,
                "chosen": self.chosen,
                "rejected": self.rejected,
                "chosen_reward": self.chosen_reward,
                "rejected_reward": self.rejected_reward,
                "reward_gap": self.reward_gap,
                "metadata": self.metadata,
            },
            self.extra,
        )
