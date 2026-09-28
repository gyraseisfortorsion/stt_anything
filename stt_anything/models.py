"""Provider-independent contracts. Timestamps are seconds from file start."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]
Array = NDArray[Any]


@dataclass(frozen=True)
class Term:
    id: str
    language: str
    display: str
    aliases: tuple[str, ...] = ()
    spoken_forms: tuple[str, ...] = ()

    @property
    def forms(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((self.display, *self.aliases, *self.spoken_forms)))


@dataclass(frozen=True)
class EncodedSequence:
    """Frame vectors or tokens with explicit time spans (shape N x 2)."""

    values: Array
    spans: FloatArray
    kind: str = "frames"

    def __post_init__(self) -> None:
        if self.values.ndim not in (1, 2) or len(self.values) == 0:
            raise ValueError("Encoded values must be a nonempty one- or two-dimensional array")
        if self.spans.shape != (len(self.values), 2):
            raise ValueError("Every encoded element must have a start/end span")
        if not np.isfinite(self.values).all() or not np.isfinite(self.spans).all():
            raise ValueError("Encoded values and spans must be finite")
        if (self.spans < 0).any() or (self.spans[:, 1] < self.spans[:, 0]).any():
            raise ValueError("Spans must satisfy 0 <= start <= end")
        if (np.diff(self.spans[:, 0]) < 0).any() or not self.kind:
            raise ValueError("Spans must be ordered and kind must be nonempty")


def frame_spans(count: int, step: float, offset: float = 0) -> FloatArray:
    starts = np.arange(count, dtype=np.float32) * step + offset
    return np.column_stack((starts, starts + step)).astype(np.float32)


@dataclass(frozen=True)
class IndexEntry:
    term_id: str
    form: str
    sequence: EncodedSequence


@dataclass(frozen=True)
class TermIndex:
    terms: tuple[Term, ...]
    entries: tuple[IndexEntry, ...]
    encoder_name: str
    encoder_signature: str
    source_signature: str
    strong_threshold: float = 0.55
    possible_threshold: float = 0.35

    def __post_init__(self) -> None:
        ids = {term.id for term in self.terms}
        if not ids or len(ids) != len(self.terms) or not self.entries:
            raise ValueError("Index needs unique terms and nonempty entries")
        if any(entry.term_id not in ids for entry in self.entries):
            raise ValueError("Index entry refers to an unknown term")
        if len({term.language for term in self.terms}) != 1:
            raise ValueError("An index must contain exactly one language")
        validate_thresholds(self.strong_threshold, self.possible_threshold)

    @property
    def language(self) -> str:
        return self.terms[0].language


def validate_thresholds(strong: float, possible: float) -> None:
    if not 0 <= possible <= strong <= 1:
        raise ValueError("Thresholds must satisfy 0 <= possible <= strong <= 1")


@dataclass(frozen=True)
class Match:
    start: float
    end: float
    score: float


@dataclass(frozen=True)
class Candidate:
    term_id: str
    match: Match


@dataclass(frozen=True)
class Hit:
    term_id: str
    term: str
    start: float
    end: float
    score: float
    tier: Literal["strong", "possible"]


@dataclass(frozen=True)
class DetectionResult:
    language: str
    duration: float
    encoder: str
    lookup: str
    hits: tuple[Hit, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "duration": round(self.duration, 3),
            "encoder": self.encoder,
            "lookup": self.lookup,
            "hits": [asdict(hit) for hit in self.hits],
            "score_note": "Match scores are ranking scores, not probabilities; timestamps are estimates.",
        }
