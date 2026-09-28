"""Keyword-only orchestration. No STT provider or transcription API is required."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Literal

import numpy as np

from .audio import SAMPLE_RATE, read_audio
from .encoder.abs import Encoder
from .index.abs import IndexLookup
from .models import Candidate, DetectionResult, FloatArray, Hit, TermIndex, validate_thresholds


def overlap(start: float, end: float, other_start: float, other_end: float) -> float:
    return max(0, min(end, other_end) - max(start, other_start)) / max(
        min(end - start, other_end - other_start), 1e-6
    )


def retier(hits: tuple[Hit, ...] | list[Hit], strong: float, possible: float) -> tuple[Hit, ...]:
    """Only the best overlapping term can be strong at a given time span."""
    validate_thresholds(strong, possible)
    chosen = [hit for hit in hits if hit.score >= possible]
    output = []
    for hit in chosen:
        competitor = any(
            other.term_id != hit.term_id
            and (other.score, other.term_id) > (hit.score, hit.term_id)
            and overlap(hit.start, hit.end, other.start, other.end) > 0.5
            for other in chosen
        )
        tier: Literal["strong", "possible"] = (
            "strong" if hit.score >= strong and not competitor else "possible"
        )
        output.append(replace(hit, tier=tier))
    return tuple(output)


def select_hits(
    candidates: list[Candidate],
    audio: FloatArray,
    index: TermIndex,
    strong: float,
    possible: float,
    min_energy: float = 0.003,
) -> tuple[Hit, ...]:
    validate_thresholds(strong, possible)
    if min_energy < 0:
        raise ValueError("min_energy cannot be negative")
    terms = {term.id: term for term in index.terms}
    duration = len(audio) / SAMPLE_RATE
    output: list[Hit] = []
    for candidate in sorted(candidates, key=lambda item: item.match.score, reverse=True):
        match = candidate.match
        if (
            candidate.term_id not in terms
            or not np.isfinite((match.start, match.end, match.score)).all()
            or not 0 <= match.score <= 1
            or not 0 <= match.start < match.end
        ):
            raise ValueError("Lookup returned an invalid candidate")
        end = min(duration, match.end)
        chunk = audio[int(match.start * SAMPLE_RATE) : int(end * SAMPLE_RATE)]
        energy = float(np.sqrt(np.mean(chunk * chunk))) if len(chunk) else 0.0
        if match.score < possible or energy < min_energy or end <= match.start:
            continue
        if any(
            hit.term_id == candidate.term_id and overlap(hit.start, hit.end, match.start, end) > 0.3
            for hit in output
        ):
            continue
        output.append(
            Hit(
                candidate.term_id,
                terms[candidate.term_id].display,
                match.start,
                end,
                match.score,
                "possible",
            )
        )
    return retier(tuple(sorted(output, key=lambda hit: (hit.start, -hit.score))), strong, possible)


class KeywordDetector:
    def __init__(self, encoder: Encoder, lookup: IndexLookup) -> None:
        self.encoder = encoder
        self.lookup = lookup

    def detect(
        self,
        audio: FloatArray,
        index: TermIndex,
        *,
        strong_threshold: float | None = None,
        possible_threshold: float | None = None,
        min_energy: float = 0.003,
    ) -> DetectionResult:
        if audio.ndim != 1 or len(audio) == 0 or not np.isfinite(audio).all():
            raise ValueError("Audio must be a finite, nonempty mono array at 16 kHz")
        if index.encoder_signature != self.encoder.signature:
            raise ValueError("Encoder does not match the index signature; recreate the index")
        strong = index.strong_threshold if strong_threshold is None else strong_threshold
        possible = index.possible_threshold if possible_threshold is None else possible_threshold
        validate_thresholds(strong, possible)
        query = self.encoder.encode(audio)
        hits = select_hits(
            self.lookup.search(index, query), audio, index, strong, possible, min_energy
        )
        return DetectionResult(
            index.language, len(audio) / SAMPLE_RATE, self.encoder.name, self.lookup.name, hits
        )

    def run(
        self,
        audio_path: str | Path,
        index: TermIndex,
        *,
        strong_threshold: float | None = None,
        possible_threshold: float | None = None,
        min_energy: float = 0.003,
    ) -> DetectionResult:
        return self.detect(
            read_audio(audio_path),
            index,
            strong_threshold=strong_threshold,
            possible_threshold=possible_threshold,
            min_energy=min_energy,
        )
