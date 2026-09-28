"""Inspect lookup scores without presenting rejected candidates as detections."""

from __future__ import annotations

from typing import Any

import numpy as np

from stt_anything.index.abs import IndexLookup
from stt_anything.models import Candidate, EncodedSequence, FloatArray, TermIndex


class ObservedLookup(IndexLookup):
    def __init__(self, delegate: IndexLookup) -> None:
        self.delegate = delegate
        self.name = delegate.name
        self.candidates: list[Candidate] = []

    def search(self, index: TermIndex, query: EncodedSequence) -> list[Candidate]:
        self.candidates = self.delegate.search(index, query)
        return self.candidates


def diagnostics(audio: FloatArray, candidates: list[Candidate], possible: float) -> dict[str, Any]:
    rms = float(np.sqrt(np.mean(audio * audio)))
    peak = float(np.max(np.abs(audio)))
    strongest = max((candidate.match.score for candidate in candidates), default=None)
    return {
        "audio_rms": round(rms, 6),
        "audio_peak": round(peak, 6),
        "quiet_audio": rms < 0.003,
        "candidate_count": len(candidates),
        "candidates_above_threshold": sum(
            candidate.match.score >= possible for candidate in candidates
        ),
        "best_candidate_score": strongest,
    }
