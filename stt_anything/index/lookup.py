from __future__ import annotations

from ..models import Candidate, EncodedSequence, TermIndex
from .abs import IndexLookup
from .matching import phone_matches, subsequence_matches


class DTWLookup(IndexLookup):
    name = "dtw"

    def __init__(self, top_k: int = 3) -> None:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        self.top_k = top_k

    def search(self, index: TermIndex, query: EncodedSequence) -> list[Candidate]:
        output: list[Candidate] = []
        for entry in index.entries:
            template = entry.sequence
            if (
                query.kind != "frames"
                or template.kind != "frames"
                or query.values.ndim != 2
                or template.values.ndim != 2
                or query.values.shape[1] != template.values.shape[1]
            ):
                raise ValueError("DTW lookup needs matching frame vector dimensions")
            output.extend(
                Candidate(entry.term_id, match)
                for match in subsequence_matches(
                    template.values, query.values, query.spans, self.top_k
                )
            )
        return output


class PhonemeLookup(IndexLookup):
    name = "phoneme"
    kind = "phones"
    word_delimiter: int | None = None

    def __init__(self, top_k: int = 3) -> None:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        self.top_k = top_k

    def search(self, index: TermIndex, query: EncodedSequence) -> list[Candidate]:
        output: list[Candidate] = []
        for entry in index.entries:
            template = entry.sequence
            if (
                query.kind != self.kind
                or template.kind != self.kind
                or query.values.ndim != 1
                or template.values.ndim != 1
            ):
                raise ValueError(
                    f"{type(self).__name__.removesuffix('Lookup')} lookup needs matching one-dimensional token sequences"
                )
            output.extend(
                Candidate(entry.term_id, match)
                for match in phone_matches(
                    template.values.tolist(),
                    query.values.tolist(),
                    query.spans,
                    self.top_k,
                    self.word_delimiter,
                )
            )
        return output


class CharacterLookup(PhonemeLookup):
    """Match language-specific CTC character tokens without producing a transcript."""

    name = "characters"
    kind = "characters"
    word_delimiter = 4
