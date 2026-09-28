from __future__ import annotations

from collections.abc import Sequence

from ..encoder.abs import Encoder
from ..models import IndexEntry, Term, TermIndex
from ..templates.abs import TemplateSource
from .abs import IndexBuilder


def default_thresholds(encoder: str, language: str) -> tuple[float, float]:
    if encoder == "mfcc":
        return (0.62, 0.50) if language == "en" else (0.58, 0.48)
    if encoder.startswith("whisper"):
        return (0.35, 0.25) if language == "en" else (0.33, 0.24)
    if encoder == "russian-ctc":
        return 0.85, 0.60
    return 0.55, 0.35


class TemplateIndexBuilder(IndexBuilder):
    def create(self, terms: Sequence[Term], encoder: Encoder, source: TemplateSource) -> TermIndex:
        if not terms:
            raise ValueError("Cannot create an index from an empty glossary")
        entries = tuple(
            IndexEntry(term.id, form, encoder.encode_term(form, term.language, source))
            for term in terms
            for form in term.forms
        )
        strong, possible = default_thresholds(encoder.name, terms[0].language)
        return TermIndex(
            tuple(terms),
            entries,
            encoder.name,
            encoder.signature,
            source.signature,
            strong,
            possible,
        )
