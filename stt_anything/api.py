from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from . import components
from .detection import KeywordDetector
from .encoder.abs import Encoder
from .glossary import load_glossary
from .index import IndexBuilder, IndexLookup, IndexStore, NpzIndexStore, TemplateIndexBuilder
from .models import DetectionResult, Term, TermIndex
from .templates.abs import TemplateSource


def create_index(
    glossary: str | Path | Sequence[Term],
    language: str,
    encoder: Encoder,
    source: TemplateSource,
    *,
    builder: IndexBuilder | None = None,
) -> TermIndex:
    terms = (
        load_glossary(glossary, language) if isinstance(glossary, (str, Path)) else list(glossary)
    )
    return (builder or TemplateIndexBuilder()).create(terms, encoder, source)


def run(
    audio_path: str | Path,
    index: TermIndex | str | Path,
    *,
    encoder: Encoder | None = None,
    lookup: IndexLookup | None = None,
    store: IndexStore | None = None,
    offline: bool = True,
    strong_threshold: float | None = None,
    possible_threshold: float | None = None,
) -> DetectionResult:
    loaded = (store or NpzIndexStore()).load(index) if isinstance(index, (str, Path)) else index
    selected_encoder = encoder or components.encoder(loaded.encoder_name, loaded.language, offline)
    selected_lookup = lookup or components.lookup(
        {"phones": "phoneme", "characters": "characters"}.get(
            loaded.entries[0].sequence.kind, "dtw"
        )
    )
    return KeywordDetector(selected_encoder, selected_lookup).run(
        audio_path, loaded, strong_threshold=strong_threshold, possible_threshold=possible_threshold
    )
