"""Test every adapter with local fakes; never download model weights in unit tests."""

import os

os.environ["NUMBA_DISABLE_JIT"] = "1"

import numpy as np
import pytest

from stt_anything.encoder.abs import Encoder
from stt_anything.index.abs import IndexLookup
from stt_anything.models import (
    Candidate,
    EncodedSequence,
    IndexEntry,
    Match,
    Term,
    TermIndex,
    frame_spans,
)
from stt_anything.templates.abs import TemplateSource


class FakeEncoder(Encoder):
    name = "fake"
    signature = "fake:v1"

    def encode(self, audio, sample_rate=16000):
        values = np.column_stack((np.arange(6), -np.arange(6))).astype(np.float32)
        return EncodedSequence(values, frame_spans(6, 0.02))


class FakeSource(TemplateSource):
    signature = "fake-source:v1"

    def synthesize(self, text, language):
        return np.ones(16000, dtype=np.float32) * 0.1


class FakeLookup(IndexLookup):
    name = "fake-lookup"

    def search(self, index, query):
        return [Candidate(index.terms[0].id, Match(0.1, 0.4, 0.9))]


@pytest.fixture
def index():
    encoder = FakeEncoder()
    sequence = encoder.encode(np.ones(16000, dtype=np.float32))
    return TermIndex(
        (Term("one", "ru", "омепразол"), Term("two", "ru", "метформин")),
        (IndexEntry("one", "омепразол", sequence),),
        encoder.name,
        encoder.signature,
        "fake-source:v1",
        0.8,
        0.5,
    )


@pytest.fixture
def glossary(tmp_path):
    path = tmp_path / "glossary.jsonl"
    path.write_text(
        '{"id":"one","language":"ru","display":"омепразол","aliases":["Омепразол"],"spoken_forms":["омепразол"]}\n',
        encoding="utf-8",
    )
    return path
