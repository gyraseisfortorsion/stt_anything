import json
import subprocess
import wave
from dataclasses import replace
from unittest.mock import Mock

import numpy as np
import pytest
from conftest import FakeEncoder, FakeLookup, FakeSource

from stt_anything import create_index, run
from stt_anything.audio import read_audio, write_audio
from stt_anything.detection import KeywordDetector, overlap, retier, select_hits
from stt_anything.encoder.abs import Encoder
from stt_anything.glossary import load_glossary
from stt_anything.index import (
    IndexBuilder,
    IndexLookup,
    IndexStore,
    NpzIndexStore,
    TemplateIndexBuilder,
)
from stt_anything.index.builder import default_thresholds
from stt_anything.models import (
    Candidate,
    EncodedSequence,
    Hit,
    IndexEntry,
    Match,
    Term,
    frame_spans,
)
from stt_anything.templates.abs import TemplateSource


def test_audio_roundtrip_and_conversion(tmp_path):
    audio = (np.sin(2 * np.pi * 440 * np.arange(8000) / 8000) * 0.2).astype(np.float32)
    path = tmp_path / "sample.wav"
    write_audio(path, audio)
    assert np.max(np.abs(read_audio(path) - audio)) < 0.001
    with wave.open(str(path), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(8000)
        output.writeframes(
            np.column_stack((audio, audio))
            .astype(np.float32)
            .__mul__(32767)
            .astype("<i2")
            .tobytes()
        )
    converted = read_audio(path)
    assert len(converted) == 16000
    assert np.corrcoef(converted[::2], audio)[0, 1] > 0.99


@pytest.mark.parametrize(
    "result,error",
    [
        (FileNotFoundError(), RuntimeError),
        (subprocess.CalledProcessError(1, "ffmpeg", stderr=b"bad input"), RuntimeError),
        (subprocess.CompletedProcess([], 0, stdout=b"", stderr=b""), ValueError),
    ],
)
def test_audio_errors(monkeypatch, result, error):
    mock = Mock(side_effect=result) if isinstance(result, Exception) else Mock(return_value=result)
    monkeypatch.setattr("stt_anything.audio.subprocess.run", mock)
    with pytest.raises(error):
        read_audio("missing.ogg")


def test_glossary_unicode_forms_and_language(glossary):
    terms = load_glossary(glossary)
    assert terms[0].forms == ("омепразол", "Омепразол")
    glossary.write_text(
        glossary.read_text() + "\n" + '{"id":"de-one","language":"de","display":"test"}\n'
    )
    assert len(load_glossary(glossary, "ru")) == 1
    assert load_glossary(glossary, "de")[0].language == "de"
    with pytest.raises(ValueError, match="No glossary"):
        load_glossary(glossary, "fr")


@pytest.mark.parametrize(
    "line",
    [
        "{",
        "{}",
        "null",
        "[]",
        '{"id":"","language":"ru","display":"x"}',
        '{"id":"one","language":"ru","display":"x","aliases":"x"}',
        '{"id":"one","language":"ru","display":"x","spoken_forms":[2]}',
    ],
)
def test_glossary_invalid(glossary, line):
    glossary.write_text(line)
    with pytest.raises(ValueError, match=":1:"):
        load_glossary(glossary)


def test_glossary_duplicate(glossary):
    glossary.write_text(glossary.read_text() * 2)
    with pytest.raises(ValueError, match="duplicate id"):
        load_glossary(glossary)


@pytest.mark.parametrize(
    "values,spans,kind",
    [
        (np.zeros((1, 1, 1)), np.zeros((1, 2)), "frames"),
        (np.zeros(0), np.zeros((0, 2)), "frames"),
        (np.zeros(2), np.zeros((1, 2)), "frames"),
        (np.array([np.nan]), np.zeros((1, 2)), "frames"),
        (np.zeros(1), np.array([[0, np.inf]]), "frames"),
        (np.zeros(1), np.array([[-1, 1]]), "frames"),
        (np.zeros(1), np.array([[2, 1]]), "frames"),
        (np.zeros(2), np.array([[2, 3], [1, 2]]), "frames"),
        (np.zeros(1), np.zeros((1, 2)), ""),
    ],
)
def test_invalid_encoded_sequences(values, spans, kind):
    with pytest.raises(ValueError):
        EncodedSequence(values, spans, kind)


def test_index_contract_and_thresholds(index):
    assert index.language == "ru"
    np.testing.assert_allclose(frame_spans(2, 0.02, 1), [[1, 1.02], [1.02, 1.04]])
    for updates in [
        dict(terms=()),
        dict(terms=(index.terms[0], index.terms[0])),
        dict(entries=()),
        dict(entries=(replace(index.entries[0], term_id="missing"),)),
        dict(terms=(index.terms[0], Term("three", "en", "x"))),
        dict(strong_threshold=0.1),
        dict(possible_threshold=-1),
    ]:
        with pytest.raises(ValueError):
            replace(index, **updates)
    assert default_thresholds("whisper", "en") == (0.35, 0.25)
    assert default_thresholds("whisper", "ru") == (0.33, 0.24)
    assert default_thresholds("mfcc", "en") == (0.62, 0.5)
    assert default_thresholds("mfcc", "ru") == (0.58, 0.48)
    assert default_thresholds("custom", "de") == (0.55, 0.35)


def test_abstract_contracts_raise():
    calls = [
        (Encoder.encode, (None, np.zeros(1))),
        (TemplateSource.synthesize, (None, "term", "en")),
        (IndexBuilder.create, (None, [], None, None)),
        (IndexStore.save, (None, None, "path")),
        (IndexStore.load, (None, "path")),
        (IndexLookup.search, (None, None, None)),
    ]
    for method, args in calls:
        with pytest.raises(NotImplementedError):
            method(*args)
    for abstract in [Encoder, TemplateSource, IndexBuilder, IndexStore, IndexLookup]:
        with pytest.raises(TypeError):
            abstract()


def test_create_index_and_safe_roundtrip(glossary, tmp_path):
    encoder, source = FakeEncoder(), FakeSource()
    index = create_index(glossary, "ru", encoder, source)
    assert len(index.entries) == 2
    assert (
        create_index(index.terms, "ru", encoder, source, builder=TemplateIndexBuilder()).terms
        == index.terms
    )
    with pytest.raises(ValueError, match="empty"):
        TemplateIndexBuilder().create([], encoder, source)
    path = tmp_path / "nested" / "index.bin"
    storage = NpzIndexStore()
    storage.save(index, path)
    loaded = storage.load(path)
    assert loaded.terms == index.terms
    assert loaded.encoder_signature == encoder.signature
    np.testing.assert_array_equal(
        loaded.entries[0].sequence.values, index.entries[0].sequence.values
    )
    with np.load(path, allow_pickle=False) as data:
        arrays = dict(data)
    metadata = json.loads(str(arrays["metadata"]))
    metadata["version"] = 2
    arrays["metadata"] = np.asarray(json.dumps(metadata))
    with path.open("wb") as output:
        np.savez(output, **arrays)
    with pytest.raises(ValueError, match="version"):
        storage.load(path)


def test_selection_and_competitors(index):
    candidates = [
        Candidate("one", Match(0.1, 0.5, 0.9)),
        Candidate("two", Match(0.12, 0.52, 0.85)),
        Candidate("one", Match(0.2, 0.55, 0.8)),
        Candidate("one", Match(0.7, 1.2, 0.6)),
        Candidate("one", Match(0.6, 0.8, 0.1)),
        Candidate("one", Match(1.1, 1.3, 0.7)),
    ]
    audio = np.ones(16000, dtype=np.float32) * 0.1
    chosen = select_hits(candidates, audio, index, 0.8, 0.5)
    assert [(h.term_id, h.tier) for h in chosen] == [
        ("one", "strong"),
        ("two", "possible"),
        ("one", "possible"),
    ]
    assert chosen[-1].end == 1.0
    assert select_hits(candidates, np.zeros_like(audio), index, 0.8, 0.5) == ()
    assert select_hits([candidates[-1]], audio, index, 0.8, 0.5, min_energy=0) == ()
    assert overlap(0, 0, 0, 0) == 0
    assert retier([Hit("one", "x", 0, 1, 0.1, "possible")], 0.8, 0.5) == ()
    with pytest.raises(ValueError, match="energy"):
        select_hits([], audio, index, 0.8, 0.5, -1)


@pytest.mark.parametrize(
    "candidate",
    [
        Candidate("missing", Match(0, 1, 0.5)),
        Candidate("one", Match(0, 1, float("nan"))),
        Candidate("one", Match(0, 1, 2)),
        Candidate("one", Match(-1, 1, 0.5)),
        Candidate("one", Match(1, 0, 0.5)),
    ],
)
def test_invalid_candidates(index, candidate):
    with pytest.raises(ValueError, match="candidate"):
        select_hits([candidate], np.ones(16000, dtype=np.float32), index, 0.8, 0.5)


def test_detector_and_api(index, tmp_path, monkeypatch):
    audio = np.ones(16000, dtype=np.float32) * 0.1
    detector = KeywordDetector(FakeEncoder(), FakeLookup())
    result = detector.detect(audio, index)
    assert result.hits[0].tier == "strong"
    assert "transcript" not in result.to_dict()
    assert "not probabilities" in result.to_dict()["score_note"]
    path = tmp_path / "sample.wav"
    write_audio(path, audio)
    assert (
        detector.run(path, index, strong_threshold=1, possible_threshold=0.4).hits[0].tier
        == "possible"
    )
    assert run(path, index, encoder=FakeEncoder(), lookup=FakeLookup()).duration == 1
    store = NpzIndexStore()
    index_path = tmp_path / "index.npz"
    store.save(index, index_path)
    monkeypatch.setattr("stt_anything.components.encoder", lambda *_: FakeEncoder())
    monkeypatch.setattr("stt_anything.components.lookup", lambda *_: FakeLookup())
    assert run(path, index_path).hits
    assert run(path, index_path, store=store).hits
    phones = EncodedSequence(np.array([1, 2]), frame_spans(2, 0.02), "phones")
    phone_index = replace(index, entries=(IndexEntry("one", "x", phones),))
    assert run(path, phone_index).hits
    with pytest.raises(ValueError, match="signature"):
        detector.detect(audio, replace(index, encoder_signature="other"))
    for invalid in [np.zeros(0), np.zeros((2, 3)), np.array([np.nan])]:
        with pytest.raises(ValueError, match="Audio"):
            detector.detect(invalid, index)
    with pytest.raises(ValueError, match="Thresholds"):
        detector.detect(audio, index, strong_threshold=0.2)


def test_legacy_index_rejected(tmp_path):
    path = tmp_path / "legacy.npz"
    np.savez(path, ids=np.array(["one"]))
    with pytest.raises(ValueError, match="recreate"):
        NpzIndexStore().load(path)
    metadata = {"version": 1, "terms": None}
    np.savez(path, metadata=np.asarray(json.dumps(metadata)))
    with pytest.raises(ValueError, match="recreate"):
        NpzIndexStore().load(path)


def test_nonfinite_candidate_time(index):
    with pytest.raises(ValueError, match="candidate"):
        select_hits(
            [Candidate("one", Match(0, float("inf"), 0.5))],
            np.ones(16000, dtype=np.float32),
            index,
            0.8,
            0.5,
        )
