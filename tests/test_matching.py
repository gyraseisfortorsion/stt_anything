from dataclasses import replace

import numpy as np
import pytest

from stt_anything.encoder.mfcc import MFCCEncoder, mfcc_features
from stt_anything.index import DTWLookup, PhonemeLookup
from stt_anything.index.matching import (
    _dtw,
    edit_distance,
    normalize_frames,
    phone_matches,
    subsequence_matches,
)
from stt_anything.models import EncodedSequence, IndexEntry, frame_spans


def test_mfcc_and_normalization():
    audio = np.sin(np.arange(16000) * 0.1).astype(np.float32)
    encoder = MFCCEncoder()
    sequence = encoder.encode(audio)
    assert sequence.values.shape == (51, 13)
    np.testing.assert_allclose(sequence.values[:, 0], 0)
    assert sequence.spans[-1, 1] == pytest.approx(1.02)
    assert mfcc_features(audio)[1] == 0.02
    np.testing.assert_array_equal(
        normalize_frames(np.ones((4, 3), dtype=np.float32)), np.zeros((4, 3))
    )
    for invalid, rate in [(audio, 8000), (np.zeros((2, 400)), 16000), (np.zeros(399), 16000)]:
        with pytest.raises(ValueError):
            encoder.encode(invalid, rate)


def test_dtw_repeated_localization_and_stretching(index):
    rng = np.random.default_rng(4)
    query = rng.normal(size=(10, 16)).astype(np.float32)
    audio = np.vstack([rng.normal(size=(6, 16)), query, rng.normal(size=(8, 16)), query]).astype(
        np.float32
    )
    spans = frame_spans(len(audio), 0.02, 2)
    found = subsequence_matches(query, audio, spans, 20)
    assert any(abs(match.start - 2.12) <= 0.04 and abs(match.end - 2.32) <= 0.04 for match in found)
    assert any(abs(match.start - 2.48) <= 0.04 for match in found)
    assert len(found) >= 2
    assert subsequence_matches(query, audio, spans, 1) == found[:1]
    template = EncodedSequence(query, frame_spans(len(query), 0.02))
    index = replace(index, entries=(IndexEntry("one", "x", template),))
    assert DTWLookup().search(index, EncodedSequence(audio, spans))
    # Stress all three DTW path directions and rejected durations.
    for shape in [(20, 5), (5, 80), (12, 25)]:
        cost = rng.uniform(0, 2, shape).astype(np.float32)
        assert np.isfinite(_dtw(cost)[0]).all()
    assert (
        subsequence_matches(query, np.zeros((2, 16), dtype=np.float32), frame_spans(2, 0.02)) == []
    )
    assert subsequence_matches(np.zeros((0, 16), dtype=np.float32), audio, spans) == []
    assert (
        subsequence_matches(query, np.zeros((0, 16), dtype=np.float32), frame_spans(0, 0.02)) == []
    )


@pytest.mark.parametrize("top_k", [0, -1])
def test_lookup_top_k(top_k):
    for lookup in [DTWLookup, PhonemeLookup]:
        with pytest.raises(ValueError):
            lookup(top_k)


def test_lookup_incompatibility(index):
    frames = index.entries[0].sequence
    bad = [
        EncodedSequence(np.array([1, 2]), frame_spans(2, 0.02), "phones"),
        EncodedSequence(np.zeros((3, 7)), frame_spans(3, 0.02)),
        EncodedSequence(np.zeros(3), frame_spans(3, 0.02)),
    ]
    for query in bad:
        with pytest.raises(ValueError, match="DTW"):
            DTWLookup().search(index, query)
    with pytest.raises(ValueError, match="DTW"):
        DTWLookup().search(
            replace(index, entries=(replace(index.entries[0], sequence=bad[0]),)), frames
        )
    with pytest.raises(ValueError, match="DTW"):
        DTWLookup().search(
            replace(index, entries=(replace(index.entries[0], sequence=bad[2]),)), frames
        )
    for query in [frames, EncodedSequence(np.zeros((3, 2)), frame_spans(3, 0.02), "phones")]:
        with pytest.raises(ValueError, match="Phoneme"):
            PhonemeLookup().search(index, query)
    phones = EncodedSequence(np.array([1, 2]), frame_spans(2, 0.02), "phones")
    for template in [frames, EncodedSequence(np.zeros((3, 2)), frame_spans(3, 0.02), "phones")]:
        with pytest.raises(ValueError, match="Phoneme"):
            PhonemeLookup().search(
                replace(index, entries=(replace(index.entries[0], sequence=template),)), phones
            )


def test_phone_edit_matching(index):
    phones = [9, 3, 4, 5, 8, 3, 4, 5]
    spans = frame_spans(len(phones), 0.1)
    assert edit_distance([3, 4, 5], [3, 7, 5]) == 1
    assert edit_distance([], [1, 2]) == 2
    found = phone_matches([3, 4, 5], phones, spans, 20)
    assert found[0].score == 1
    assert found[0].start == pytest.approx(0.1)
    assert found[0].end == pytest.approx(0.4)
    assert phone_matches([3, 4, 5], phones, spans, 1) == found[:1]
    assert phone_matches([], phones, spans) == []
    assert phone_matches([1], [], frame_spans(0, 0.02)) == []
    # No permissible window is long enough to fit the template.
    assert phone_matches(list(range(20)), [1], frame_spans(1, 0.02)) == []
    template = EncodedSequence(np.array([3, 4, 5]), frame_spans(3, 0.02), "phones")
    index = replace(index, entries=(IndexEntry("one", "x", template),))
    candidates = PhonemeLookup().search(index, EncodedSequence(np.array(phones), spans, "phones"))
    assert candidates[0].term_id == "one"


def test_token_paths_cannot_cross_long_pauses_or_stretched_tokens():
    spans = np.array([[0, 0.1], [0.12, 0.2], [3, 3.1]], dtype=np.float32)
    hits = phone_matches([1, 2, 3], [1, 2, 3], spans, 20)
    assert all(hit.score < 1 and hit.end - hit.start < 1.5 for hit in hits)
    assert phone_matches([1], [1], np.array([[0, 8]], dtype=np.float32)) == []
    continuous = frame_spans(3, 0.1)
    assert phone_matches([1, 2, 3], [1, 2, 3], continuous)[0].score == 1


def test_character_lookup_and_api(index, tmp_path, monkeypatch):
    from stt_anything import components
    from stt_anything.api import run
    from stt_anything.audio import write_audio
    from stt_anything.encoder.abs import Encoder
    from stt_anything.index import CharacterLookup

    sequence = EncodedSequence(np.array([1, 2, 3]), frame_spans(3, 0.1), "characters")

    class CharacterEncoder(Encoder):
        name = "russian-ctc"
        signature = "fake:v1"

        def encode(self, audio, sample_rate=16000):
            return sequence

    index = replace(index, entries=(IndexEntry("one", "x", sequence),))
    assert CharacterLookup().search(index, sequence)[0].match.score == 1
    with pytest.raises(ValueError, match="Character"):
        CharacterLookup().search(index, replace(sequence, kind="phones"))
    path = tmp_path / "audio.wav"
    write_audio(path, np.ones(16000, dtype=np.float32) * 0.1)
    monkeypatch.setattr(components, "encoder", lambda *args: CharacterEncoder())
    assert run(path, index).lookup == "characters"


def test_character_word_boundaries_reject_word_fragments():
    spans = frame_spans(5, 0.1)
    assert all(
        h.score < 1 for h in phone_matches([2, 3], [1, 2, 3, 9, 8], spans, 20, word_delimiter=9)
    )
    assert all(
        h.score < 1 for h in phone_matches([1, 2], [1, 2, 3, 9, 8], spans, 20, word_delimiter=9)
    )
    found = phone_matches([2, 3], [1, 9, 2, 3, 9], spans, 20, word_delimiter=9)
    assert found[0].score == 1 and found[0].start == pytest.approx(0.2)
    gaps = np.array([[0, 0.1], [1, 1.1], [1.12, 1.2]], dtype=np.float32)
    assert phone_matches([2, 3], [1, 2, 3], gaps, 20, word_delimiter=9)[0].score == 1
