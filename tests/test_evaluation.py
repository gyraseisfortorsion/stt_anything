import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from conftest import FakeEncoder, FakeLookup, FakeSource
from test_adapters import Voice

from stt_anything.evaluation import benchmark, synthetic
from stt_anything.evaluation.metrics import _is_true, calibrate, metrics
from stt_anything.evaluation.synthetic import Case, _modify, _piper, _say, generate_cases
from stt_anything.models import Hit


def test_speakers_and_conditions(monkeypatch):
    monkeypatch.setitem(sys.modules, "piper", SimpleNamespace(SynthesisConfig=lambda **kw: kw))
    audio = np.sin(np.arange(16000) * 0.1).astype(np.float32) * 0.2
    process = Mock()
    monkeypatch.setattr(synthetic.subprocess, "run", process)
    monkeypatch.setattr(synthetic, "read_audio", lambda _: audio)
    np.testing.assert_array_equal(_say("term", "voice"), audio)
    assert process.call_args.args[0][-1] == "term"
    np.testing.assert_array_equal(_piper("term", Voice()), audio)
    assert len(_modify(audio, "speed", 1)) == int(16000 / 1.12)
    np.testing.assert_array_equal(_modify(audio, "phone", 42), _modify(audio, "phone", 42))
    assert not np.array_equal(_modify(audio, "phone", 42), _modify(audio, "phone", 43))
    np.testing.assert_array_equal(_modify(audio, "clean", 0), audio)


def test_synthetic_generation_and_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("STT_ANYTHING_CACHE", str(tmp_path / "cache"))
    audio = np.ones(16000, dtype=np.float32) * 0.1
    say = Mock(return_value=audio)
    monkeypatch.setattr(synthetic, "_say", say)
    monkeypatch.setattr(synthetic, "_piper", lambda *args: audio)
    monkeypatch.setattr(synthetic, "prepare_eval_russian_voice", lambda **kw: tmp_path / "voice")
    monkeypatch.setitem(
        sys.modules, "piper", SimpleNamespace(PiperVoice=SimpleNamespace(load=lambda _: Voice()))
    )
    glossary = tmp_path / "terms.jsonl"
    rows = [
        dict(
            id=f"{language}-{idx}",
            language=language,
            display=f"term{idx}",
            spoken_forms=["spoken"] if idx == 0 else [],
        )
        for language in ("en", "ru")
        for idx in range(3)
    ]
    glossary.write_text("\n".join(json.dumps(row) for row in rows))
    for language in ("en", "ru"):
        cases = generate_cases(glossary, language, offline=True)
        assert len(cases) == 12
        assert len({case.path for case in cases}) == 12
        assert all(case.path.exists() for case in cases)
        assert cases[3].start < cases[2].start  # speed-adjusted time span
        previous_calls = say.call_count
        assert generate_cases(glossary, language) == cases
        assert say.call_count == previous_calls
        # Missing audio invalidates a complete manifest.
        cases[0].path.unlink()
        assert len(generate_cases(glossary, language)) == 12
        manifest = cases[0].path.parent / "manifest.json"
        manifest.write_text("[]")
        assert len(generate_cases(glossary, language)) == 12
    assert any(call.args[0] == "spoken" for call in say.call_args_list)


def test_metrics_and_development_calibration():
    positive = Case(Path("positive.wav"), "test", "ru", "one", 1.0, 2.0)
    negative = Case(Path("negative.wav"), "test", "ru", None, None, None)
    true = Hit("one", "x", 1.1, 2.1, 0.9, "possible")
    false = Hit("wrong", "y", 3, 4, 0.7, "possible")
    assert _is_true(positive, true)
    assert not _is_true(negative, true)
    assert not _is_true(positive, replace(true, end=1.15))
    assert not _is_true(replace(positive, end=None), true)
    rows = [(positive, (true, false)), (negative, (false,))]
    result = metrics(rows, 0.8, 0.5)
    assert result["term_recall"] == 1
    assert result["strong_precision"] == 1
    assert result["false_hits_on_negative_cases"] == 1
    assert result["mean_timestamp_error_seconds"] == 0.1
    assert metrics([(negative, ())], 1, 1)["term_recall"] is None
    assert metrics([(positive, ())], 1, 1)["strong_precision"] is None
    strong, possible = calibrate(rows)
    assert 0 <= possible <= strong <= 1
    assert metrics(rows, strong, possible)["strong_precision"] >= 0.9
    assert calibrate([]) == (0.25, 0.25)


def test_benchmark_orchestration(monkeypatch, tmp_path, index, glossary):
    import stt_anything.evaluation.runner as runner

    cases = [
        Case(tmp_path / "dev.wav", "dev", "ru", "one", 0.1, 0.4),
        Case(tmp_path / "test.wav", "test", "ru", "one", 0.1, 0.4),
    ]
    monkeypatch.setenv("STT_ANYTHING_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(runner, "generate_cases", lambda *args, **kwargs: cases)
    monkeypatch.setattr(runner, "read_audio", lambda _: np.ones(16000, dtype=np.float32) * 0.1)
    monkeypatch.setattr(runner, "create_index", lambda *args: index)
    monkeypatch.setattr(runner.components, "encoder", lambda *args: FakeEncoder())
    monkeypatch.setattr(runner.components, "lookup", lambda *args: FakeLookup())
    monkeypatch.setattr(runner.components, "source", lambda *args: FakeSource())
    report = benchmark(
        glossary, "ru", ["whisper", "phoneme"], offline=True, index_dir=tmp_path / "indexes"
    )
    assert report["languages"]["ru"]["whisper"]["term_recall"] == 1
    assert Path(report["languages"]["ru"]["phoneme"]["index"]).exists()
    assert set(benchmark(glossary)["languages"]) == {"en", "ru"}
