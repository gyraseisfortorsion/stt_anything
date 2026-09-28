import json
import runpy
import sys
from importlib import import_module
from types import ModuleType
from unittest.mock import Mock

import numpy as np
import pytest
from conftest import FakeEncoder, FakeLookup, FakeSource

from stt_anything import components
from stt_anything.audio import write_audio
from stt_anything.cli import main
from stt_anything.encoder import Encoder, MFCCEncoder, PhonemeEncoder, WhisperEncoder
from stt_anything.index import DTWLookup, NpzIndexStore, PhonemeLookup, TemplateIndexBuilder
from stt_anything.templates import PiperTemplateSource


def test_component_factories(monkeypatch):
    for spec, kind in [
        ("mfcc", MFCCEncoder),
        ("whisper", WhisperEncoder),
        ("phoneme", PhonemeEncoder),
    ]:
        assert isinstance(components.encoder(spec, "ru", True), kind)
    assert isinstance(components.lookup("dtw"), DTWLookup)
    assert isinstance(components.lookup("phoneme"), PhonemeLookup)
    assert isinstance(components.builder("templates"), TemplateIndexBuilder)
    assert isinstance(components.store("npz"), NpzIndexStore)
    assert isinstance(components.source("piper", True), PiperTemplateSource)
    assert isinstance(components.component("builtins:list", list, {}), list)
    with pytest.raises(ValueError, match="Unknown"):
        components.encoder("missing", "ru", True)
    with pytest.raises(TypeError, match="Encoder"):
        components.component("builtins:list", Encoder, {})
    with pytest.raises(ModuleNotFoundError):
        components.encoder("missing_module:factory", "ru", True)


@pytest.fixture
def plugin(monkeypatch):
    module = ModuleType("test_keyword_plugin")
    module.encoder = lambda **kw: FakeEncoder()
    module.source = lambda **kw: FakeSource()
    module.lookup = lambda: FakeLookup()
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module.__name__


def test_cli_create_and_run(glossary, tmp_path, plugin, capsys):
    index_path = tmp_path / "index.npz"
    main(
        [
            "create-index",
            "--glossary",
            str(glossary),
            "--language",
            "ru",
            "--encoder",
            f"{plugin}:encoder",
            "--source",
            f"{plugin}:source",
            "--output",
            str(index_path),
            "--offline",
        ]
    )
    assert json.loads(capsys.readouterr().out)["templates"] == 2
    audio = tmp_path / "audio.wav"
    write_audio(audio, np.ones(16000, dtype=np.float32) * 0.1)
    args = ["run", str(audio), "--index", str(index_path), "--offline"]
    main(args + ["--encoder", f"{plugin}:encoder", "--lookup", f"{plugin}:lookup"])
    result = json.loads(capsys.readouterr().out)
    assert result["hits"][0]["term"] == "омепразол"
    assert "transcript" not in result
    output = tmp_path / "nested" / "result.json"
    main(args + ["--output", str(output)])
    assert output.exists()
    assert capsys.readouterr().out == ""
    assert json.loads(output.read_text())["lookup"] == "dtw"
    main(
        args
        + ["--lookup", f"{plugin}:lookup", "--strong-threshold", "1", "--possible-threshold", "0.3"]
    )
    assert json.loads(capsys.readouterr().out)["hits"][0]["tier"] == "possible"


@pytest.mark.parametrize(
    "encoder,language,evaluation",
    [("whisper", "both", True), ("mfcc", "en", False), ("phoneme", "ru", False)],
)
def test_prepare(monkeypatch, encoder, language, evaluation, capsys):
    whisper = Mock()
    voice = Mock()
    phoneme = Mock()
    eval_voice = Mock()
    module = import_module("stt_anything.cli.main")
    monkeypatch.setattr(module.WhisperEncoder, "prepare", whisper)
    monkeypatch.setattr(module, "prepare_voice", voice)
    monkeypatch.setattr(module, "prepare_phoneme", phoneme)
    monkeypatch.setattr(module, "prepare_eval_russian_voice", eval_voice)
    main(
        ["prepare", "--encoder", encoder, "--language", language]
        + (["--evaluation"] if evaluation else [])
    )
    assert json.loads(capsys.readouterr().out)["encoder"] == encoder
    assert whisper.call_count == int(encoder == "whisper")
    assert phoneme.call_count == int(encoder == "phoneme")
    assert voice.call_count == (2 if language == "both" else 1) * int(encoder != "phoneme")
    assert eval_voice.call_count == int(evaluation)


def test_cli_benchmark(monkeypatch, glossary, capsys):
    benchmark = Mock(return_value={"languages": {}})
    monkeypatch.setattr("stt_anything.evaluation.benchmark", benchmark)
    main(["benchmark", "--glossary", str(glossary), "--language", "ru", "--offline"])
    assert json.loads(capsys.readouterr().out) == {"languages": {}}
    benchmark.assert_called_once()


def test_cli_failures(glossary, capsys):
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "create-index",
                "--glossary",
                str(glossary),
                "--language",
                "ru",
                "--encoder",
                "unknown",
                "--output",
                "unused",
            ]
        )
    assert exc.value.code == 2
    assert "Unknown component" in capsys.readouterr().err
    with pytest.raises(SystemExit) as exc:
        main(["run", "missing.wav"])
    assert exc.value.code == 2


def test_module_entrypoint(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["stt-anything", "--help"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("stt_anything", run_name="__main__")
    assert exc.value.code == 0
    assert "create-index" in capsys.readouterr().out


@pytest.mark.parametrize("name", ["whisper-base", "whisper-small", "russian-ctc"])
def test_prepare_new_encoders(monkeypatch, name, capsys):
    module = import_module("stt_anything.cli.main")
    whisper = Mock()
    russian = Mock()
    monkeypatch.setattr(module.WhisperEncoder, "prepare", whisper)
    monkeypatch.setattr(module, "prepare_russian_ctc", russian)
    monkeypatch.setattr(module, "prepare_voice", Mock())
    main(["prepare", "--encoder", name, "--language", "ru"])
    assert json.loads(capsys.readouterr().out)["encoder"] == name
    assert russian.call_count == int(name == "russian-ctc")
    assert whisper.call_count == int(name.startswith("whisper"))
