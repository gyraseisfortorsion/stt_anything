import sys
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from conftest import FakeSource

from stt_anything import cache
from stt_anything.encoder import PhonemeEncoder, WhisperEncoder
from stt_anything.templates import PiperTemplateSource


class Tensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def __getitem__(self, key):
        return Tensor(self.value[key])

    def unsqueeze(self, axis):
        return Tensor(np.expand_dims(self.value, axis))

    def float(self):
        return self

    def cpu(self):
        return self

    def to(self, device):
        return self

    def numpy(self):
        return self.value

    def argmax(self, dim):
        return Tensor(self.value.argmax(axis=dim))

    def tolist(self):
        return self.value.tolist()


@pytest.fixture
def mock_models(monkeypatch, tmp_path):
    monkeypatch.setenv("STT_ANYTHING_CACHE", str(tmp_path))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(no_grad=nullcontext))
    model = SimpleNamespace(
        dims=SimpleNamespace(n_mels=80),
        device="cpu",
        encoder=lambda _mel: Tensor(np.arange(6000).reshape(1, 1500, 4)),
    )
    whisper = SimpleNamespace(
        pad_or_trim=lambda a: a,
        log_mel_spectrogram=lambda a, **kw: Tensor(a),
        load_model=Mock(return_value=model),
    )
    monkeypatch.setitem(sys.modules, "whisper", whisper)
    return model, whisper


def test_whisper_lazy_offline_and_chunking(mock_models, tmp_path):
    model, module = mock_models
    encoder = WhisperEncoder(offline=True)
    with pytest.raises(FileNotFoundError):
        encoder.prepare()
    root = tmp_path / "whisper"
    root.mkdir()
    (root / "tiny.pt").touch()
    encoder.prepare()
    encoder.prepare()
    module.load_model.assert_called_once_with(
        str(root / "tiny.pt"), device="cpu", download_root=str(root)
    )
    audio = np.zeros(16000 * 30 + 500, dtype=np.float32)
    sequence = encoder.encode(audio)
    assert sequence.values.shape == (1502, 4)
    assert sequence.spans[1500, 0] == 30
    assert WhisperEncoder(model=model).encode(np.zeros(500, dtype=np.float32)).values.shape == (
        2,
        4,
    )
    for invalid, rate in [(audio, 8000), (np.zeros(0), 16000), (np.zeros((2, 2)), 16000)]:
        with pytest.raises(ValueError):
            encoder.encode(invalid, rate)
    WhisperEncoder().prepare()


def fake_phoneme_modules(monkeypatch, path=(0, 1, 1, 0, 1, 2, 2, 0)):
    tokenizer = SimpleNamespace(pad_token_id=0, unk_token_id=9)

    class Tokenizer:
        pad_token_id = 0
        unk_token_id = 9

        def __call__(self, text, **kwargs):
            return SimpleNamespace(input_ids=[9, 0] if text == "unknown" else [0, 1, 2, 9])

    processor = SimpleNamespace(tokenizer=tokenizer)

    class Processor:
        def __init__(self):
            self.tokenizer = tokenizer

        def __call__(self, audio, **kwargs):
            return SimpleNamespace(input_values=audio)

    # Define outside the class body to avoid local-name shadowing.
    processor = Processor()
    logits = np.eye(10, dtype=np.float32)[list(path)][None, :, :]

    class Model:
        def eval(self):
            return self

        def __call__(self, inputs):
            return SimpleNamespace(logits=Tensor(logits))

    factories = SimpleNamespace(
        Wav2Vec2Processor=SimpleNamespace(from_pretrained=Mock(return_value=processor)),
        Wav2Vec2PhonemeCTCTokenizer=SimpleNamespace(from_pretrained=Mock(return_value=Tokenizer())),
        Wav2Vec2ForCTC=SimpleNamespace(from_pretrained=Mock(return_value=Model())),
    )
    monkeypatch.setitem(sys.modules, "transformers", factories)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(no_grad=nullcontext))
    monkeypatch.setattr("stt_anything.encoder.phoneme.prepare_phoneme", lambda **kw: "cached-model")
    return factories


def test_phoneme_language_and_ctc_spans(monkeypatch):
    factories = fake_phoneme_modules(monkeypatch)
    encoder = PhonemeEncoder("ru", offline=True)
    encoded = encoder.encode_term("омепразол", "ru", FakeSource())
    assert encoded.values.tolist() == [1, 2]
    factories.Wav2Vec2PhonemeCTCTokenizer.from_pretrained.assert_called_once_with(
        "cached-model", phonemizer_lang="ru", local_files_only=True
    )
    sequence = encoder.encode(np.ones(16000, dtype=np.float32))
    assert sequence.values.tolist() == [1, 1, 2]
    np.testing.assert_allclose(sequence.spans, [[0.125, 0.375], [0.5, 0.625], [0.625, 0.875]])
    sequence = encoder.encode(np.ones(20 * 16000 + 500, dtype=np.float32))
    assert sequence.spans[3, 0] > 20
    assert len(encoder.encode(np.ones(20 * 16000 + 100, dtype=np.float32)).values) == 3
    with pytest.raises(ValueError, match="language"):
        encoder.encode_term("x", "en", FakeSource())
    with pytest.raises(ValueError, match="nonempty"):
        encoder.encode_term("unknown", "ru", FakeSource())
    for invalid, rate in [(np.ones(500), 8000), (np.ones((2, 500)), 16000), (np.ones(100), 16000)]:
        with pytest.raises(ValueError):
            encoder.encode(invalid, rate)
    en = PhonemeEncoder()
    en.prepare()
    assert (
        factories.Wav2Vec2PhonemeCTCTokenizer.from_pretrained.call_args.kwargs["phonemizer_lang"]
        == "en-us"
    )


def test_blank_ctc(monkeypatch):
    fake_phoneme_modules(monkeypatch, (0, 0, 0))
    sequence = PhonemeEncoder("ru").encode(np.ones(500, dtype=np.float32))
    assert sequence.values.tolist() == [0]
    assert sequence.spans.tolist()[0] == pytest.approx([0, 500 / 16000])


class Voice:
    def synthesize_wav(self, text, output, **kwargs):
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(np.ones(16000, dtype="<i2").tobytes())


def test_piper_template_cache(monkeypatch):
    factory = Mock(return_value=Voice())
    monkeypatch.setitem(
        sys.modules,
        "piper",
        SimpleNamespace(PiperVoice=SimpleNamespace(load=factory), SynthesisConfig=lambda **kw: kw),
    )
    monkeypatch.setattr("stt_anything.templates.piper.prepare_voice", lambda lang, **kw: lang)
    source = PiperTemplateSource(offline=True)
    assert len(source.synthesize("one", "en")) == 16000
    source.synthesize("two", "en")
    source.synthesize("три", "ru")
    assert factory.call_count == 2


def test_cache_download_and_offline(monkeypatch, tmp_path):
    monkeypatch.setenv("STT_ANYTHING_CACHE", str(tmp_path))
    assert cache.cache_root() == tmp_path
    monkeypatch.delenv("STT_ANYTHING_CACHE")
    assert cache.cache_root().name == "stt-anything"
    monkeypatch.setenv("STT_ANYTHING_CACHE", str(tmp_path))
    remote = Mock()
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=remote))
    cache.download("repo", "file", tmp_path)
    remote.assert_called_once_with("repo", "file", repo_type="model", local_dir=tmp_path)

    def download(repo, filename, root):
        target = root / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()

    monkeypatch.setattr(cache, "download", download)
    with pytest.raises(ValueError, match="custom"):
        cache.prepare_voice("de")
    for language in ("en", "ru"):
        with pytest.raises(FileNotFoundError):
            cache.prepare_voice(language, offline=True)
        result = cache.prepare_voice(language)
        assert cache.prepare_voice(language, offline=True) == result
        result.with_suffix(result.suffix + ".json").unlink()
        with pytest.raises(FileNotFoundError):
            cache.prepare_voice(language, offline=True)
    with pytest.raises(FileNotFoundError):
        cache.prepare_eval_russian_voice(offline=True)
    path = cache.prepare_eval_russian_voice()
    assert cache.prepare_eval_russian_voice(offline=True) == path
    with pytest.raises(FileNotFoundError):
        cache.prepare_phoneme(offline=True)
    root = cache.prepare_phoneme()
    assert cache.prepare_phoneme(offline=True) == root
    (root / "pytorch_model.bin").unlink()
    with pytest.raises(FileNotFoundError):
        cache.prepare_phoneme(offline=True)


def test_whisper_sizes_and_signature(mock_models, tmp_path):
    _, module = mock_models
    for size in ("base", "small"):
        encoder = WhisperEncoder(checkpoint=size)
        encoder.prepare()
        assert module.load_model.call_args.args[0] == size
        assert encoder.name == f"whisper-{size}"
        assert size in encoder.signature
        root = tmp_path / "whisper"
        (root / f"{size}.pt").touch()
        WhisperEncoder(checkpoint=size, offline=True).prepare()
        assert module.load_model.call_args.args[0] == str(root / f"{size}.pt")
    with pytest.raises(ValueError):
        WhisperEncoder(checkpoint="large")


def test_russian_ctc_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("STT_ANYTHING_CACHE", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        cache.prepare_russian_ctc(offline=True)

    def download(repo, filename, root):
        root.mkdir(parents=True, exist_ok=True)
        (root / filename).touch()

    monkeypatch.setattr(cache, "download", download)
    root = cache.prepare_russian_ctc()
    assert cache.prepare_russian_ctc(offline=True) == root
    assert root.name == "russian-ctc"


def test_russian_ctc_tokens_without_transcription(monkeypatch):
    from stt_anything.encoder import RussianCTCEncoder

    factories = fake_phoneme_modules(monkeypatch)
    received = []

    class Tokenizer:
        pad_token_id = 0
        unk_token_id = 9

        def __call__(self, text, **options):
            received.append(text)
            return SimpleNamespace(input_ids=[9] if text == "unknown" else [1, 2])

    factories.Wav2Vec2Processor.from_pretrained.return_value.tokenizer = Tokenizer()
    monkeypatch.setattr(
        "stt_anything.encoder.russian.prepare_russian_ctc", lambda **kw: "cached-russian"
    )
    encoder = RussianCTCEncoder(offline=True)
    term = encoder.encode_term("  Ёж  Препарат  ", "ru", FakeSource())
    assert received[-1] == "ёж препарат"
    assert term.kind == "characters" and term.values.tolist() == [1, 2]
    sequence = encoder.encode(np.ones(16000, dtype=np.float32))
    assert sequence.kind == "characters" and sequence.values.tolist() == [1, 1, 2]
    factories.Wav2Vec2ForCTC.from_pretrained.assert_called_once()
    with pytest.raises(ValueError, match="Russian only"):
        RussianCTCEncoder("en")
    with pytest.raises(ValueError, match="match"):
        encoder.encode_term("x", "en", FakeSource())
    with pytest.raises(ValueError, match="supported"):
        encoder.encode_term("unknown", "ru", FakeSource())
