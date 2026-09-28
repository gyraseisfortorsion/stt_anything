from __future__ import annotations

import os
from pathlib import Path

VOICE = {"en": "en_US-lessac-low", "ru": "ru_RU-denis-medium"}
VOICE_FILES = {
    "en": "en/en_US/lessac/low/en_US-lessac-low.onnx",
    "ru": "ru/ru_RU/denis/medium/ru_RU-denis-medium.onnx",
}
EVAL_RUSSIAN_VOICE = "ru/ru_RU/dmitri/medium/ru_RU-dmitri-medium.onnx"
PHONEME_MODEL = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"


def cache_root() -> Path:
    return Path(os.environ.get("STT_ANYTHING_CACHE", Path.home() / ".cache" / "stt-anything"))


def download(repo: str, filename: str, root: Path) -> None:
    from huggingface_hub import hf_hub_download

    hf_hub_download(repo, filename, repo_type="model", local_dir=root)


def prepare_voice(language: str, offline: bool = False) -> Path:
    if language not in VOICE_FILES:
        raise ValueError(f"No bundled Piper voice for {language}; provide a custom TemplateSource")
    filename = VOICE_FILES[language]
    root = cache_root() / "voices"
    target = root / filename
    if offline and (not target.exists() or not Path(f"{target}.json").exists()):
        raise FileNotFoundError(f"Missing cached Piper voice {VOICE[language]}; run prepare online")
    if not offline:
        for item in (filename, f"{filename}.json"):
            download("rhasspy/piper-voices", item, root)
    return target


def prepare_eval_russian_voice(offline: bool = False) -> Path:
    root = cache_root() / "voices"
    target = root / EVAL_RUSSIAN_VOICE
    if offline and (not target.exists() or not Path(f"{target}.json").exists()):
        raise FileNotFoundError("Missing cached Russian evaluation voice; run prepare online")
    if not offline:
        for item in (EVAL_RUSSIAN_VOICE, f"{EVAL_RUSSIAN_VOICE}.json"):
            download("rhasspy/piper-voices", item, root)
    return target


def prepare_phoneme(offline: bool = False) -> Path:
    root = cache_root() / "phoneme"
    filenames = (
        "config.json",
        "preprocessor_config.json",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "vocab.json",
        "pytorch_model.bin",
    )
    if offline:
        if not all((root / filename).exists() for filename in filenames):
            raise FileNotFoundError("Missing phoneme model; run prepare --phoneme online")
        return root
    for filename in filenames:
        download(PHONEME_MODEL, filename, root)
    return root


RUSSIAN_CTC_MODEL = "jonatasgrosman/wav2vec2-large-xlsr-53-russian"
RUSSIAN_CTC_FILES = (
    "config.json",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "pytorch_model.bin",
)


def prepare_russian_ctc(offline: bool = False) -> Path:
    root = cache_root() / "russian-ctc"
    if offline:
        if not all((root / filename).exists() for filename in RUSSIAN_CTC_FILES):
            raise FileNotFoundError(
                "Missing Russian CTC model; run prepare --encoder russian-ctc online"
            )
        return root
    for filename in RUSSIAN_CTC_FILES:
        download(RUSSIAN_CTC_MODEL, filename, root)
    return root
