"""Synthetic fixtures remain separate from the keyword detection runtime."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import butter, resample, sosfilt

from ..audio import SAMPLE_RATE, read_audio, write_audio
from ..cache import cache_root, prepare_eval_russian_voice
from ..glossary import load_glossary
from ..models import FloatArray


@dataclass(frozen=True)
class Case:
    path: Path
    split: str
    language: str
    term_id: str | None
    start: float | None
    end: float | None


NEGATIVES = {
    "en": [
        "Please check the order again",
        "I need more information today",
        "The doctor called yesterday",
        "The metro form is on the desk",
        "Please repeat the product name",
        "This is a different prescription",
    ],
    "ru": [
        "Пожалуйста проверьте заказ",
        "Мне нужна информация сегодня",
        "Врач звонил вчера",
        "Мы приехали на метро",
        "Повторите название товара",
        "Это другой рецепт",
    ],
}
PREFIX = {"en": "I am asking about", "ru": "Я спрашиваю про"}
SUFFIX = {"en": "today", "ru": "сегодня"}


def _say(text: str, voice: str) -> FloatArray:
    with tempfile.NamedTemporaryFile(suffix=".aiff") as output:
        subprocess.run(["say", "-v", voice, "-o", output.name, text], check=True)
        return read_audio(output.name)


def _piper(text: str, voice: Any) -> FloatArray:
    from piper import SynthesisConfig

    with tempfile.NamedTemporaryFile(suffix=".wav") as output:
        with wave.open(output.name, "wb") as wav_file:
            voice.synthesize_wav(
                text, wav_file, syn_config=SynthesisConfig(noise_scale=0, noise_w_scale=0)
            )
        return read_audio(output.name)


def _modify(audio: FloatArray, condition: str, seed: int) -> FloatArray:
    if condition == "speed":
        return np.asarray(resample(audio, int(len(audio) / 1.12)), dtype=np.float32)
    if condition == "phone":
        filtered = sosfilt(
            butter(4, [300, 3400], btype="bandpass", fs=SAMPLE_RATE, output="sos"), audio
        )
        rng = np.random.default_rng(seed)
        noise = rng.standard_normal(len(audio)) * max(0.001, np.std(filtered) * 0.025)
        return np.asarray(filtered + noise, dtype=np.float32)
    return audio.astype(np.float32)


def generate_cases(glossary: str | Path, language: str, *, offline: bool = False) -> list[Case]:
    terms = load_glossary(glossary, language)
    root = (
        cache_root()
        / "benchmark-v2"
        / hashlib.sha256(Path(glossary).read_bytes()).hexdigest()[:16]
        / language
    )
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "manifest.json"
    if manifest.exists():
        saved = json.loads(manifest.read_text(encoding="utf-8"))
        if len(saved) == 2 * len(terms) + len(NEGATIVES[language]) and all(
            Path(row["path"]).exists() for row in saved
        ):
            return [
                Case(
                    Path(row["path"]),
                    row["split"],
                    row["language"],
                    row["term_id"],
                    row["start"],
                    row["end"],
                )
                for row in saved
            ]
    russian_dev_voice: Any = None
    if language == "ru":
        from piper import PiperVoice

        russian_dev_voice = PiperVoice.load(str(prepare_eval_russian_voice(offline=offline)))

    def speak(text: str, split: str) -> FloatArray:
        if language == "en":
            return _say(text, "Kathy" if split == "dev" else "Samantha")
        return _piper(text, russian_dev_voice) if split == "dev" else _say(text, "Milena")

    cases: list[Case] = []
    gap = np.zeros(1600, dtype=np.float32)
    for idx, term in enumerate(terms):
        for split in ("dev", "test"):
            form = term.display if split == "dev" or not term.spoken_forms else term.spoken_forms[0]
            before = speak(PREFIX[language], split)
            target = speak(form, split)
            after = speak(SUFFIX[language], split)
            start = (len(before) + len(gap)) / SAMPLE_RATE
            end = start + len(target) / SAMPLE_RATE
            combined = np.concatenate((before, gap, target, gap, after))
            condition = ("clean", "speed", "phone")[idx % 3] if split == "test" else "clean"
            combined = _modify(combined, condition, idx)
            if condition == "speed":
                start /= 1.12
                end /= 1.12
            destination = root / f"{split}-positive-{idx:02d}.wav"
            write_audio(destination, combined)
            cases.append(
                Case(destination, split, language, term.id, round(start, 3), round(end, 3))
            )
    for idx, sentence in enumerate(NEGATIVES[language]):
        split = "dev" if idx % 2 == 0 else "test"
        destination = root / f"{split}-negative-{idx:02d}.wav"
        condition = ("clean", "speed", "phone")[idx % 3]
        write_audio(destination, _modify(speak(sentence, split), condition, idx + 200))
        cases.append(Case(destination, split, language, None, None, None))
    manifest.write_text(
        json.dumps(
            [
                {
                    "path": str(case.path),
                    "split": case.split,
                    "language": case.language,
                    "term_id": case.term_id,
                    "start": case.start,
                    "end": case.end,
                }
                for case in cases
            ],
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return cases
