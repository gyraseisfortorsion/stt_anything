"""Serialized local model jobs, persistent indexes, and session-only recordings."""

from __future__ import annotations

import gc
import importlib.util
import json
import re
import shutil
import tempfile
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

from stt_anything import components
from stt_anything.audio import SAMPLE_RATE, read_audio, write_audio
from stt_anything.cache import VOICE_FILES, cache_root
from stt_anything.detection import KeywordDetector
from stt_anything.encoder.abs import Encoder
from stt_anything.glossary import load_glossary
from stt_anything.index import NpzIndexStore, TemplateIndexBuilder
from stt_anything.index.builder import default_thresholds
from stt_anything.models import EncodedSequence, FloatArray, Term, validate_thresholds
from stt_anything.templates.abs import TemplateSource

from .diagnostics import ObservedLookup, diagnostics

MAX_UPLOAD = 32 * 1024 * 1024
MAX_DURATION = 120
STATIC = Path(__file__).parent / "static"
SAMPLE = Path(__file__).parents[1] / "examples" / "pharma.jsonl"


class APIError(ValueError):
    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


def identifier(value: str) -> str:
    if len(value) != 32 or any(char not in "0123456789abcdef" for char in value):
        raise APIError("Unknown resource", 404)
    return value


def terms_from_request(payload: dict[str, Any]) -> list[Term]:
    language = payload.get("language")
    rows = payload.get("terms")
    if language not in ("en", "ru") or not isinstance(rows, list) or not 1 <= len(rows) <= 100:
        raise APIError("Choose English or Russian and provide 1–100 terms.")
    canonical = []
    for number, row in enumerate(rows):
        if not isinstance(row, dict) or row.get("language", language) != language:
            raise APIError("Use one language per index.")
        canonical.append(
            {
                "id": f"{language}-{number}",
                "language": language,
                "display": row.get("display"),
                "aliases": row.get("aliases", []),
                "spoken_forms": row.get("spoken_forms", []),
            }
        )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", encoding="utf-8") as glossary:
        glossary.write("\n".join(json.dumps(row, ensure_ascii=False) for row in canonical))
        glossary.flush()
        terms = load_glossary(glossary.name, language)
    if language == "en" and any(
        re.search("[А-Яа-яЁё]", form) for term in terms for form in term.forms
    ):
        raise APIError(
            "Russian spellings need a Russian index. Choose Russian, or import English spellings separately."
        )
    names = [term.display.casefold() for term in terms]
    if len(set(names)) != len(names):
        raise APIError("Duplicate names: combine their aliases into one row.")
    if sum(len(term.forms) for term in terms) > 300 or any(
        len(form) > 150 for term in terms for form in term.forms
    ):
        raise APIError("Limit the dictionary to 300 pronunciations and 150 characters per form.")
    return terms


class ProgressEncoder(Encoder):
    """Observe the public encoder contract without duplicating index creation."""

    def __init__(self, encoder: Encoder, update: Callable[[str], None]) -> None:
        self.delegate = encoder
        self.name = encoder.name
        self.signature = encoder.signature
        self.update = update

    def encode(self, audio: FloatArray, sample_rate: int = SAMPLE_RATE) -> EncodedSequence:
        return self.delegate.encode(audio, sample_rate)

    def encode_term(self, text: str, language: str, source: TemplateSource) -> EncodedSequence:
        sequence = self.delegate.encode_term(text, language, source)
        self.update(text)
        return sequence


def waveform(audio: FloatArray, bins: int = 1200) -> list[list[float]]:
    chunks = np.array_split(audio, min(bins, len(audio)))
    return [[round(float(chunk.min()), 4), round(float(chunk.max()), 4)] for chunk in chunks]


class DemoApp:
    def __init__(self, data_dir: Path | None = None) -> None:
        self.root = data_dir or cache_root() / "demo"
        self.root.mkdir(parents=True, exist_ok=True)
        self.session = tempfile.TemporaryDirectory(prefix="stt-anything-demo-")
        self.jobs: dict[str, dict[str, Any]] = {}
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="keyword-demo")
        self.cached_encoder: Encoder | None = None
        self.encoder_key: tuple[str, str, bool] | None = None

    def close(self) -> None:
        self.executor.shutdown(wait=True)
        self.cached_encoder = None
        self.session.cleanup()

    def indexes(self) -> list[dict[str, Any]]:
        result = []
        for path in self.root.glob("*.json"):
            try:
                row = json.loads(path.read_text(encoding="utf-8"))
                identifier(row["id"])
                float(row["created_at"])
                if (self.root / f"{row['id']}.npz").exists():
                    result.append(row)
            except (OSError, ValueError, KeyError, TypeError):
                continue
        return sorted(result, key=lambda row: row["created_at"], reverse=True)

    def index_info(self, index_id: str) -> dict[str, Any]:
        identifier(index_id)
        for row in self.indexes():
            if row["id"] == index_id:
                return row
        raise APIError("Index not found. Build or select an index first.", 404)

    def config(self) -> dict[str, Any]:
        def installed(name: str) -> bool:
            return importlib.util.find_spec(name) is not None

        voices = {
            language: (installed("piper") and (cache_root() / "voices" / filename).exists())
            for language, filename in VOICE_FILES.items()
        }
        return {
            "encoders": [
                {
                    "id": "whisper",
                    "name": "Whisper tiny",
                    "hint": "Recommended starting point · small local model",
                    "installed": installed("whisper"),
                    "cached": (cache_root() / "whisper" / "tiny.pt").exists(),
                    "lookup": "dtw",
                    "needs_voice": True,
                },
                *[
                    {
                        "id": f"whisper-{size}",
                        "name": f"Whisper {size}",
                        "hint": f"Multilingual {size} acoustic encoder · larger than tiny; compare on your recordings",
                        "installed": installed("whisper"),
                        "cached": (cache_root() / "whisper" / f"{size}.pt").exists(),
                        "lookup": "dtw",
                        "needs_voice": True,
                    }
                    for size in ("base", "small")
                ],
                {
                    "id": "russian-ctc",
                    "name": "Russian CTC · XLSR",
                    "hint": "Russian-trained character encoder · larger model, no TTS pronunciation templates",
                    "installed": installed("transformers") and installed("torch"),
                    "cached": (cache_root() / "russian-ctc" / "pytorch_model.bin").exists(),
                    "lookup": "characters",
                    "needs_voice": False,
                    "languages": ["ru"],
                },
                {
                    "id": "mfcc",
                    "name": "MFCC",
                    "hint": "Experimental speed baseline · unreliable across speakers; prefer Whisper or phoneme for recordings",
                    "installed": True,
                    "cached": True,
                    "lookup": "dtw",
                    "needs_voice": True,
                },
                {
                    "id": "phoneme",
                    "name": "Phoneme CTC",
                    "hint": "Multilingual phoneme comparison · larger model; can confuse Russian words",
                    "installed": installed("transformers")
                    and installed("phonemizer")
                    and bool(shutil.which("espeak-ng")),
                    "cached": (cache_root() / "phoneme" / "pytorch_model.bin").exists(),
                    "lookup": "phoneme",
                    "needs_voice": False,
                },
            ],
            "voice_cached": voices,
            "samples": [asdict(term) for term in load_glossary(SAMPLE)],
            "indexes": self.indexes(),
            "max_duration": MAX_DURATION,
            "thresholds": {
                language: {
                    name: default_thresholds(name, language)
                    for name in (
                        "whisper",
                        "whisper-base",
                        "whisper-small",
                        "russian-ctc",
                        "mfcc",
                        "phoneme",
                    )
                }
                for language in ("en", "ru")
            },
        }

    def encoder(self, name: str, language: str, offline: bool) -> Encoder:
        key = (name, language, offline)
        if self.encoder_key != key:
            self.cached_encoder = None
            gc.collect()
            self.cached_encoder = components.encoder(name, language, offline)
            self.encoder_key = key
        assert self.cached_encoder is not None
        return self.cached_encoder

    def update(self, job_id: str, **fields: Any) -> None:
        with self.lock:
            self.jobs[job_id].update(fields)

    def job(self, job_id: str) -> dict[str, Any]:
        identifier(job_id)
        with self.lock:
            if job_id not in self.jobs:
                raise APIError("Job not found", 404)
            return dict(self.jobs[job_id])

    def submit(self, operation: Callable[[str], dict[str, Any]]) -> dict[str, str]:
        job_id = uuid.uuid4().hex
        with self.lock:
            self.jobs[job_id] = {
                "id": job_id,
                "state": "queued",
                "progress": 0,
                "message": "Waiting for the local worker…",
            }

        def work() -> None:
            self.update(job_id, state="running", progress=1, message="Preparing local components…")
            try:
                result = operation(job_id)
                self.update(job_id, state="complete", progress=100, message="Done", result=result)
            except Exception as error:
                self.update(job_id, state="error", message=str(error))

        self.executor.submit(work)
        return {"job_id": job_id}

    def create_index(self, payload: dict[str, Any]) -> dict[str, str]:
        terms = terms_from_request(payload)
        name = payload.get("name", "My dictionary")
        model = payload.get("encoder", "whisper")
        top_k = payload.get("top_k", 3)
        if not isinstance(name, str) or not 1 <= len(name.strip()) <= 80:
            raise APIError("Give the dictionary a name of 1–80 characters.")
        if model not in (
            "whisper",
            "whisper-base",
            "whisper-small",
            "russian-ctc",
            "mfcc",
            "phoneme",
        ) or top_k not in (3, 10, 25):
            raise APIError("Choose a supported encoder and occurrence limit.")
        language = terms[0].language
        if model == "russian-ctc" and language != "ru":
            raise APIError("Russian CTC needs a Russian dictionary.")
        strong, possible = default_thresholds(model, language)
        strong = float(payload.get("strong_threshold", strong))
        possible = float(payload.get("possible_threshold", possible))
        validate_thresholds(strong, possible)
        offline = not bool(payload.get("allow_downloads", False))
        total = sum(len(term.forms) for term in terms)

        def build(job_id: str) -> dict[str, Any]:
            base = self.encoder(model, language, offline)
            completed = 0

            def progress(form: str) -> None:
                nonlocal completed
                completed += 1
                self.update(
                    job_id,
                    progress=round(95 * completed / total),
                    completed=completed,
                    total=total,
                    message=f"Encoded {form} · {completed}/{total} pronunciations",
                )

            self.update(
                job_id,
                total=total,
                completed=0,
                message="Loading encoder and pronunciation source…",
            )
            index = TemplateIndexBuilder().create(
                terms, ProgressEncoder(base, progress), components.source("piper", offline)
            )
            index = replace(index, strong_threshold=strong, possible_threshold=possible)
            index_id = uuid.uuid4().hex
            NpzIndexStore().save(index, self.root / f"{index_id}.npz")
            metadata = {
                "id": index_id,
                "name": name.strip(),
                "language": language,
                "encoder": model,
                "lookup": {"phoneme": "phoneme", "russian-ctc": "characters"}.get(model, "dtw"),
                "top_k": top_k,
                "strong_threshold": strong,
                "possible_threshold": possible,
                "terms": [asdict(term) for term in terms],
                "templates": total,
                "created_at": time.time(),
                "offline": offline,
            }
            (self.root / f"{index_id}.json").write_text(
                json.dumps(metadata, ensure_ascii=False), encoding="utf-8"
            )
            return metadata

        return self.submit(build)

    def analyze(self, index_id: str, content: bytes) -> dict[str, str]:
        info = self.index_info(index_id)
        if not content or len(content) > MAX_UPLOAD:
            raise APIError("Upload nonempty audio up to 32 MB.", 413)

        def process(job_id: str) -> dict[str, Any]:
            started = time.perf_counter()
            folder = Path(self.session.name) / job_id
            folder.mkdir()
            source = folder / "input.audio"
            source.write_bytes(content)
            self.update(job_id, progress=10, message="Decoding audio…")
            audio = read_audio(source)
            if len(audio) / SAMPLE_RATE > MAX_DURATION:
                raise APIError("Keep recordings to two minutes or less.")
            write_audio(folder / "audio.wav", audio)
            source.unlink()
            index = NpzIndexStore().load(self.root / f"{index_id}.npz")
            model = self.encoder(info["encoder"], info["language"], info["offline"])
            search = components.demo_lookup(info["lookup"], info["top_k"])
            self.update(
                job_id, progress=25, message="Encoding audio and looking for dictionary terms…"
            )
            observed = ObservedLookup(search)
            result = KeywordDetector(model, observed).detect(audio, index).to_dict()
            result.update(
                {
                    "diagnostics": diagnostics(
                        audio, observed.candidates, index.possible_threshold
                    ),
                    "index_id": index_id,
                    "dictionary_name": info["name"],
                    "audio_url": f"/api/audio/{job_id}",
                    "peaks": waveform(audio),
                    "runtime_seconds": round(time.perf_counter() - started, 2),
                    "strong_threshold": index.strong_threshold,
                    "possible_threshold": index.possible_threshold,
                }
            )
            return result

        return self.submit(process)

    def audio(self, job_id: str) -> bytes:
        job = self.job(job_id)
        path = Path(self.session.name) / job_id / "audio.wav"
        if job["state"] != "complete" or not path.exists():
            raise APIError("Audio not found", 404)
        return path.read_bytes()
