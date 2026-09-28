from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import psutil

from .. import components
from ..api import create_index
from ..audio import read_audio
from ..cache import cache_root
from ..detection import KeywordDetector
from ..index import NpzIndexStore
from .metrics import calibrate, metrics
from .synthetic import generate_cases


def benchmark(
    glossary: str | Path,
    language: str = "both",
    encoders: Sequence[str] = ("whisper", "mfcc"),
    *,
    offline: bool = False,
    index_dir: Path | None = None,
) -> dict[str, Any]:
    languages = ("en", "ru") if language == "both" else (language,)
    report: dict[str, Any] = {
        "note": "Synthetic validation only; human-call accuracy is unmeasured.",
        "languages": {},
    }
    root = index_dir or cache_root() / "indexes-v2"
    process = psutil.Process()
    for lang in languages:
        cases = generate_cases(glossary, lang, offline=offline)
        reports = {}
        for name in encoders:
            model = components.encoder(name, lang, offline)
            index = create_index(glossary, lang, model, components.source("piper", offline))
            detector = KeywordDetector(
                model,
                components.lookup(
                    {"phoneme": "phoneme", "russian-ctc": "characters"}.get(name, "dtw")
                ),
            )
            started = time.perf_counter()
            peak_mb = process.memory_info().rss / (1024**2)
            rows = []
            for case in cases:
                result = detector.detect(
                    read_audio(case.path), index, strong_threshold=1.0, possible_threshold=0.0
                )
                rows.append((case, result.hits))
                peak_mb = max(peak_mb, process.memory_info().rss / (1024**2))
            strong, possible = calibrate([row for row in rows if row[0].split == "dev"])
            index = replace(index, strong_threshold=strong, possible_threshold=possible)
            destination = root / f"{lang}-{name}.npz"
            NpzIndexStore().save(index, destination)
            metric = metrics([row for row in rows if row[0].split == "test"], strong, possible)
            metric.update(
                {
                    "strong_threshold": strong,
                    "possible_threshold": possible,
                    "elapsed_seconds": round(time.perf_counter() - started, 2),
                    "observed_peak_rss_mb": round(peak_mb, 1),
                    "index": str(destination),
                }
            )
            reports[name] = metric
            del detector, model
        report["languages"][lang] = reports
    return report
