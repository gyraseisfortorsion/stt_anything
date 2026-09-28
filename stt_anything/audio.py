from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from .models import FloatArray

SAMPLE_RATE = 16000


def read_audio(path: str | Path) -> FloatArray:
    command = [
        "ffmpeg",
        "-nostdin",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-f",
        "f32le",
        "-",
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required on PATH") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Could not decode {path}: {exc.stderr.decode(errors='replace')}"
        ) from exc
    audio = np.frombuffer(result.stdout, dtype="<f4").copy()
    if audio.size == 0:
        raise ValueError(f"Audio is empty: {path}")
    return audio


def write_audio(path: str | Path, audio: FloatArray, sample_rate: int = SAMPLE_RATE) -> None:
    import wave

    samples = np.clip(audio, -1, 1)
    pcm = (samples * 32767).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm.tobytes())
