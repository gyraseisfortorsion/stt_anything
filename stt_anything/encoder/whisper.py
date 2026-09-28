"""Whisper is used only as a feature encoder; no decoding or STT is performed."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..audio import SAMPLE_RATE
from ..cache import cache_root
from ..models import EncodedSequence, FloatArray, frame_spans
from .abs import Encoder


class WhisperEncoder(Encoder):
    name = "whisper"
    signature = "whisper:tiny:20ms:v1"

    def __init__(
        self,
        language: str = "",
        *,
        offline: bool = False,
        model: Any = None,
        checkpoint: str = "tiny",
    ) -> None:
        if checkpoint not in ("tiny", "base", "small"):
            raise ValueError("Whisper checkpoint must be tiny, base, or small")
        self.checkpoint = checkpoint
        self.name = "whisper" if checkpoint == "tiny" else f"whisper-{checkpoint}"
        self.signature = f"whisper:{checkpoint}:20ms:v1"
        self.model = model
        self.offline = offline

    def prepare(self) -> None:
        import whisper

        if self.model is None:
            destination = cache_root() / "whisper"
            if self.offline and not (destination / f"{self.checkpoint}.pt").exists():
                raise FileNotFoundError(f"Missing Whisper {self.checkpoint}; run prepare online")
            destination.mkdir(parents=True, exist_ok=True)
            checkpoint = (
                str(destination / f"{self.checkpoint}.pt") if self.offline else self.checkpoint
            )
            self.model = whisper.load_model(
                checkpoint, device="cpu", download_root=str(destination)
            )

    def encode(self, audio: FloatArray, sample_rate: int = SAMPLE_RATE) -> EncodedSequence:
        import torch
        import whisper

        if sample_rate != SAMPLE_RATE or audio.ndim != 1 or len(audio) == 0:
            raise ValueError("Whisper encoder needs nonempty mono 16 kHz audio")
        self.prepare()
        pieces = []
        for start in range(0, len(audio), SAMPLE_RATE * 30):
            chunk = audio[start : start + SAMPLE_RATE * 30]
            padded = whisper.pad_or_trim(chunk)
            mel = whisper.log_mel_spectrogram(padded, n_mels=self.model.dims.n_mels).to(
                self.model.device
            )
            with torch.no_grad():
                frames = self.model.encoder(mel.unsqueeze(0))[0].float().cpu().numpy()
            count = max(1, min(len(frames), int(np.ceil(len(chunk) / 320))))
            pieces.append(frames[:count])
        values = np.concatenate(pieces).astype(np.float32)
        return EncodedSequence(values, frame_spans(len(values), 0.02))
