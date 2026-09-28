from __future__ import annotations

import numpy as np
from scipy.fft import dct
from scipy.signal import stft

from ..audio import SAMPLE_RATE
from ..models import EncodedSequence, FloatArray, frame_spans
from .abs import Encoder


def mfcc_features(audio: FloatArray) -> tuple[FloatArray, float]:
    """13 MFCCs at a 20 ms frame step; no learned model or download."""
    frame_step = 320
    _, _, spectrum = stft(
        audio.astype(np.float32),
        fs=SAMPLE_RATE,
        nperseg=400,
        noverlap=80,
        nfft=512,
        boundary="zeros",
        padded=True,
    )
    power = np.abs(spectrum).T ** 2
    mel_points = np.linspace(0, 2595 * np.log10(1 + 8000 / 700), 42)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    bins = np.floor((513 * hz_points) / SAMPLE_RATE).astype(int)
    bank = np.zeros((40, power.shape[1]), dtype=np.float32)
    for idx in range(40):
        left, center, right = bins[idx : idx + 3]
        bank[idx, left:center] = np.linspace(0, 1, center - left, endpoint=False)
        bank[idx, center:right] = np.linspace(1, 0, right - center, endpoint=False)
    logged = np.log(np.maximum(power @ bank.T, 1e-10))
    coefficients = np.asarray(dct(logged, type=2, axis=1, norm="ortho")[:, :13], dtype=np.float32)
    coefficients[:, 0] = 0  # energy changes more than phonetic identity
    return coefficients, frame_step / SAMPLE_RATE


class MFCCEncoder(Encoder):
    name = "mfcc"
    signature = "mfcc:13:20ms:v1"

    def __init__(self, language: str = "", *, offline: bool = False) -> None:
        pass

    def encode(self, audio: FloatArray, sample_rate: int = SAMPLE_RATE) -> EncodedSequence:
        if sample_rate != SAMPLE_RATE or audio.ndim != 1 or len(audio) < 400:
            raise ValueError("MFCC needs mono 16 kHz audio with at least 400 samples")
        values, step = mfcc_features(audio)
        return EncodedSequence(values, frame_spans(len(values), step))
