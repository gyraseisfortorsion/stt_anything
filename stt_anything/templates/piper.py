from __future__ import annotations

import tempfile
import wave
from typing import Any

from ..audio import read_audio
from ..cache import prepare_voice
from ..models import FloatArray
from .abs import TemplateSource


class PiperTemplateSource(TemplateSource):
    signature = "piper:lessac-low:denis-medium:zero-noise:v2"

    def __init__(self, *, offline: bool = False) -> None:
        self.offline = offline
        self.voices: dict[str, Any] = {}

    def synthesize(self, text: str, language: str) -> FloatArray:
        from piper import PiperVoice, SynthesisConfig

        if language not in self.voices:
            self.voices[language] = PiperVoice.load(
                str(prepare_voice(language, offline=self.offline))
            )
        with tempfile.NamedTemporaryFile(suffix=".wav") as output:
            with wave.open(output.name, "wb") as wav_file:
                self.voices[language].synthesize_wav(
                    text, wav_file, syn_config=SynthesisConfig(noise_scale=0, noise_w_scale=0)
                )
            return read_audio(output.name)
