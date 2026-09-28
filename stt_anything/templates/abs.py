"""Templates may come from TTS, recordings, or any pronunciation source."""

from abc import ABC, abstractmethod

from ..models import FloatArray


class TemplateSource(ABC):
    signature: str

    @abstractmethod
    def synthesize(self, text: str, language: str) -> FloatArray:
        """Return mono float32 audio at 16 kHz."""
        raise NotImplementedError
