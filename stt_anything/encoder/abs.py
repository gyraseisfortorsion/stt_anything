"""Extension point for any local or remote acoustic encoder."""

from abc import ABC, abstractmethod

from ..audio import SAMPLE_RATE
from ..models import EncodedSequence, FloatArray
from ..templates.abs import TemplateSource


class Encoder(ABC):
    name: str
    signature: str

    @abstractmethod
    def encode(self, audio: FloatArray, sample_rate: int = SAMPLE_RATE) -> EncodedSequence:
        """Encode audio without generating a transcript."""
        raise NotImplementedError

    def encode_term(self, text: str, language: str, source: TemplateSource) -> EncodedSequence:
        """Override for text-to-token encoders that do not need synthesized audio."""
        return self.encode(source.synthesize(text, language))
