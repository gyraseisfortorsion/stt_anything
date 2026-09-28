"""Independent extension points for creation, persistence, and search."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path

from ..encoder.abs import Encoder
from ..models import Candidate, EncodedSequence, Term, TermIndex
from ..templates.abs import TemplateSource


class IndexBuilder(ABC):
    @abstractmethod
    def create(self, terms: Sequence[Term], encoder: Encoder, source: TemplateSource) -> TermIndex:
        raise NotImplementedError


class IndexStore(ABC):
    @abstractmethod
    def save(self, index: TermIndex, path: str | Path) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str | Path) -> TermIndex:
        raise NotImplementedError


class IndexLookup(ABC):
    name: str

    @abstractmethod
    def search(self, index: TermIndex, query: EncodedSequence) -> list[Candidate]:
        """Return candidates in seconds with match scores between zero and one."""
        raise NotImplementedError
