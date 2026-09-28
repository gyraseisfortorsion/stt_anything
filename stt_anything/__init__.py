"""Provider-independent keyword detection with timestamped glossary hits."""

from .api import create_index, run
from .detection import KeywordDetector
from .encoder.abs import Encoder
from .index.abs import IndexBuilder, IndexLookup, IndexStore
from .models import DetectionResult, EncodedSequence, Term, TermIndex
from .templates.abs import TemplateSource

__all__ = [
    "DetectionResult",
    "EncodedSequence",
    "Encoder",
    "IndexBuilder",
    "IndexLookup",
    "IndexStore",
    "KeywordDetector",
    "TemplateSource",
    "Term",
    "TermIndex",
    "create_index",
    "run",
]
