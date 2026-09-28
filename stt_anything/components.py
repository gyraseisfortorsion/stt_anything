"""CLI factories. Custom components use an importable module:factory identifier."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import partial
from importlib import import_module
from typing import Any, TypeVar

from .encoder import Encoder, MFCCEncoder, PhonemeEncoder, RussianCTCEncoder, WhisperEncoder
from .index import (
    CharacterLookup,
    DTWLookup,
    IndexBuilder,
    IndexLookup,
    IndexStore,
    NpzIndexStore,
    PhonemeLookup,
    TemplateIndexBuilder,
)
from .templates import PiperTemplateSource, TemplateSource

T = TypeVar("T")


def component(
    spec: str, expected: type[Any], builtins: Mapping[str, Callable[..., T]], **options: Any
) -> T:
    factory: Callable[..., T]
    if spec in builtins:
        factory = builtins[spec]
    elif ":" in spec:
        module, attribute = spec.split(":", 1)
        factory = getattr(import_module(module), attribute)
    else:
        raise ValueError(f"Unknown component {spec!r}; use a built-in name or module:factory")
    instance = factory(**options)
    if not isinstance(instance, expected):
        raise TypeError(f"{spec} must return an instance of {expected.__name__}")
    return instance


def encoder(spec: str, language: str, offline: bool) -> Encoder:
    factories: dict[str, Callable[..., Encoder]] = {
        "mfcc": MFCCEncoder,
        "whisper": WhisperEncoder,
        "whisper-base": partial(WhisperEncoder, checkpoint="base"),
        "whisper-small": partial(WhisperEncoder, checkpoint="small"),
        "russian-ctc": RussianCTCEncoder,
        "phoneme": PhonemeEncoder,
    }
    return component(
        spec,
        Encoder,
        factories,
        language=language,
        offline=offline,
    )


def lookup(spec: str) -> IndexLookup:
    return component(
        spec,
        IndexLookup,
        {"dtw": DTWLookup, "phoneme": PhonemeLookup, "characters": CharacterLookup},
    )


def builder(spec: str) -> IndexBuilder:
    return component(spec, IndexBuilder, {"templates": TemplateIndexBuilder})


def store(spec: str) -> IndexStore:
    return component(spec, IndexStore, {"npz": NpzIndexStore})


def source(spec: str, offline: bool) -> TemplateSource:
    return component(spec, TemplateSource, {"piper": PiperTemplateSource}, offline=offline)


def demo_lookup(spec: str, top_k: int) -> IndexLookup:
    """Built-in lookup construction with the demo's occurrence limit."""
    return component(
        spec,
        IndexLookup,
        {"dtw": DTWLookup, "phoneme": PhonemeLookup, "characters": CharacterLookup},
        top_k=top_k,
    )
