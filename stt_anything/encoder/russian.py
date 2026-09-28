"""Russian-trained CTC character features; no transcription decoding or language model."""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from ..audio import SAMPLE_RATE
from ..cache import prepare_russian_ctc
from ..models import EncodedSequence, FloatArray, frame_spans
from ..templates.abs import TemplateSource
from .phoneme import PhonemeEncoder


class RussianCTCEncoder(PhonemeEncoder):
    name = "russian-ctc"

    def __init__(self, language: str = "ru", *, offline: bool = False) -> None:
        if language != "ru":
            raise ValueError(
                "Russian CTC supports Russian only; choose another encoder for English"
            )
        super().__init__(language, offline=offline)
        self.signature = "xlsr53-russian:characters:v1"

    def prepare(self) -> None:
        if self.model is None:
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

            root = prepare_russian_ctc(offline=self.offline)
            self.processor = Wav2Vec2Processor.from_pretrained(str(root), local_files_only=True)
            self.model = Wav2Vec2ForCTC.from_pretrained(str(root), local_files_only=True)
            self.model.eval()

    def encode_term(self, text: str, language: str, source: TemplateSource) -> EncodedSequence:
        if language != self.language:
            raise ValueError("Russian CTC encoder language must match the index language")
        self.prepare()
        text = re.sub(r"\s+", " ", text.lower()).strip()
        tokenizer: Any = self.processor.tokenizer
        ids = tokenizer(text, add_special_tokens=False).input_ids
        if tokenizer.unk_token_id in ids:
            raise ValueError(
                "Russian CTC glossary forms must use supported Russian letters and spaces"
            )
        return EncodedSequence(
            np.asarray(ids, dtype=np.int64), frame_spans(len(ids), 0.02), "characters"
        )

    def encode(self, audio: FloatArray, sample_rate: int = SAMPLE_RATE) -> EncodedSequence:
        sequence = super().encode(audio, sample_rate)
        return EncodedSequence(sequence.values, sequence.spans, "characters")
