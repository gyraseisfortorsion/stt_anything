"""Optional CTC phoneme features and text phonemization, without transcription."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..audio import SAMPLE_RATE
from ..cache import prepare_phoneme
from ..models import EncodedSequence, FloatArray, frame_spans
from ..templates.abs import TemplateSource
from .abs import Encoder


class PhonemeEncoder(Encoder):
    name = "phoneme"

    def __init__(self, language: str = "en", *, offline: bool = False) -> None:
        self.language = language
        self.offline = offline
        self.signature = f"xlsr-espeak-cv:phones:v1:{language}"
        self.processor: Any = None
        self.model: Any = None

    def prepare(self) -> None:
        if self.model is None:
            from transformers import (
                Wav2Vec2ForCTC,
                Wav2Vec2PhonemeCTCTokenizer,
                Wav2Vec2Processor,
            )

            root = prepare_phoneme(offline=self.offline)
            self.processor = Wav2Vec2Processor.from_pretrained(str(root), local_files_only=True)
            self.processor.tokenizer = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(
                str(root),
                phonemizer_lang="en-us" if self.language == "en" else self.language,
                local_files_only=True,
            )
            self.model = Wav2Vec2ForCTC.from_pretrained(str(root), local_files_only=True)
            self.model.eval()

    def encode_term(self, text: str, language: str, source: TemplateSource) -> EncodedSequence:
        if language != self.language:
            raise ValueError("Phoneme encoder language must match the index language")
        self.prepare()
        tokenizer = self.processor.tokenizer
        ids = [
            token
            for token in tokenizer(text, add_special_tokens=False).input_ids
            if token not in (tokenizer.unk_token_id, tokenizer.pad_token_id)
        ]
        return EncodedSequence(
            np.asarray(ids, dtype=np.int64), frame_spans(len(ids), 0.02), "phones"
        )

    def encode(self, audio: FloatArray, sample_rate: int = SAMPLE_RATE) -> EncodedSequence:
        import torch

        if sample_rate != SAMPLE_RATE or audio.ndim != 1 or len(audio) < 400:
            raise ValueError("Phoneme encoder needs mono 16 kHz audio with at least 400 samples")
        self.prepare()
        phones: list[int] = []
        spans: list[tuple[float, float]] = []
        # Chunking bounds memory; a term spanning a chunk boundary may be missed.
        for start in range(0, len(audio), SAMPLE_RATE * 20):
            chunk = audio[start : start + SAMPLE_RATE * 20]
            if len(chunk) < 400:
                continue
            inputs = self.processor(chunk, sampling_rate=SAMPLE_RATE, return_tensors="pt")
            with torch.no_grad():
                logits = self.model(inputs.input_values).logits[0]
            path = logits.argmax(dim=-1).cpu().tolist()
            blank = self.processor.tokenizer.pad_token_id
            previous = blank
            step = len(chunk) / SAMPLE_RATE / len(path)
            offset = start / SAMPLE_RATE
            for idx, token in enumerate(path):
                if token != blank and token != previous:
                    phones.append(token)
                    spans.append((offset + idx * step, offset + (idx + 1) * step))
                elif token != blank and token == previous:
                    spans[-1] = (spans[-1][0], offset + (idx + 1) * step)
                previous = token
        if not phones:
            # A blank token cannot match a nonblank glossary pronunciation.
            phones = [self.processor.tokenizer.pad_token_id]
            spans = [(0.0, len(audio) / SAMPLE_RATE)]
        return EncodedSequence(
            np.asarray(phones, dtype=np.int64), np.asarray(spans, dtype=np.float32), "phones"
        )
