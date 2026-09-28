"""Versioned JSON metadata + numeric arrays; never loads Python pickle."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ..models import Array, EncodedSequence, IndexEntry, Term, TermIndex
from .abs import IndexStore


class NpzIndexStore(IndexStore):
    def save(self, index: TermIndex, path: str | Path) -> None:
        metadata = {
            "version": 1,
            "terms": [asdict(term) for term in index.terms],
            "encoder_name": index.encoder_name,
            "encoder_signature": index.encoder_signature,
            "source_signature": index.source_signature,
            "strong_threshold": index.strong_threshold,
            "possible_threshold": index.possible_threshold,
            "entries": [
                {"term_id": entry.term_id, "form": entry.form, "kind": entry.sequence.kind}
                for entry in index.entries
            ],
        }
        arrays: dict[str, Array] = {
            "metadata": np.asarray(json.dumps(metadata, ensure_ascii=False))
        }
        for number, entry in enumerate(index.entries):
            arrays[f"values_{number}"] = entry.sequence.values
            arrays[f"spans_{number}"] = entry.sequence.spans
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as output:
            np.savez_compressed(output, allow_pickle=False, **arrays)

    def load(self, path: str | Path) -> TermIndex:
        try:
            with np.load(path, allow_pickle=False) as data:
                metadata = json.loads(str(data["metadata"]))
                if metadata["version"] != 1:
                    raise ValueError("Unsupported index version; recreate the index")
                terms = tuple(
                    Term(
                        row["id"],
                        row["language"],
                        row["display"],
                        tuple(row["aliases"]),
                        tuple(row["spoken_forms"]),
                    )
                    for row in metadata["terms"]
                )
                entries = tuple(
                    IndexEntry(
                        row["term_id"],
                        row["form"],
                        EncodedSequence(
                            data[f"values_{number}"], data[f"spans_{number}"], row["kind"]
                        ),
                    )
                    for number, row in enumerate(metadata["entries"])
                )
            return TermIndex(
                terms,
                entries,
                metadata["encoder_name"],
                metadata["encoder_signature"],
                metadata["source_signature"],
                metadata["strong_threshold"],
                metadata["possible_threshold"],
            )
        except (KeyError, TypeError) as error:
            raise ValueError("Invalid index metadata; recreate the index") from error
