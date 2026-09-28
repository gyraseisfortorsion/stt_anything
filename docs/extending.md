# Extending the toolkit

The detection engine never invokes a transcription API. Use a speech encoder, a signal-processing encoder, a remote embedding service, or an existing model's internal representations. A provider-specific adapter stays behind `Encoder`.

## Interfaces

| Interface | Methods | Contract |
| --- | --- | --- |
| `encoder.abs.Encoder` | `encode(audio, sample_rate)`, `encode_term(text, language, source)` | `EncodedSequence` with values and explicit time spans; name and a configuration signature |
| `templates.abs.TemplateSource` | `synthesize(text, language)` | Mono 16 kHz float32 pronunciation audio; source signature |
| `index.abs.IndexBuilder` | `create(terms, encoder, source)` | A `TermIndex` with glossary metadata and index entries |
| `index.abs.IndexStore` | `save(index, path)`, `load(path)` | Persist/reconstruct the chosen index representation |
| `index.abs.IndexLookup` | `search(index, query)` | `Candidate` objects containing term IDs, start/end seconds, and normalized match scores |

`EncodedSequence.values` can contain frame vectors (N x D) or token IDs (N). `spans` is N x 2 float32 with ordered, nonnegative start/end times. `kind` identifies the representation. The built-in DTW lookup accepts `frames`; the phoneme lookup accepts `phones`. A custom lookup can handle a different kind. Timestamps must remain relative to the start of the complete recording, including encoder chunk offsets.

The default `Encoder.encode_term` synthesizes a pronunciation and calls `encode`. Override it for text-to-token or text-to-embedding models. The phoneme adapter does this and never calls the supplied pronunciation source. The default builder records the configured source's signature even when an encoder skips synthesis.

The NPZ store handles numeric representations without pickle. For an ANN index or external database, implement a builder and store together, and a lookup compatible with that representation. You can subclass `TermIndex` for additional index state and inject those components directly in Python. Preserve `terms`, `entries`, and `encoder_signature` when using the convenience API or built-in detector. A custom encoder supplied to `run` must have the same signature used during creation.

## Example: a custom lookup

Create an importable `my_adapters.py`:

```python
from stt_anything.index import DTWLookup, IndexLookup
from stt_anything.models import Candidate, EncodedSequence, TermIndex

class MoreOccurrences(IndexLookup):
    name = "more-occurrences"

    def __init__(self) -> None:
        self.searcher = DTWLookup(top_k=10)

    def search(self, index: TermIndex, query: EncodedSequence) -> list[Candidate]:
        return self.searcher.search(index, query)
```

Supply it as a Python object:

```python
from my_adapters import MoreOccurrences
from stt_anything import run

result = run("recording.wav", "indexes/pharma-ru.npz", lookup=MoreOccurrences())
```

Or select it through the CLI:

```sh
uv run stt-anything run recording.wav --index indexes/pharma-ru.npz \
  --lookup my_adapters:MoreOccurrences --offline
```

The module must be importable in the active Python environment. A `module:factory` may reference a class or a function returning a component instance. CLI factories receive these keyword arguments:

| Selector | Factory arguments |
| --- | --- |
| `--encoder` | `language: str`, `offline: bool` |
| `--source` | `offline: bool` |
| `--builder`, `--lookup`, `--store` | No arguments |

`create-index` accepts encoder, builder, source, and store selectors. `run` accepts encoder, lookup, and store selectors. The encoder factory identifier used by the CLI is saved in the index so future CLI runs can recreate it. Programmatic APIs accept objects directly and impose no factory signature.

## Compatibility and score semantics

Change the encoder signature when its model, layer, tokenizer, normalization, dimensions, or chunking behavior changes. Model identity alone is insufficient. A signature mismatch fails before encoding query audio.

Lookup scores must be finite values from 0 to 1, with larger values indicating better matches. If a retrieval system returns distances or logits, convert them to that scale in its adapter. The scores are not probabilities. Calibrate thresholds using held-out development audio for the domain and recording conditions. The engine applies threshold filtering, rejects low-energy spans, suppresses duplicate overlapping hits for the same term, and allows only the best overlapping term to be strong.

Default DTW and phone lookups return up to three candidates per pronunciation template. Use a custom configuration such as the example above for recordings with more occurrences. Large glossaries may need an ANN candidate shortlist followed by temporal matching; the current exhaustive lookup is intended as a small baseline.
