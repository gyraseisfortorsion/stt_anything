# stt-anything

A modular **acoustic keyword spotting** library and CLI. Give it a glossary and an audio recording; it returns glossary matches, estimated start/end times, match scores, and **strong** or **possible** tiers. The bundled examples are English and Russian pharma terms.

**No transcription step is performed.** Whisper is one optional acoustic encoder, and its decoder is never called. MFCC needs no speech model; the optional phoneme encoder emits phone tokens. There is no dependency on an STT service or transcription API. No OpenAI or ElevenLabs keys are needed for the bundled adapters.

## Web demo

Try the local dictionary editor, microphone recorder, and clickable keyword waveform:

```sh
uv run python -m demo.server
```

Open http://localhost:8765. See [demo setup and usage](demo/README.md) for model
installation, dictionary imports, encoder options, and offline operation.

## Quick start

Requires Python 3.11 or 3.12, `uv`, and `ffmpeg` on `PATH`.

```sh
uv sync --extra local
uv run stt-anything prepare --encoder whisper --language ru
uv run stt-anything create-index --glossary examples/pharma.jsonl --language ru \
  --encoder whisper --offline --output indexes/pharma-ru.npz
uv run stt-anything run recording.ogg --index indexes/pharma-ru.npz --offline
uv run stt-anything run recording.ogg --index indexes/pharma-ru.npz --offline \
  --output results.json
```

`run` accepts any format that FFmpeg can decode. Audio is converted to mono float32 at 16 kHz. Without `--output`, the CLI prints JSON to the terminal. An index includes its glossary, encoder configuration signature, pronunciation forms, feature arrays, and default thresholds. You do not need the original glossary to run an existing index.

Use `--encoder mfcc` during index creation for the small baseline. To use the larger phoneme adapter:

```sh
uv sync --extra phoneme
brew install espeak-ng  # macOS; install the equivalent system package on Linux
uv run stt-anything prepare --encoder phoneme --language ru
uv run stt-anything create-index --glossary examples/pharma.jsonl --language ru \
  --encoder phoneme --offline --output indexes/phones-ru.npz
uv run stt-anything run recording.ogg --index indexes/phones-ru.npz --offline
```

The phoneme adapter creates term tokens directly from text, so index creation needs no TTS voice. Install `--extra piper` for Piper templates and `--extra whisper` for Whisper independently, or `--extra local` for both. The base package includes only numeric audio processing and matching dependencies. Downloads are cached in `~/.cache/stt-anything`; set `STT_ANYTHING_CACHE` to change that location. `--offline` prevents bundled adapters from downloading weights or voices. Initial `prepare` needs internet access.

## More encoder options

Whisper `tiny` remains `--encoder whisper`. Select `whisper-base` or
`whisper-small` for larger multilingual feature encoders, using the same matching
pipeline. Prepare each checkpoint before offline use.

For Russian-specific CTC character features:

```sh
uv sync --extra russian
uv run stt-anything prepare --encoder russian-ctc --language ru
uv run stt-anything create-index --glossary examples/pharma.jsonl --language ru \
  --encoder russian-ctc --offline --output indexes/russian-ctc.npz
uv run stt-anything run recording.wav --index indexes/russian-ctc.npz --offline
```

The Russian checkpoint is [XLSR Russian](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-russian).
It emits timestamped character tokens; the toolkit does not decode a transcript
or load a language model. It creates dictionary tokens from spelling, so Piper
and eSpeak are unnecessary for this adapter. Its lookup checks word boundaries
and rejects paths spanning long pauses. Larger models do not guarantee better
keyword accuracy: evaluate your glossary and recordings, and calibrate thresholds.
See [the noisy Russian experiment](docs/noisy-russian.md) for measured limitations.

## Glossary and output

UTF-8 JSONL, one term per line:

```json
{"id":"ru-omeprazole","language":"ru","display":"омепразол","aliases":[],"spoken_forms":[]}
```

IDs must be unique. Aliases and spoken forms produce additional templates. The core accepts arbitrary language codes; bundled Piper voices cover English and Russian. For another language or pronunciation source, supply a custom template adapter.

```json
{
  "language": "ru",
  "duration": 19.52,
  "encoder": "phoneme",
  "lookup": "phoneme",
  "hits": [
    {"term_id":"ru-metformin","term":"метформин","start":12.192,"end":12.853,"score":0.6667,"tier":"strong"}
  ],
  "score_note": "Match scores are ranking scores, not probabilities; timestamps are estimates."
}
```

Scores are specific to the lookup algorithm. They cannot be compared as probabilities across encoders. Strong is a threshold tier, not a guarantee of correctness. Overlapping weaker alternatives stay possible; repeated occurrences of a term can have separate hits. `--strong-threshold` and `--possible-threshold` override index thresholds, with `0 <= possible <= strong <= 1`.

## Python API

```python
from stt_anything import create_index, run
from stt_anything.encoder import MFCCEncoder
from stt_anything.index import DTWLookup, NpzIndexStore
from stt_anything.templates import PiperTemplateSource

encoder = MFCCEncoder()
index = create_index(
    "examples/pharma.jsonl", "ru", encoder,
    PiperTemplateSource(offline=True),
)
NpzIndexStore().save(index, "indexes/pharma-ru.npz")
result = run("recording.ogg", index, encoder=encoder, lookup=DTWLookup())
print(result.to_dict())
```

For already decoded audio, use `KeywordDetector(encoder, lookup).detect(audio, index)` with a mono 16 kHz float32 array. All components can be injected as Python objects. See [the extension guide](docs/extending.md) for custom encoder, builder, store, source, and lookup adapters, including CLI plugin factories.

## Layout

```text
stt_anything/
  cli/                 argparse commands and JSON output
  encoder/             Encoder ABC, MFCC, Whisper, phoneme/Russian CTC adapters
    abs.py
  index/               IndexBuilder, IndexStore, IndexLookup ABCs
    abs.py             template creation, safe NPZ persistence, DTW/token search
  templates/           TemplateSource ABC and optional Piper adapter
  evaluation/          synthetic fixtures, calibration, held-out metrics
  api.py               create_index and run convenience functions
  detection.py         component orchestration, filtering, tier assignment
  models.py            provider-independent data contracts
  audio.py             FFmpeg conversion
  glossary.py          UTF-8 glossary validation
  cache.py             optional model/voice preparation
 tests/                offline unit and contract tests
```

NPZ indexes contain versioned JSON metadata and numeric arrays and load with `allow_pickle=False`. Encoder signatures prevent accidental use of an incompatible index. Model weights, generated artifacts, and personal recordings are ignored by Git.

## Quality checks

```sh
uv sync --extra dev
make check
```

The checks run Ruff linting, Ruff formatting verification, strict mypy, and pytest. Pytest enforces **100% line and branch coverage across all package modules**, including CLI and optional adapter code, without coverage exclusions. Adapter unit tests use local fakes and never download weights. Numba JIT is disabled inside the unit tests so coverage measures the matching implementation; normal CLI runs use JIT. Model inference quality is assessed separately with real local model runs. GitHub Actions runs these checks and a wheel installation smoke test on Python 3.11 and 3.12 for pushes and pull requests. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Synthetic evaluation

```sh
uv sync --extra local --extra evaluation
uv run stt-anything prepare --encoder whisper --evaluation
uv run stt-anything benchmark --glossary examples/pharma.jsonl --language both \
  --encoders whisper mfcc --offline --index-dir indexes/calibrated \
  --output reports/benchmark.json
```

Add `--extra phoneme` and prepare that encoder to include it in `--encoders`. Synthetic fixtures use separate voices for templates, development, and test, with speed, noise, and telephone-band variants and similar-sounding negative utterances. Evaluation uses macOS `say` Kathy/Samantha for English, Piper Dmitri for Russian development, and `say` Milena for Russian test. Those `say` voices must be installed on the Mac.

Thresholds are chosen on the development split and stored in the resulting indexes. Recall across both tiers, strong precision, negative-hit counts, timestamp error, runtime, and sampled process RSS are reported on the test split. Runtime excludes index creation and model loading. Sampled RSS may miss a brief higher memory peak. No transcript accuracy metric is used in v0.2. See [the measured synthetic results](docs/evaluation.md).

Synthetic results validate the pipeline; they do not establish performance on customer calls. Raw Whisper features were not trained as a word retrieval embedding, and human speech may match poorly. The phoneme adapter also produces false matches. The current phoneme adapter processes 20-second chunks and Whisper uses 30-second chunks; words crossing chunk boundaries may be missed. Index creation and repeated runs reuse the persisted template values. Piper templates disable sampling noise to make rebuilds stable on the same runtime.

## Migration from v0.1

`create-index` replaces `index`; `run --index ...` replaces `analyze --glossary ...`. `run` returns `DetectionResult` with hits and no transcript. The old `analyze(...)` API and transcription-based benchmark were removed. Recreate old indexes: v0.1 pickle-based cache files are not accepted by the new store. Model and voice downloads in the existing cache remain reusable.

## License

Project code is Apache-2.0. Optional dependencies and voice datasets have their own licenses; see [dependency and voice notes](docs/licenses.md). Downloaded weights and voices are not bundled in the release.
