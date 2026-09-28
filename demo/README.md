# Keyword lab web demo

A local, dependency-free web frontend for the keyword spotting library. Plain HTML,
CSS, and JavaScript; the Python standard library serves the UI and runs real model
jobs. No frontend build step, CDN, OpenAI key, or ElevenLabs key.

## Start

Run from the repository root with Python 3.11 or 3.12 and FFmpeg installed:

```sh
uv sync --extra local --extra dev
uv run stt-anything prepare --encoder whisper --language both
uv run python -m demo.server
```

Open **http://localhost:8765**. `--port 8766` changes the port;
`--data-dir /path/to/indexes` changes index storage. To keep previously installed
optional encoders when syncing, include their extras too.

For the Russian-trained character encoder (no eSpeak or TTS templates needed):

```sh
uv sync --extra local --extra russian --extra dev
uv run stt-anything prepare --encoder russian-ctc --language ru
```

Its checkpoint is about 1.2 GB on disk; it supports Russian dictionaries only.
Models still run one at a time. Initial model loading is slower than repeat clips.

For larger Whisper acoustic encoders:

```sh
uv run stt-anything prepare --encoder whisper-base --language both
uv run stt-anything prepare --encoder whisper-small --language both
```

For the optional phoneme encoder:

```sh
# macOS; Linux users can install espeak-ng with their package manager
brew install espeak-ng
uv sync --extra local --extra phoneme --extra dev
uv run stt-anything prepare --encoder phoneme --language both
```

Models and voices need an initial download. With weights cached, the demo defaults
to offline processing. The UI also has an explicit “Download missing models and
voices” option; it requires the selected encoder's dependencies to be installed.

## Use

1. Choose English or Russian. Edit the dictionary rows, add pronunciations separated
   by semicolons, or import a file/pasted list. “Use pharma sample” loads eight terms
   in the selected language. Each index uses one language, up to 100 terms and 300
   pronunciation forms. Russian spellings cannot be indexed with English
   pronunciation settings; use a Russian index or a separate English dictionary.
2. Choose an encoder. Whisper tiny is the small default. For Russian, the demo
   recommends Russian CTC when installed and cached, then the multilingual phoneme
   model when available. Whisper base and small are larger multilingual acoustic
   comparisons; their scores and thresholds still need evaluation on your audio. MFCC is
   an experimental speed baseline: it missed terms in our human recordings and
   should not be your starting choice for microphone audio. Startup selects a
   saved index for the recommended encoder and the latest dictionary rather than
   the newest MFCC experiment. When no matching index exists, it keeps your latest
   Russian dictionary in the editor, ready to rebuild. The
   lookup is paired with the representation: subsequence DTW for frame features,
   phoneme matching for phoneme tokens, and character matching for Russian CTC.
   Token matching rejects paths across pauses longer than 0.6 seconds or
   implausibly long word durations. Character matching checks token word boundaries
   to avoid accepting arbitrary fragments of longer words. The Russian defaults
   are strong 0.85 / possible 0.60; these scores represent edit similarity, not
   neural posterior confidence. All detectors can still miss or misidentify words. The CLI/library expose the wider plugin
   contracts; the demo offers these built-in pairings. Advanced options expose
   strong/possible thresholds and occurrences per pronunciation. Defaults are
   starting values, not a calibrated accuracy guarantee.
3. Build the index. Progress counts actual completed pronunciation templates;
   model loading/downloads happen before the first template completes. Indexes are
   saved and can be selected after a restart.
4. Record in the browser (grant microphone permission) or upload an existing audio
   clip. Clips are limited to two minutes / 32 MB. Press “Stop & detect” to analyze.
   Encoding and lookup run locally; there is no transcription step.
5. See terms over the waveform, click a marker or result to play its time span,
   filter strong/possible hits, sort by time/score, and save results as JSON.
   If a run misses words, use “Try phoneme/Whisper on this clip” to rebuild the
   same dictionary with the other encoder and reprocess the original clip. Or
   select/build another index and use “Analyze this clip with the active index”.
   No new recording is needed. A new run resets the result filter to All.
   No-match feedback distinguishes quiet audio, the MFCC baseline, and candidates
   rejected by thresholds. It does not promote rejected scores to detections.
   Keyboard arrows seek the focused waveform; Space toggles playback.

Amber-to-teal colors show increasing **match scores**, not probabilities. Timestamps
are estimates. The waveform includes all reported hits; tier filters apply to the
list. A new result replaces the previous recording in the browser.

## Runtime and storage

The backend uses one worker to serialize model work and retains the most recently
used encoder for faster repeat recordings. Index creation and analysis reuse the
library's public components. The first request for a model is slower than repeat
requests. Detection progress reports stages; the duration of each stage depends
on model/audio length. Indexing progress reports completed templates.

Saved indexes live in `~/.cache/stt-anything/demo` (or under `STT_ANYTHING_CACHE`).
Uploaded/recorded audio is held in a temporary server session and removed on clean
shutdown. Audio is served to the browser as a normalized WAV for playback. The
server binds to `127.0.0.1`; this is a local demo, not a hosted multiuser service.
Stop with Ctrl+C. No data or downloaded weights are written into Git.

## Development checks

```sh
make check
node --test demo/tests/*.test.mjs
```

`make check` runs Ruff, strict mypy, Python tests with 100% line and branch coverage
(including the demo backend), and the JavaScript dictionary/score helper tests.
Node is used only for these development tests. Python tests use fake model adapters
and real localhost HTTP requests; they require no downloads. Browser recording and
visual interactions are checked separately against the running server.

Structure:

- `server.py`: localhost HTTP routes and static assets.
- `service.py`: dictionary validation, job queue, model reuse, index persistence,
  normalized playback audio, and waveform peaks.
- `diagnostics.py`: observe raw candidate scores and microphone levels for
  no-match feedback, without changing detection thresholds.
- `static/`: frontend, microphone recorder, waveform rendering, and pure helpers.
- `tests/model.test.mjs`: JavaScript format, timestamp, score, and filtering checks.
- `../tests/test_demo.py`: backend and HTTP tests.

Synthetic tests validate the pipeline. Customer-call accuracy remains unmeasured;
use labeled recordings to calibrate thresholds before drawing accuracy conclusions.
Dependency and voice licenses are documented in [../docs/licenses.md](../docs/licenses.md).
