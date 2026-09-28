# Noisy Russian recording: development check

The Russian pipeline now has a Russian-trained CTC character adapter alongside
Whisper tiny, base, small, and the multilingual phoneme adapter. It performs
dictionary lookup on timestamped character tokens; it does not return a transcript
or use a transcription service. The checkpoint is
[XLSR Russian](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-russian),
licensed Apache 2.0. It requires about 1.2 GB of downloaded weights.

## Supplied recording

The supplied recording is 40.23 seconds long. The user identified two expected
mentions, near 8 and 16 seconds. This recording was used during development,
including choosing the possible threshold. It is a regression example, **not an
independent accuracy test**. Its audio and generated results are excluded from Git.

Using the user's ten-term Russian dictionary and Russian CTC defaults
(strong 0.85, possible 0.60), the updated pipeline returns:

| Term | Start | End | Match score | Tier |
| --- | ---: | ---: | ---: | --- |
| нимесил | 8.569 s | 8.929 s | 0.7143 | possible |
| антигриппин | 16.717 s | 17.818 s | 0.6364 | possible |

There are no additional reported hits. Scores are character edit similarities,
not probabilities. The timestamps describe the matching CTC tokens; particularly
in noise, they may cover only part of the spoken word.

The same recording was uploaded through the web UI and produced these exact two
results. The first demo analysis after a server restart took 42.40 seconds on the
local 8 GB Apple Silicon Mac, including loading the model and converting audio.
This larger model prioritizes Russian recognition over the tiny model's latency.
Repeat jobs reuse the loaded encoder; their speed has not been measured here.

Comparison on this recording with the same dictionary:

| Configuration | Observed result |
| --- | --- |
| Whisper tiny | нимесил only |
| Whisper base | нимесил only |
| Original multilingual phoneme matching | антигриппин, false метформин, false сертралин |
| Phoneme matching with pause/duration checks | антигриппин only |
| Russian CTC with boundary/pause/duration checks | Both expected terms; no extra hits |

Whisper small is available as an option but was not evaluated on this recording.
A larger acoustic model alone did not solve this example: Whisper base still
missed антигриппин.

## Lookup changes

- Reject token paths across pauses longer than 0.6 seconds. CTC collapse removes
  blank frames, so naive token lookup could otherwise combine separate words.
- Reject implausibly long candidates, with a duration limit that scales with
  dictionary token length. This removes the former nine-second сертралин match.
- For character lookup, require word boundaries or a speech gap at the candidate's
  ends. Internal word delimiters remain allowed to tolerate words split by noise.
- Keep uncertain matches in the possible tier. None of the rules depend on these
  particular drug names or their expected timestamps.

## Separate synthetic test

An initial development calibration selected thresholds of 0.45/0.45. On separate
synthetic speech that permissive configuration produced two false hits on negative
clips and only 64.3% strong precision. Those thresholds were rejected.

The revised boundary checks and defaults of 0.85/0.60 were then checked on 13
held-out synthetic clips: ten drug mentions and three negative utterances. Test
speech used macOS Milena, with clean, speed, noise, and telephone-band variants;
development speech used Piper Dmitri. The dictionary contained the same ten terms.

| Metric | Result |
| --- | ---: |
| Term recall, strong + possible | 8/10 (80%) |
| Strong precision | 2/2 (100%) |
| False hits on three negative clips | 0 |
| Mean timestamp boundary error on correct hits | 0.084 s |

The missed synthetic terms were омепразол and пантопразол. There were no incorrect
reported terms on these positive clips. These small, synthetic results validate
the pipeline; they do not establish performance on customer calls. The default
thresholds were developed using the supplied human example, not calibrated solely
on a separate synthetic development split. Broader labeled real recordings are
needed to measure generalization and calibrate the detector for a deployment.

## Reproduce locally

```sh
uv sync --extra local --extra russian --extra dev
uv run stt-anything prepare --encoder russian-ctc --language ru
uv run stt-anything create-index --glossary my-russian-dictionary.jsonl \
  --language ru --encoder russian-ctc --offline --output indexes/russian-ctc.npz
uv run stt-anything run recording.wav --index indexes/russian-ctc.npz \
  --offline --output reports/russian-result.json
```

Use a dictionary containing нимесил and антигриппин: the replaceable eight-term
sample is not the user's ten-term dictionary. The web demo preserves saved
dictionaries and recommends Russian CTC when its dependencies and weights are
available. New indexes use the revised defaults; existing saved indexes retain
their chosen thresholds.

The local experiment artifacts are in `reports/noisy-russian/`, including
`improved-human.json`, `heldout-rows.json`, and `heldout-improved.json`. They are
ignored by Git. To evaluate another dictionary independently, use the documented
`benchmark` command; it calibrates on its development split and reports the test
split separately. Its calibrated thresholds may differ from these defaults.
