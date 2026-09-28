# Synthetic keyword detection evaluation

Measured locally on this Mac with v0.2 and the sample pharma glossary. Each language has eight positive and three negative held-out test utterances. Thresholds were selected on a separate development split. These small, synthetic tests validate the pipeline; customer-call accuracy remains unmeasured.

Recall counts both tiers; precision counts strong hits only. False hits count all returned hits on negative utterances. Runtime covers detection of development and test audio plus calibration and index persistence, and excludes template/model setup. RSS is sampled process memory, not a guaranteed peak.

| Language | Encoder | Recall | Strong precision | Negative hits | Timestamp error | Runtime | Observed RSS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| en | whisper | 100.0% | 100.0% | 0 | 0.052 s | 3.90 s | 561.6 MB |
| en | mfcc | 62.5% | 75.0% | 0 | 0.096 s | 1.38 s | 402.9 MB |
| en | phoneme | 87.5% | 87.5% | 1 | 0.061 s | 155.25 s | 972.3 MB |
| ru | whisper | 100.0% | 88.9% | 2 | 0.045 s | 4.50 s | 496.5 MB |
| ru | mfcc | 62.5% | 100.0% | 0 | 0.163 s | 1.45 s | 281.8 MB |
| ru | phoneme | 100.0% | 100.0% | 0 | 0.128 s | 128.32 s | 1210.4 MB |

The synthetic utterances concatenate separately synthesized prefix, term, and suffix audio with pauses. This makes boundaries easier than spontaneous speech. Human accents, continuous speech, quiet recordings, and long audio need separate labeled evaluation.

```sh
uv run stt-anything benchmark --glossary examples/pharma.jsonl --language both \
  --encoders whisper mfcc phoneme --offline --index-dir reports/calibrated \
  --output reports/benchmark-v2.json
```

Install the local, phoneme, and evaluation extras and prepare models and evaluation voices first. Full JSON results, indexes, and generated audio are local artifacts excluded from Git. The phoneme model is a larger optional comparison; the two small default adapters remain available independently.
