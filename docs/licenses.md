# Dependency and voice licenses

The toolkit's own source is Apache-2.0, as provided in the root `LICENSE`. Model weights and voices are downloaded separately and are not part of the source distribution. This list describes the direct components used by the project; installed transitive dependencies retain their own notices.

## Runtime components

| Component | Role | Upstream license |
| --- | --- | --- |
| [NumPy](https://github.com/numpy/numpy/blob/main/LICENSE.txt) | Numeric arrays | BSD-3-Clause |
| [SciPy](https://github.com/scipy/scipy/blob/main/LICENSE.txt) | Signal processing | BSD-3-Clause |
| [Numba](https://github.com/numba/numba/blob/main/LICENSE) | DTW acceleration | BSD-2-Clause |
| [Whisper](https://github.com/openai/whisper/blob/main/LICENSE) | Optional tiny/base/small acoustic encoders | MIT for code and weights |
| [Piper](https://github.com/OHF-Voice/piper1-gpl/blob/main/COPYING) | Optional template TTS | GPL-3.0 |
| [Transformers](https://github.com/huggingface/transformers/blob/main/LICENSE) | Optional phoneme adapter | Apache-2.0 |
| [PyTorch](https://github.com/pytorch/pytorch/blob/main/LICENSE) | Optional model inference | BSD-style license; see upstream notices |
| [Hugging Face Hub](https://github.com/huggingface/huggingface_hub/blob/main/LICENSE) | Optional model downloads | Apache-2.0 |
| [Phonemizer](https://github.com/bootphon/phonemizer/blob/master/LICENSE) | Optional text phonemization | GPL-3.0 |
| [espeak-ng](https://github.com/espeak-ng/espeak-ng/blob/master/COPYING) | Phoneme backend system dependency | GPL-3.0 |
| [Meta phoneme checkpoint](https://huggingface.co/facebook/wav2vec2-xlsr-53-espeak-cv-ft) | Optional CTC weights | Apache-2.0 according to the model card |
| [Russian XLSR checkpoint](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-russian) | Optional Russian CTC character weights | Apache-2.0 according to the model card; trained on Russian Common Voice and CSS10 |
| [psutil](https://github.com/giampaolo/psutil/blob/master/LICENSE) | Optional benchmark memory sampling | BSD-3-Clause |
| [FFmpeg](https://ffmpeg.org/legal.html) | Audio conversion, external executable | LGPL-2.1-or-later or GPL depending on build options |

Development tools: pytest (MIT), pytest-cov (MIT), coverage.py (Apache-2.0), Ruff (MIT), mypy (MIT), and types-psutil (Apache-2.0). Bundled SDKs and optional voices are not required for unit tests.

## Piper voices

The voice repository labels its repository-level license MIT; each voice card also identifies training-data terms. They should not be collapsed into one blanket license claim.

| Voice | Usage | Model-card dataset terms |
| --- | --- | --- |
| [en_US-lessac-low](https://huggingface.co/rhasspy/piper-voices/blob/main/en/en_US/lessac/low/MODEL_CARD) | English indexed pronunciations | Links to the custom [Blizzard 2013 research license](https://www.cstr.ed.ac.uk/projects/blizzard/2013/lessac_blizzard2013/license.html) for its dataset |
| [ru_RU-denis-medium](https://huggingface.co/rhasspy/piper-voices/blob/main/ru/ru_RU/denis/medium/MODEL_CARD) | Russian indexed pronunciations | CC0 dataset; card states fine-tuning from Lessac medium |
| [ru_RU-dmitri-medium](https://huggingface.co/rhasspy/piper-voices/blob/main/ru/ru_RU/dmitri/medium/MODEL_CARD) | Russian synthetic development audio | CC0 dataset; card states fine-tuning from Lessac medium |

The Lessac dataset license describes research-only use of those materials; the voice card does not itself resolve downstream commercial rights for generated templates. Select a pronunciation source with suitable terms for your distribution. The template abstraction makes replacing it independent of encoding and lookup.

macOS `say` voices are supplied by Apple under the operating system's terms. They are used only for local synthetic evaluation and are not redistributed. The human recording in `human_test_dataset` and generated reports are excluded from Git.
