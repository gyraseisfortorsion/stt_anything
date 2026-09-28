# Contributing

Use Python 3.11 or 3.12, Node.js 20 or newer, and FFmpeg. The developer environment needs no downloaded models:

```sh
uv sync --locked --extra dev
make check
```

- `make lint`: Ruff lint and formatting checks, plus frontend JavaScript syntax checks.
- `make typecheck`: strict mypy for library, CLI, and demo backend source.
- `make test`: pytest with a 100% line and branch coverage gate for all Python source modules, plus JavaScript dictionary/score helper tests.
- `make format`: apply Ruff fixes and formatting.
- `make build`: create a wheel and source distribution.

Keep adapter imports lazy so a base installation does not import model frameworks. Define an encoder signature that changes whenever the feature representation or model configuration changes. Preserve timestamp offsets across chunks. Add tests for real behavior, invalid input, repeated terms, silence, and index compatibility. Stub model SDK boundaries in unit tests; keep actual inference runs and their results separate.

Do not relax the coverage gate or exclude source modules to make checks pass. Unit coverage measures execution and contract behavior, not model accuracy. The demo frontend has separate helper tests and browser verification; it is not included in Python coverage. The unit suite runs with Numba JIT disabled to exercise the Python matching code; offline integration runs exercise the compiled path.

The sample pharma glossary is replaceable. Record the glossary and index configuration used for an experiment and keep development calibration separate from held-out evaluation. Personal audio, generated audio, indexes, downloaded model weights, and local reports stay outside committed source.

GitHub Actions runs `make check` and a clean wheel installation check on Python 3.11 and 3.12 for pushes and pull requests. The workflow does not download speech models or publish releases. Before committing, inspect staged files and keep personal audio, generated indexes, weights, credentials, and local reports out of Git.
