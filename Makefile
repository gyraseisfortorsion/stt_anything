UV ?= uv

.PHONY: check lint typecheck test format build
check: lint typecheck test

lint:
	$(UV) run --no-sync ruff check stt_anything tests demo
	$(UV) run --no-sync ruff format --check stt_anything tests demo
	node --check demo/static/app.js
	node --check demo/static/model.mjs

typecheck:
	$(UV) run --no-sync mypy

test:
	$(UV) run --no-sync pytest
	node --test demo/tests/*.test.mjs

format:
	$(UV) run --no-sync ruff check stt_anything tests demo --fix
	$(UV) run --no-sync ruff format stt_anything tests demo

build:
	$(UV) build
