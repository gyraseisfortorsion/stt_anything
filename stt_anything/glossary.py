from __future__ import annotations

import json
from pathlib import Path

from .models import Term


def load_glossary(path: str | Path, language: str | None = None) -> list[Term]:
    terms: list[Term] = []
    seen: set[str] = set()
    for line_no, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            item = json.loads(raw)
            term_id = item["id"]
            lang = item["language"]
            display = item["display"]
            aliases = item.get("aliases", [])
            spoken = item.get("spoken_forms", [])
            if not all(
                isinstance(value, str) and value.strip() for value in (term_id, lang, display)
            ):
                raise ValueError("id, language, and display must be nonempty strings")
            if not isinstance(aliases, list) or not isinstance(spoken, list):
                raise ValueError("aliases and spoken_forms must be lists")
            if not all(isinstance(value, str) and value.strip() for value in aliases + spoken):
                raise ValueError("aliases and spoken_forms must contain nonempty strings")
            if term_id in seen:
                raise ValueError(f"duplicate id: {term_id}")
            seen.add(term_id)
            if language is None or lang == language:
                terms.append(Term(term_id, lang, display, tuple(aliases), tuple(spoken)))
        except (KeyError, TypeError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"{path}:{line_no}: {exc}") from exc
    if not terms:
        raise ValueError(f"No glossary terms found for {language or 'any language'} in {path}")
    return terms
