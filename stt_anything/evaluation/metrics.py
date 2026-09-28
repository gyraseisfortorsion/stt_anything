from __future__ import annotations

from typing import Any

import numpy as np

from ..detection import retier
from ..models import Hit
from .synthetic import Case


def _is_true(case: Case, hit: Hit) -> bool:
    if case.term_id != hit.term_id or case.start is None or case.end is None:
        return False
    overlap = max(0, min(case.end, hit.end) - max(case.start, hit.start))
    return overlap / max(case.end - case.start, 1e-6) >= 0.3


def metrics(
    rows: list[tuple[Case, tuple[Hit, ...]]], strong: float, possible: float
) -> dict[str, Any]:
    positive = sum(case.term_id is not None for case, _ in rows)
    true_all = 0
    true_strong = 0
    strong_count = 0
    false_negative_hits = 0
    errors = []
    for case, hits in rows:
        chosen = retier(hits, strong, possible)
        strong_hits = [hit for hit in chosen if hit.tier == "strong"]
        strong_count += len(strong_hits)
        true_strong += sum(_is_true(case, hit) for hit in strong_hits)
        correct = [hit for hit in chosen if _is_true(case, hit)]
        if correct:
            true_all += 1
            best = max(correct, key=lambda hit: hit.score)
            assert case.start is not None and case.end is not None
            errors.append((abs(best.start - case.start) + abs(best.end - case.end)) / 2)
        if case.term_id is None:
            false_negative_hits += len(chosen)
    return {
        "term_recall": round(true_all / positive, 4) if positive else None,
        "strong_precision": round(true_strong / strong_count, 4) if strong_count else None,
        "false_hits_on_negative_cases": false_negative_hits,
        "mean_timestamp_error_seconds": round(float(np.mean(errors)), 3) if errors else None,
        "positive_cases": positive,
        "total_cases": len(rows),
    }


def calibrate(rows: list[tuple[Case, tuple[Hit, ...]]]) -> tuple[float, float]:
    grid = [round(float(value), 3) for value in np.linspace(0.25, 1, 76)]
    strong = 1.0
    best_recall = -1.0
    for value in grid:
        metric = metrics(rows, value, value)
        precision = metric["strong_precision"]
        recall = metric["term_recall"] or 0
        if (precision is None or precision >= 0.9) and recall > best_recall:
            strong, best_recall = value, recall
    possible = strong
    best_f2 = -1.0
    for value in grid:
        if value > strong:
            continue
        metric = metrics(rows, strong, value)
        recall = metric["term_recall"] or 0
        hits = sum(sum(hit.score >= value for hit in detected) for _, detected in rows)
        true_hits = sum(
            sum(hit.score >= value and _is_true(case, hit) for hit in detected)
            for case, detected in rows
        )
        precision = true_hits / hits if hits else 0
        f2 = 5 * precision * recall / (4 * precision + recall) if precision + recall else 0
        if f2 > best_f2:
            possible, best_f2 = value, f2
    return strong, possible
