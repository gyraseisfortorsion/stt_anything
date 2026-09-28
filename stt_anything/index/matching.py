from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar, cast

import numpy as np
from numba import njit

from ..models import Array, FloatArray, Match

F = TypeVar("F", bound=Callable[..., object])


def normalize_frames(features: FloatArray) -> FloatArray:
    centered = features - features.mean(axis=0, keepdims=True)
    norm = np.linalg.norm(centered, axis=1, keepdims=True)
    return np.asarray(centered / np.maximum(norm, 1e-6), dtype=np.float32)


def compiled(function: F) -> F:
    return cast(F, njit(cache=True)(function))


@compiled
def _dtw(cost: FloatArray) -> tuple[Array, Array]:
    rows, cols = cost.shape
    previous = np.full(cols + 1, np.inf, dtype=np.float32)
    previous[:] = 0
    previous_steps = np.zeros(cols + 1, dtype=np.int32)
    previous_start = np.arange(cols + 1, dtype=np.int32)
    for row in range(1, rows + 1):
        current = np.full(cols + 1, np.inf, dtype=np.float32)
        current_steps = np.zeros(cols + 1, dtype=np.int32)
        current_start = np.zeros(cols + 1, dtype=np.int32)
        for col in range(1, cols + 1):
            diagonal = previous[col - 1]
            above = previous[col] + 0.08
            left = current[col - 1] + 0.08
            if diagonal <= above and diagonal <= left:
                base = diagonal
                steps = previous_steps[col - 1]
                begin = previous_start[col - 1]
            elif above <= left:
                base = above
                steps = previous_steps[col]
                begin = previous_start[col]
            else:
                base = left
                steps = current_steps[col - 1]
                begin = current_start[col - 1]
            current[col] = base + cost[row - 1, col - 1]
            current_steps[col] = steps + 1
            current_start[col] = begin
        previous, previous_steps, previous_start = current, current_steps, current_start
    average = previous[1:] / np.maximum(previous_steps[1:], 1)
    return average, previous_start[1:]


def subsequence_matches(
    query: FloatArray, audio: FloatArray, spans: FloatArray, top_k: int = 3
) -> list[Match]:
    if query.size == 0 or audio.size == 0 or len(audio) < max(4, len(query) // 2):
        return []
    q = normalize_frames(query)
    a = normalize_frames(audio)
    costs = np.clip(1 - q @ a.T, 0, 2).astype(np.float32)
    average, begins = _dtw(costs)
    candidates: list[Match] = []
    for endpoint in np.argsort(average):
        start = int(begins[endpoint])
        end = int(endpoint + 1)
        duration = end - start
        if duration < max(3, int(len(query) * 0.45)) or duration > len(query) * 2.2:
            continue
        score = float(np.exp(-2 * average[endpoint]))
        candidate = Match(
            round(float(spans[start, 0]), 3), round(float(spans[end - 1, 1]), 3), round(score, 4)
        )
        if all(_overlap(candidate, prior) < 0.3 for prior in candidates):
            candidates.append(candidate)
        if len(candidates) >= top_k:
            break
    return candidates


def _overlap(first: Match, second: Match) -> float:
    common = max(0, min(first.end, second.end) - max(first.start, second.start))
    shorter = min(first.end - first.start, second.end - second.start)
    return common / max(shorter, 1e-6)


def edit_distance(first: list[int], second: list[int]) -> int:
    previous = list(range(len(second) + 1))
    for left in first:
        current = [previous[0] + 1]
        for idx, right in enumerate(second, 1):
            current.append(
                min(current[-1] + 1, previous[idx] + 1, previous[idx - 1] + (left != right))
            )
        previous = current
    return previous[-1]


def phone_matches(
    query: list[int],
    phones: list[int],
    spans: FloatArray,
    top_k: int = 3,
    word_delimiter: int | None = None,
) -> list[Match]:
    if not query or not phones:
        return []
    candidates = []
    lower = max(1, int(len(query) * 0.6))
    upper = max(lower, int(len(query) * 1.4) + 1)
    for start in range(len(phones)):
        if word_delimiter is not None and (
            phones[start] == word_delimiter
            or (
                start > 0
                and phones[start - 1] != word_delimiter
                and spans[start, 0] - spans[start - 1, 1] < 0.3
            )
        ):
            continue
        for length in range(lower, min(upper, len(phones) - start) + 1):
            end = start + length
            if word_delimiter is not None and (
                phones[end - 1] == word_delimiter
                or (
                    end < len(phones)
                    and phones[end] != word_delimiter
                    and spans[end, 0] - spans[end - 1, 1] < 0.3
                )
            ):
                continue
            window = spans[start : start + length]
            # Blank CTC frames disappear during collapse; do not stitch a word across pauses.
            duration = float(window[-1, 1] - window[0, 0])
            gaps = window[1:, 0] - window[:-1, 1]
            if duration > max(1.5, len(query) * 0.35) or (gaps > 0.6).any():
                continue
            distance = edit_distance(query, phones[start : start + length])
            score = max(0.0, 1 - distance / max(len(query), length))
            candidates.append(
                Match(
                    round(float(spans[start, 0]), 3),
                    round(float(spans[start + length - 1, 1]), 3),
                    round(score, 4),
                )
            )
    selected: list[Match] = []
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        if all(_overlap(candidate, prior) < 0.3 for prior in selected):
            selected.append(candidate)
        if len(selected) >= top_k:
            break
    return selected
