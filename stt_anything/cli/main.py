from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from .. import components
from ..api import create_index, run
from ..cache import (
    cache_root,
    prepare_eval_russian_voice,
    prepare_phoneme,
    prepare_russian_ctc,
    prepare_voice,
)
from ..encoder import WhisperEncoder


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        prog="stt-anything", description="Modular keyword spotting without transcription"
    )
    sub = result.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser(
        "prepare", help="Download optional encoder weights and local template voices"
    )
    prepare.add_argument(
        "--encoder",
        choices=["whisper", "whisper-base", "whisper-small", "mfcc", "phoneme", "russian-ctc"],
        default="whisper",
    )
    prepare.add_argument("--language", choices=["en", "ru", "both"], default="both")
    prepare.add_argument(
        "--evaluation", action="store_true", help="Also prepare synthetic evaluation voices"
    )

    create = sub.add_parser("create-index", help="Build a reusable glossary index")
    create.add_argument("--glossary", required=True, type=Path)
    create.add_argument("--language", required=True)
    create.add_argument(
        "--encoder",
        default="whisper",
        help="whisper, whisper-base, whisper-small, russian-ctc, mfcc, phoneme, or module:factory",
    )
    create.add_argument("--builder", default="templates")
    create.add_argument("--source", default="piper")
    create.add_argument("--store", default="npz")
    create.add_argument("--output", required=True, type=Path)
    create.add_argument("--offline", action="store_true")

    detect = sub.add_parser("run", help="Return timestamped keywords from an audio file")
    detect.add_argument("audio", type=Path)
    detect.add_argument("--index", required=True, type=Path)
    detect.add_argument(
        "--encoder", help="Override encoder factory (signature must match the index)"
    )
    detect.add_argument("--lookup", help="dtw, phoneme, or module:factory; inferred if omitted")
    detect.add_argument("--store", default="npz")
    detect.add_argument("--strong-threshold", type=float)
    detect.add_argument("--possible-threshold", type=float)
    detect.add_argument("--offline", action="store_true")
    detect.add_argument("--output", type=Path, help="Write JSON instead of printing it")

    benchmark = sub.add_parser(
        "benchmark", help="Calibrate on synthetic development audio and evaluate held-out audio"
    )
    benchmark.add_argument("--glossary", required=True, type=Path)
    benchmark.add_argument("--language", choices=["en", "ru", "both"], default="both")
    benchmark.add_argument(
        "--encoders",
        nargs="+",
        choices=["whisper", "whisper-base", "whisper-small", "russian-ctc", "mfcc", "phoneme"],
        default=["whisper", "mfcc"],
    )
    benchmark.add_argument("--offline", action="store_true")
    benchmark.add_argument("--index-dir", type=Path)
    benchmark.add_argument("--output", type=Path)
    return result


def emit(value: dict[str, Any], output: Path | None = None) -> None:
    encoded = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    if output is None:
        print(encoded, end="")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    cli = parser()
    args = cli.parse_args(argv)
    try:
        if args.command == "prepare":
            if args.encoder.startswith("whisper"):
                WhisperEncoder(
                    checkpoint="tiny"
                    if args.encoder == "whisper"
                    else args.encoder.removeprefix("whisper-")
                ).prepare()
            elif args.encoder == "phoneme":
                prepare_phoneme()
            elif args.encoder == "russian-ctc":
                prepare_russian_ctc()
            languages = ("en", "ru") if args.language == "both" else (args.language,)
            if args.encoder not in ("phoneme", "russian-ctc"):
                for language in languages:
                    prepare_voice(language)
            if args.evaluation:
                prepare_eval_russian_voice()
            emit({"cache": str(cache_root()), "encoder": args.encoder})
        elif args.command == "create-index":
            model = components.encoder(args.encoder, args.language, args.offline)
            index = create_index(
                args.glossary,
                args.language,
                model,
                components.source(args.source, args.offline),
                builder=components.builder(args.builder),
            )
            index = replace(index, encoder_name=args.encoder)
            components.store(args.store).save(index, args.output)
            emit(
                {
                    "index": str(args.output),
                    "terms": len(index.terms),
                    "templates": len(index.entries),
                }
            )
        elif args.command == "run":
            storage = components.store(args.store)
            index = storage.load(args.index)
            model = components.encoder(
                args.encoder or index.encoder_name, index.language, args.offline
            )
            search = components.lookup(args.lookup) if args.lookup else None
            result = run(
                args.audio,
                index,
                encoder=model,
                lookup=search,
                strong_threshold=args.strong_threshold,
                possible_threshold=args.possible_threshold,
            )
            emit(result.to_dict(), args.output)
        else:
            from ..evaluation import benchmark

            emit(
                benchmark(
                    args.glossary,
                    args.language,
                    args.encoders,
                    offline=args.offline,
                    index_dir=args.index_dir,
                ),
                args.output,
            )
    except (OSError, ValueError, TypeError, RuntimeError, ImportError, AttributeError) as error:
        cli.exit(2, f"stt-anything: {error}\n")
