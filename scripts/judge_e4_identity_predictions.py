#!/usr/bin/env python3
"""Apply a bounded, cached, blinded VLM judge to captured E4 predictions."""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.e4_identity_judging import summarize_judgments  # noqa: E402
from providers.vehicle.vlm_reid import OpenAIVisionIdentityComparator  # noqa: E402


def _data_url(path: Path) -> str:
    suffix = path.suffix.lower()
    media = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}.get(suffix)
    if media is None:
        raise ValueError(f"unsupported judge image type: {path}")
    return f"data:{media};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--model", default=os.getenv("E4_VLM_JUDGE_MODEL", "gpt-4o-mini-2024-07-18"))
    parser.add_argument("--maximum-unique-pairs", type=int, default=30)
    parser.add_argument("--determined-confidence", type=float, default=0.60)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--execute", action="store_true", help="authorize paid, uncached VLM requests")
    args = parser.parse_args()
    if not 1 <= args.maximum_unique_pairs <= 100:
        parser.error("--maximum-unique-pairs must be between 1 and 100")
    if not 0 <= args.determined_confidence <= 1:
        parser.error("--determined-confidence must be in [0, 1]")

    predictions = []
    for manifest in args.manifest:
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("schema_version") != "fable.e4_identity_prediction.v1":
                    raise ValueError(f"unsupported prediction schema in {manifest}")
                predictions.append(row)
    unique = {row["pair_id"]: row for row in predictions}
    if len(unique) > args.maximum_unique_pairs:
        raise RuntimeError(
            f"{len(unique)} unique pairs exceed the explicit budget of "
            f"{args.maximum_unique_pairs}; reduce the bounded trace set or raise the cap"
        )
    cache = json.loads(args.cache.read_text(encoding="utf-8")) if args.cache.is_file() else {}
    missing = [pair_id for pair_id in sorted(unique) if f"{args.model}:{pair_id}" not in cache]
    if missing and not args.execute:
        print(json.dumps({"unique_pairs": len(unique), "cached_pairs": len(unique) - len(missing), "paid_calls_required": len(missing), "execute": False}, sort_keys=True))
        return 2
    comparator = None
    if missing:
        key = os.getenv("OPENAI_API_KEY", "")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is required for uncached E4 judgments")
        comparator = OpenAIVisionIdentityComparator(api_key=key, model=args.model, timeout_seconds=args.timeout_seconds)
    for pair_id in missing:
        row = unique[pair_id]
        decision = comparator.compare(
            entity_kind=row["entity_kind"],
            left_image_url=_data_url(Path(row["left_image_path"])),
            right_image_url=_data_url(Path(row["right_image_path"])),
        )
        cache[f"{args.model}:{pair_id}"] = {
            "same_identity": decision.same_identity,
            "confidence": decision.confidence,
            "reason": decision.reason,
        }
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        args.cache.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    judged = []
    for row in predictions:
        decision = cache[f"{args.model}:{row['pair_id']}"]
        label = (
            "UNDETERMINED"
            if decision["confidence"] < args.determined_confidence
            else "MATCH" if decision["same_identity"] else "NO_MATCH"
        )
        judged.append({
            **row,
            "judge_model_id": args.model,
            "judge_label": label,
            "judge_confidence": decision["confidence"],
            "judge_reason": decision["reason"],
            "agreement": label != "UNDETERMINED" and bool(row["predicted_same_identity"]) == bool(decision["same_identity"]),
        })
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "identity_judgments.jsonl").open("w", encoding="utf-8") as handle:
        for row in judged:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary = summarize_judgments(judged)
    summary.update({"judge_model_id": args.model, "paid_calls_this_invocation": len(missing), "cache_path": str(args.cache.resolve())})
    (args.output / "identity_judgment_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
