"""Command line: `dry-run` before you spend, `evaluate` when you mean it."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .checkpoint import Checkpoint
from .evaluator import Evaluator
from .policy import Policy


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gvc", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    for name in ("dry-run", "evaluate"):
        p = sub.add_parser(name)
        p.add_argument("--checkpoint", required=True, type=Path)
        p.add_argument("--image", required=True, type=Path)
        p.add_argument("--threshold", type=float, default=None)
        p.add_argument("--json", type=Path, default=None)

    args = parser.parse_args(argv)
    checkpoint = Checkpoint.load(args.checkpoint)
    policy = Policy(args.threshold) if args.threshold is not None else Policy()
    evaluator = Evaluator(policy=policy)

    if args.command == "dry-run":
        payload = evaluator.dry_run(checkpoint, args.image)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        if payload["missing_measurements"]:
            missing = ", ".join(payload["missing_measurements"])
            print(f"\nNote: no measurer registered for {missing}.", file=sys.stderr)
            print("Checks depending on it will be reported not_assessable.", file=sys.stderr)
        return 0

    result = evaluator.evaluate(checkpoint, args.image)
    if args.json:
        args.json.write_text(result.model_dump_json(indent=2), encoding="utf-8")

    if not result.image_usable:
        print(f"Image not usable: {result.unusable_reason}")
        print("Recapture needed. The subject is not evaluated.")
        return 0

    print(f"{result.checkpoint_id} — {result.summary}\n")
    for f in result.findings:
        mark = {"pass": "ok ", "fail": "FAIL", "not_assessable": "??  "}[f.status.value]
        flag = "  [human review]" if f.needs_human_review else ""
        print(f"{mark} {f.check}  ({f.confidence:.2f}){flag}")
        print(f"     observed: {f.observed}")
        print(f"     action:   {f.action}")
    print(f"\ncost: US${result.usage.cost_usd:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
