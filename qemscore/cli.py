"""Command-line interface: qemscore generate | run."""

from __future__ import annotations

import argparse

from qemscore.budget import TIERS
from qemscore.datasets.generate import PRESETS
from qemscore.datasets.split_generate import SPLIT_PRESETS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="qemscore")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser(
        "generate",
        help="generate legacy-v1 or split-v2 benchmark data",
    )
    gen.add_argument(
        "--preset", required=True, choices=sorted(PRESETS.keys() | SPLIT_PRESETS.keys())
    )
    gen.add_argument("--out", required=True)
    gen.add_argument("--master-seed", type=int, default=None)

    runp = sub.add_parser("run", help="run baselines on legacy-v1 or split-v2 data")
    runp.add_argument("--data", required=True)
    runp.add_argument("--out", required=True)
    runp.add_argument(
        "--tier",
        type=str.upper,
        choices=sorted(TIERS),
        help=(
            "budget tier; required for split-v2 unless split_spec.budget_tier "
            "declares it"
        ),
    )

    args = parser.parse_args(argv)

    if args.command == "generate":
        from qemscore.datasets.generate import generate

        manifest = generate(args.preset, args.out, master_seed=args.master_seed)
        print(f"generated {manifest['counts']['items']} items -> {args.out}")
        print(f"dataset_hash {manifest['dataset_hash']}")
        return 0

    if args.command == "run":
        from qemscore.runner.run import run

        try:
            run(args.data, args.out, budget_tier=args.tier)
        except ValueError as error:
            runp.error(str(error))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
