"""Command-line interface: qem-bench generate | run."""

from __future__ import annotations

import argparse

from qem_bench.datasets.generate import PRESETS
from qem_bench.datasets.split_generate import SPLIT_PRESETS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="qem-bench")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser(
        "generate",
        help=(
            "split-v2 generation and validation are available; "
            "run supports legacy-v1 only"
        ),
    )
    gen.add_argument(
        "--preset", required=True, choices=sorted(PRESETS.keys() | SPLIT_PRESETS.keys())
    )
    gen.add_argument("--out", required=True)
    gen.add_argument("--master-seed", type=int, default=None)

    runp = sub.add_parser(
        "run", help="run baselines on legacy-v1 data; split-v2 is rejected"
    )
    runp.add_argument("--data", required=True)
    runp.add_argument("--out", required=True)

    args = parser.parse_args(argv)

    if args.command == "generate":
        from qem_bench.datasets.generate import generate

        manifest = generate(args.preset, args.out, master_seed=args.master_seed)
        print(f"generated {manifest['counts']['items']} items -> {args.out}")
        print(f"dataset_hash {manifest['dataset_hash']}")
        return 0

    if args.command == "run":
        from qem_bench.runner.run import run

        try:
            run(args.data, args.out)
        except ValueError as error:
            if str(error).startswith("split-v2 runner loading"):
                runp.error(
                    "split-v2 artifacts can be generated and validated, but qem-bench "
                    "run currently supports legacy-v1 artifacts only"
                )
            raise
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
