"""Command-line interface: qem-bench generate | run."""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="qem-bench")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="generate a dataset from a preset")
    gen.add_argument("--preset", required=True, choices=["t0-smoke", "t0-micro"])
    gen.add_argument("--out", required=True)
    gen.add_argument("--master-seed", type=int, default=None)

    runp = sub.add_parser("run", help="run baselines on a generated dataset")
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

        run(args.data, args.out)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
