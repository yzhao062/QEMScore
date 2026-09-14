from __future__ import annotations

import argparse

from qemscore.reports.generate import generate_report


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m qemscore.reports")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=2_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260818)
    args = parser.parse_args()
    outputs = generate_report(
        args.manifest,
        args.out,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
