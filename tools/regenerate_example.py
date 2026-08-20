"""Regenerate the shipped report walking skeleton.

Run from the repository root with::

    python tools/regenerate_example.py
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_ROOT = REPOSITORY_ROOT / "examples" / "report-walking-skeleton"
LOCK_PATH = REPOSITORY_ROOT / "requirements" / "ci-py312.lock"
PRESETS = (
    "t0-heisenberg-micro",
    "t0-micro",
    "t0-nc-micro",
    "t0-qaoa-micro",
    "t0-rc-micro",
    "t0-smoke",
)
RUN_METHODS = (
    "raw",
    "ridge",
    "zne",
    "liao",
    "feat-only",
    "noisy-only",
    "shrinkage",
    "shuf-noisy",
)
REPORT_METHODS = ("raw", "ridge", "zne")
REPORT_BOOTSTRAP_SEED = 20260818
SOURCE_DATE_EPOCH = "1787011200"
FROZEN_DATASET_HASHES = {
    "t0-micro": "b9ed7863d6a1ce0d718907262d842e90e59c6eb7479a350837655bf542166bb7",
    "t0-smoke": "edf5837e0da10f1dc93151d5b29d1855f66ad76341ff6c28019096308f8fcdf3",
}
EXPECTED_FILES = (
    *(Path("runs") / preset / "results.json" for preset in PRESETS),
    Path("results-manifest.json"),
    Path("report-trace.json"),
    Path("table1.tex"),
    Path("table1.pdf"),
    Path("figure1.pdf"),
)


def _restart_with_deterministic_hash_seed() -> int | None:
    if os.environ.get("PYTHONHASHSEED") == "0":
        return None
    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = "0"
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        env=environment,
        check=False,
    )
    return completed.returncode


def _configure_environment() -> None:
    os.environ["SOURCE_DATE_EPOCH"] = SOURCE_DATE_EPOCH
    os.environ["FORCE_SOURCE_DATE"] = "1"
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"


def _canonical_lock_sha256() -> str:
    content = LOCK_PATH.read_bytes()
    if b"\r" in content.replace(b"\r\n", b""):
        raise ValueError("CI lock contains an unsupported bare CR line ending")
    return hashlib.sha256(content.replace(b"\r\n", b"\n")).hexdigest()


def _write_lf(path: Path, content: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(content)


def _write_json_lf(path: Path, value: object) -> None:
    _write_lf(path, json.dumps(value, indent=2) + "\n")


def _report_manifest() -> dict[str, object]:
    return {
        "schema_version": "qem-bench-report-manifest-v1",
        "runs": [
            {"results": f"runs/{preset}/results.json"} for preset in PRESETS
        ],
    }


def _compile_table(table_path: Path, build_dir: Path) -> Path:
    pdflatex = shutil.which("pdflatex")
    if pdflatex is None:
        raise RuntimeError("pdflatex is required to regenerate table1.pdf")
    build_dir.mkdir()
    shutil.copyfile(table_path, build_dir / table_path.name)
    completed = subprocess.run(
        [
            pdflatex,
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-file-line-error",
            table_path.name,
        ],
        cwd=build_dir,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        output = "\n".join((completed.stdout, completed.stderr)).strip()
        raise RuntimeError(f"pdflatex failed:\n{output[-4000:]}")
    return build_dir / "table1.pdf"


def _publish(staging_root: Path) -> None:
    staged_files = tuple(
        sorted(
            (
                path.relative_to(staging_root)
                for path in staging_root.rglob("*")
                if path.is_file()
            ),
            key=lambda path: path.as_posix(),
        )
    )
    expected_files = tuple(sorted(EXPECTED_FILES, key=lambda path: path.as_posix()))
    if staged_files != expected_files:
        raise RuntimeError(
            "staged example tree differs from the expected file set: "
            f"{[path.as_posix() for path in staged_files]}"
        )
    for relative_path in EXPECTED_FILES:
        destination = EXAMPLE_ROOT / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(staging_root / relative_path, destination)


def regenerate() -> None:
    sys.path.insert(0, str(REPOSITORY_ROOT))

    from qem_bench.datasets.generate import PRESETS as DATASET_PRESETS
    from qem_bench.datasets.generate import generate
    from qem_bench.reports import generate_report
    from qem_bench.reproducibility import CI_LOCK_SHA256, CI_LOCK_SHA256_ENV
    from qem_bench.runner.run import run, validate_run_artifact

    lock_sha256 = _canonical_lock_sha256()
    if lock_sha256 != CI_LOCK_SHA256:
        raise RuntimeError(
            f"CI lock drift: expected {CI_LOCK_SHA256}, got {lock_sha256}"
        )
    os.environ[CI_LOCK_SHA256_ENV] = lock_sha256

    missing_presets = set(PRESETS) - set(DATASET_PRESETS)
    if missing_presets:
        raise RuntimeError(f"walking-skeleton presets are missing: {missing_presets}")

    with tempfile.TemporaryDirectory(prefix="qem-bench-example-") as temporary:
        work_root = Path(temporary)
        os.environ["MPLCONFIGDIR"] = str(work_root / "matplotlib")
        staging_root = work_root / "report-walking-skeleton"
        staging_root.mkdir()

        for preset in PRESETS:
            print(f"regenerating {preset}")
            data_dir = work_root / "datasets" / preset
            run_dir = staging_root / "runs" / preset
            master_seed = int(DATASET_PRESETS[preset]["master_seed"])
            manifest = generate(preset, data_dir, master_seed=master_seed)
            frozen_hash = FROZEN_DATASET_HASHES.get(preset)
            if frozen_hash is not None and manifest["dataset_hash"] != frozen_hash:
                raise RuntimeError(
                    f"{preset} hash moved: expected {frozen_hash}, "
                    f"got {manifest['dataset_hash']}"
                )
            result = run(data_dir, run_dir)
            if tuple(result["methods"]) != RUN_METHODS:
                raise RuntimeError(
                    f"{preset} methods changed: expected {RUN_METHODS}, "
                    f"got {tuple(result['methods'])}"
                )
            validate_run_artifact(result)
            _write_json_lf(run_dir / "results.json", result)

        manifest_path = staging_root / "results-manifest.json"
        _write_json_lf(manifest_path, _report_manifest())
        report_paths = generate_report(
            manifest_path,
            staging_root,
            methods=REPORT_METHODS,
            bootstrap_seed=REPORT_BOOTSTRAP_SEED,
        )
        _write_json_lf(
            report_paths["trace"],
            json.loads(report_paths["trace"].read_text(encoding="utf-8")),
        )
        table_text = report_paths["table1"].read_text(encoding="utf-8")
        _write_lf(report_paths["table1"], table_text)
        table_pdf = _compile_table(report_paths["table1"], work_root / "latex")
        shutil.copyfile(table_pdf, staging_root / "table1.pdf")

        for preset in PRESETS:
            artifact_path = staging_root / "runs" / preset / "results.json"
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            validate_run_artifact(artifact)
        _publish(staging_root)

    relative_root = EXAMPLE_ROOT.relative_to(REPOSITORY_ROOT)
    print(f"wrote {len(EXPECTED_FILES)} files under {relative_root}")


def main() -> int:
    restarted = _restart_with_deterministic_hash_seed()
    if restarted is not None:
        return restarted
    _configure_environment()
    regenerate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
