"""Round-8 plumbing: cache records for copied trees and unpinned stacking sets."""

from __future__ import annotations

import json
from pathlib import Path
import pickle
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import descriptor_common as common  # noqa: E402
import fresh_panel  # noqa: E402
import stacking_increment_all as sia  # noqa: E402
from tools import descriptor_ladder  # noqa: E402


def _tree(root: Path) -> Path:
    (root / "cache").mkdir(parents=True)
    for seed in (401, 503):
        with open(root / "cache" / f"shipped-s{seed}-n640.pkl", "wb") as handle:
            pickle.dump({"key": f"shipped-s{seed}-n640", "seed": seed}, handle)
    return root


def test_record_caches_round_trip_and_refusal(tmp_path):
    tree = _tree(tmp_path / "tree")
    record = tmp_path / "record.json"
    assert fresh_panel.main(["record-caches", "--tree", str(tree), "--out", str(record)]) == 0
    payload = json.loads(record.read_text())
    assert set(payload["inputs"]["caches_used"]) == {"shipped-s401-n640", "shipped-s503-n640"}
    keys = ["shipped-s401-n640", "shipped-s503-n640"]
    digests = common.check_caches_recorded(tree / "cache", keys, record_path=record)
    assert all(d["sha256"] == d["recorded_sha256"] for d in digests.values())
    with open(tree / "cache" / "shipped-s503-n640.pkl", "ab") as handle:
        handle.write(b"x")
    with pytest.raises(SystemExit, match="cache SHA-256 check failed"):
        common.check_caches_recorded(tree / "cache", keys, record_path=record)
    with pytest.raises(SystemExit, match="no digest recorded"):
        common.check_caches_recorded(tree / "cache", ["shipped-s607-n640"], record_path=record)


def test_record_caches_refuses_an_empty_tree(tmp_path):
    (tmp_path / "cache").mkdir()
    with pytest.raises(SystemExit, match="no primary caches"):
        fresh_panel.main(["record-caches", "--tree", str(tmp_path), "--out", str(tmp_path / "r.json")])


def test_run_accepts_cache_record_option(tmp_path, monkeypatch):
    seen = {}

    def fake_run(args):
        seen["cache_record"] = args.cache_record
        return 0

    monkeypatch.setattr(descriptor_ladder, "cmd_run", fake_run)
    descriptor_ladder.main(["run", "--datasets", str(tmp_path), "--archive", str(tmp_path),
                            "--out", str(tmp_path), "--cache-record", str(tmp_path / "r.json")])
    assert seen["cache_record"] == tmp_path / "r.json"


def test_stacking_requires_exactly_one_of_pins_or_unpinned(tmp_path):
    base = ["--set", "a", "f", "d", "c", "x", "--out", str(tmp_path / "o.json")]
    with pytest.raises(SystemExit):
        sia.main(base)
    with pytest.raises(SystemExit):
        sia.main(base + ["--unpinned", "--pins", str(tmp_path / "p.json")])


def _run_g_analyze(tmp_path: Path, drop: str | None = None):
    """Runs the G_analyze stage with a stub interpreter that records each command."""
    import subprocess
    repo = Path(__file__).resolve().parents[1]
    sw_fits = tmp_path / "sweep" / "runs" / "shots-2048" / "fits"
    strong_fits = tmp_path / "strong" / "runs" / "derived-baseline" / "fits"
    sw_fits.mkdir(parents=True)
    strong_fits.mkdir(parents=True)
    for s in (101, 211, 307):
        for rung, where in (("R0", sw_fits), ("N1", sw_fits), ("N2", sw_fits), ("R5", sw_fits),
                            ("R4", strong_fits), ("N3", sw_fits)):
            for ext in ("json", "npz"):
                (where / f"shipped-s{s}-n640__{rung}__A.{ext}").write_text("")
            for k in range(1, 21):
                for arm in ("C", "F", "P"):
                    for ext in ("json", "npz"):
                        (where / f"shipped-s{s}-n640__{rung}__k{k:02d}__{arm}.{ext}").write_text("")
        # The derived baseline also holds other rungs; only R4 may be linked from it.
        (strong_fits / f"shipped-s{s}-n640__R0__k01__C.json").write_text("strong")
    if drop:
        (strong_fits / drop).unlink()
    log = tmp_path / "commands.log"
    stub = tmp_path / "py.sh"
    stub.write_text(f'#!/usr/bin/env bash\necho "$*" >> {log}\n')
    stub.chmod(0o755)
    env = {"PATH": "/usr/bin:/bin", "ROOT": str(tmp_path / "root"), "CODE": str(repo),
           "SW": str(tmp_path / "sweep"), "STRONG": str(tmp_path / "strong"), "PY": str(stub),
           "COMMIT": "test"}
    script = repo / "artifacts/descriptor-information/round8/scripts/round8_stage.sh"
    res = subprocess.run(["bash", str(script), "G_analyze"], env=env, capture_output=True, text=True)
    return res, log, tmp_path / "root" / "follow" / "G" / "reference-640" / "fits"


def test_g_analyze_passes_the_n640_reference(tmp_path):
    res, log, ref = _run_g_analyze(tmp_path)
    assert res.returncode == 0, res.stderr
    lines = [l for l in log.read_text().splitlines() if l.startswith("tools/descriptor_ladder_analysis.py")]
    assert len(lines) == 4
    assert all(f"--original-fits {ref}" in l for l in lines)
    linked = sorted(p.name for p in ref.iterdir())
    # As in the released inventory: per dataset and rung, A (2 files) and C, F, P at 20 seeds
    # (120 files), so 3 x 5 x 122 = 1,830 links, 1,200 of them C and F.
    assert len(linked) == 3 * 5 * (2 + 20 * 3 * 2)
    required = {f"shipped-s{s}-n640__{r}__k{k:02d}__{a}.{e}" for s in (101, 211, 307)
                for r in ("R0", "N1", "N2", "R4", "R5") for k in range(1, 21)
                for a in ("C", "F") for e in ("json", "npz")}
    assert required <= set(linked) and len(required) == 1200
    for n in linked:
        source = str((ref / n).resolve())
        assert ("strong" in source) == ("__R4__" in n)
    assert not any("__N3__" in n for n in linked)
    # Every R0 link points at the shot sweep's fit, never at the derived baseline.
    assert all("sweep" in str((ref / n).resolve()) for n in linked if "__R0__" in n)


def test_g_analyze_stops_on_a_missing_reference_fit(tmp_path):
    res, log, _ = _run_g_analyze(tmp_path, drop="shipped-s211-n640__R4__k07__F.npz")
    assert res.returncode == 5
    assert not log.exists()
