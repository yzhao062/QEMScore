"""The campaign manifest: what it records, and what it notices afterwards.

A freeze that only writes a file proves nothing. What makes it useful is that
`verify` afterwards names every field that moved, so a reader can tell whether
the run that produced the numbers is the run the freeze describes.

Every field it compares is a recorded value against a recomputed one, and an
uncommitted edit is the one difference that comparison cannot see: the revision
string stays the same while the bytes behind it change. So the freeze refuses a
dirty tree rather than archiving the dirty bytes, which is the cheaper half of
that choice and the one that lets the analysis driver treat a verified manifest
as a checkable claim.
"""

import argparse
import json

import pytest

from tools.freeze_campaign import (
    SCHEMA_VERSION,
    _differences,
    _digest,
    _payload,
    freeze,
)


@pytest.fixture(scope="module")
def manifest(tmp_path_factory):
    root = tmp_path_factory.mktemp("freeze")
    payload = _payload(root, root)
    payload["manifest_sha256"] = _digest(payload)
    return payload


def test_the_manifest_records_the_frozen_design(manifest):
    design = manifest["design"]

    assert manifest["schema_version"] == SCHEMA_VERSION
    assert design["seeds"] == [101, 211, 307]
    assert design["sizes"] == [160, 640]
    assert design["primary_size"] == 640
    assert design["n_resamples"] == 10_000
    assert design["root_seed"] == 20260904
    assert design["ratio_margin"] == 1.05
    assert design["required_families"] == 2
    assert len(manifest["expected_settings"]) == 12
    assert manifest["role_counts"]["640"] == {
        "train": 1280, "validation": 640, "test": 320}
    json.dumps(manifest, allow_nan=False)


def test_the_manifest_records_the_audit_rules(manifest):
    rules = manifest["audit_rules"]

    assert rules["indices"]["train"] == [0, 79, 159, 160, 399, 639]
    assert rules["tolerances"]["label"] == 1e-12
    assert rules["tolerances"]["prediction"] == 1e-9
    assert "histogram_replay" in rules["exact_comparisons"]
    assert rules["check_coverage"]["independent_label"] == "independent"


def test_the_manifest_records_the_environment_and_the_revision(manifest):
    environment = manifest["environment"]

    assert environment["python"].startswith("3.")
    assert set(environment["packages"]) >= {"numpy", "qiskit", "scikit-learn"}
    assert environment["packages"]["numpy"] is not None
    assert "revision" in manifest["code"]
    assert isinstance(manifest["code"]["clean"], bool)


def test_the_content_hash_covers_the_body_and_not_the_timestamp(manifest):
    assert manifest["manifest_sha256"] == _digest(manifest)

    moved = dict(manifest, frozen_utc="1999-01-01T00:00:00Z")
    assert _digest(moved) == manifest["manifest_sha256"]

    changed = json.loads(json.dumps(manifest))
    changed["design"]["n_resamples"] = 2000
    assert _digest(changed) != manifest["manifest_sha256"]


def test_verification_names_every_field_that_moved(manifest):
    current = json.loads(json.dumps(manifest))
    current["design"]["seeds"] = [101, 211, 999]
    current["audit_rules"]["tolerances"]["label"] = 1e-9
    current["code"]["revision"] = "deadbeef"

    drift = _differences(
        {key: value for key, value in manifest.items()
         if key not in ("frozen_utc", "manifest_sha256")},
        {key: value for key, value in current.items()
         if key not in ("frozen_utc", "manifest_sha256")},
    )
    fields = {entry["field"] for entry in drift}

    assert fields == {"design.seeds", "audit_rules.tolerances.label",
                      "code.revision"}
    seeds = next(entry for entry in drift if entry["field"] == "design.seeds")
    assert seeds["frozen"] == [101, 211, 307]
    assert seeds["current"] == [101, 211, 999]


def test_an_unchanged_manifest_reports_no_drift(manifest):
    body = {key: value for key, value in manifest.items()
            if key not in ("frozen_utc", "manifest_sha256")}

    assert _differences(body, json.loads(json.dumps(body))) == []


def _freeze(tmp_path, monkeypatch, code):
    """Freeze into a scratch root with the working-tree state stated outright.

    The revision is stubbed rather than read from a fixture repository, because
    what is under test is the refusal, and a fixture repository would make the
    test depend on a git binary to establish a condition the guard reads from
    one dictionary.
    """
    monkeypatch.setattr(
        "tools.freeze_campaign._code_revision", lambda repository: code)
    return freeze(argparse.Namespace(
        root=tmp_path, repository=tmp_path,
        out=tmp_path / "campaign-manifest.json"))


def test_the_freeze_refuses_a_dirty_working_tree(tmp_path, monkeypatch):
    """A dirty revision names bytes nobody can check out again.

    It used to warn and write the file anyway, which left the manifest naming a
    revision that never existed and `verify` comparing that name against itself.
    """
    with pytest.raises(SystemExit, match="clean committed revision"):
        _freeze(tmp_path, monkeypatch, {
            "revision": "abc123", "clean": False,
            "uncommitted_paths": ["qemscore/campaign/design.py"]})
    assert not (tmp_path / "campaign-manifest.json").exists()


def test_the_freeze_refuses_a_tree_with_no_resolvable_revision(
    tmp_path, monkeypatch,
):
    """Outside a repository there is no revision to freeze, only an empty one."""
    with pytest.raises(SystemExit, match="clean committed revision"):
        _freeze(tmp_path, monkeypatch, {
            "revision": None, "clean": False, "uncommitted_paths": []})


def test_a_clean_revision_is_frozen_and_hashed(tmp_path, monkeypatch):
    assert _freeze(tmp_path, monkeypatch, {
        "revision": "abc123", "clean": True, "uncommitted_paths": []}) == 0

    written = json.loads(
        (tmp_path / "campaign-manifest.json").read_text(encoding="utf-8"))
    assert written["code"] == {"revision": "abc123", "clean": True,
                               "uncommitted_paths": []}
    assert written["manifest_sha256"] == _digest(written)


def test_a_field_that_appears_or_disappears_is_named(manifest):
    body = {key: value for key, value in manifest.items()
            if key not in ("frozen_utc", "manifest_sha256")}
    without = json.loads(json.dumps(body))
    del without["audit_rules"]["indices"]["test"]
    without["audit_rules"]["indices"]["extra"] = [0]

    fields = {entry["field"] for entry in _differences(body, without)}

    assert fields == {"audit_rules.indices.test", "audit_rules.indices.extra"}
