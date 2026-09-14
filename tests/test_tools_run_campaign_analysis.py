"""The analysis driver's bindings: the rows it read, and the fit that made them.

`analyze` regenerates every reported statistic from records, and a record's
identity was the dataset hash it asserted and nothing else. Nothing checked that
the rows it retained are that dataset's whole test projection, so dropping one
observable from every circuit changed the improvement share while leaving the
cross-size circuit comparison satisfied, every prediction value untouched and
every reported dataset hash the same.

The rows are half of a record. Round 10 showed the other half: copying one saved
method's prediction map into another's slot moved the share from 1.03 to 0 or to
1 with the rows, the dataset hash, the code revision and the named
configurations all unchanged, because those outputs exist nowhere but the record
and `analyze` never refits. The receipt each frozen fit writes is what binds
them, and the pre-fit freeze is what the receipt is written under.

The artifact here is generated rather than assembled, because the comparison is
against `validate_split_artifact` output; a hand-built directory would only test
the comparison against itself.
"""

import argparse
import copy
import json

import pytest

from qemscore.campaign.analysis import RECORD_SCHEMA_VERSION, TEST_ROW_FIELDS
from qemscore.campaign.design import campaign_split_spec, setting_key
from qemscore.datasets.split_generate import generate_split
from qemscore.validation import validate_split_artifact
from tools.freeze_campaign import SCHEMA_VERSION as FREEZE_SCHEMA_VERSION
from tools.run_campaign_analysis import (
    _assert_fit_bindings,
    _assert_records_retain_whole_datasets,
    _code_revision,
    _data_dir,
    _fit_binding,
    _fit_binding_path,
    _verified_campaign_manifest,
    _write_json,
)

REGIME = "shipped"
SEED = 999
SIZE = 2
REHEARSAL = {"train": SIZE, "validation": 2, "test": 2}


@pytest.fixture(scope="module")
def campaign_root(tmp_path_factory):
    """One rehearsal-shaped setting, laid out the way the driver lays it out."""
    root = tmp_path_factory.mktemp("campaign-root")
    target = _data_dir(root, setting_key(REGIME, SEED, SIZE))
    target.parent.mkdir(parents=True, exist_ok=True)
    generate_split(
        campaign_split_spec(REGIME, SIZE, counts=REHEARSAL), target,
        master_seed=SEED)
    return root


@pytest.fixture(scope="module")
def whole_record(campaign_root):
    """A record retaining the artifact's complete test projection.

    Only the fields the binding reads are filled in. Fitting four arms to
    exercise a comparison between two row projections would cost minutes and
    say nothing the projection does not.
    """
    key = setting_key(REGIME, SEED, SIZE)
    rows, manifest = validate_split_artifact(_data_dir(campaign_root, key))
    return {
        "schema_version": RECORD_SCHEMA_VERSION,
        "setting": key,
        "regime": REGIME,
        "seed": SEED,
        "size": SIZE,
        "dataset_hash": str(manifest["dataset_hash"]),
        "test_items": [
            {field: row[field] for field in TEST_ROW_FIELDS}
            for row in rows if row["split"] == "test"
        ],
    }


def test_a_record_retaining_the_whole_test_split_is_accepted(
    campaign_root, whole_record,
):
    """The check has to admit the record the fit writes."""
    observables = {row["observable"] for row in whole_record["test_items"]}

    assert observables == {"z_mid", "zz_mid"}
    _assert_records_retain_whole_datasets(campaign_root, [whole_record])


def test_a_record_missing_one_observable_is_refused(campaign_root, whole_record):
    """Round 9's probe: dropping zz_mid moved the share from 0.50 to 0.9375.

    Filtering both sizes the same way leaves the cross-size circuit comparison
    satisfied, and the freeze records neither the retained rows nor their count,
    so the artifact is the only thing that can still notice.
    """
    filtered = copy.deepcopy(whole_record)
    filtered["test_items"] = [
        row for row in filtered["test_items"] if row["observable"] == "z_mid"]
    assert filtered["test_items"]

    with pytest.raises(ValueError, match="retained test rows differ"):
        _assert_records_retain_whole_datasets(campaign_root, [filtered])


def test_a_record_whose_retained_row_was_relabeled_is_refused(
    campaign_root, whole_record,
):
    """A row kept under its own item_id still has to be that row."""
    edited = copy.deepcopy(whole_record)
    edited["test_items"][0]["noisy_expectation"] += 0.5

    with pytest.raises(ValueError, match="retained test rows differ"):
        _assert_records_retain_whole_datasets(campaign_root, [edited])


def test_a_record_that_repeats_one_row_is_refused(campaign_root, whole_record):
    """Two rows under one item_id collapse in the comparison, not in the record.

    Without the count check the duplicate is invisible: the two maps still agree
    while the record scores that circuit's cell twice.
    """
    duplicated = copy.deepcopy(whole_record)
    duplicated["test_items"].append(dict(duplicated["test_items"][0]))

    with pytest.raises(ValueError, match="retained test rows differ"):
        _assert_records_retain_whole_datasets(campaign_root, [duplicated])


def test_a_record_naming_another_dataset_is_refused(campaign_root, whole_record):
    other = copy.deepcopy(whole_record)
    other["dataset_hash"] = "sha256:somewhere-else"

    with pytest.raises(ValueError, match="names a different dataset"):
        _assert_records_retain_whole_datasets(campaign_root, [other])


def test_a_record_whose_fields_disagree_with_its_key_is_refused(
    campaign_root, whole_record,
):
    """The key chooses the artifact, so a disagreeing key reads another one."""
    renamed = copy.deepcopy(whole_record)
    renamed["setting"] = setting_key("large", SEED, SIZE)

    with pytest.raises(ValueError, match="disagrees with its fields"):
        _assert_records_retain_whole_datasets(campaign_root, [renamed])


# --------------------------------------------------------------------------
# The fitted half: what the fit produced, under the freeze it ran against
# --------------------------------------------------------------------------

MANIFEST = {"manifest_sha256": "a" * 64, "code": {"clean": True,
                                                  "revision": "abc123"}}


@pytest.fixture
def fitted_record(whole_record):
    """The record with the outputs a fit would have written into it.

    The receipt covers the whole record, so any field would exercise it. These
    are the fields round 10 moved: the prediction map the share reads and the
    configuration the record names as the model behind it.
    """
    return dict(whole_record, code_revision="abc123", test_predictions={
        "feat-only": {row["item_id"]: 0.10 for row in whole_record["test_items"]},
        "liao-feat-only": {row["item_id"]: 0.04
                           for row in whole_record["test_items"]},
        "liao": {row["item_id"]: 0.02 for row in whole_record["test_items"]},
    }, selected_configs={"liao-feat-only": {"selected_model": "mlp"}})


@pytest.fixture
def sealed_root(tmp_path, fitted_record):
    """A campaign root holding the receipt this record's fit would have left."""
    _write_json(_fit_binding_path(tmp_path, fitted_record["setting"]),
                _fit_binding(fitted_record, MANIFEST))
    return tmp_path


def test_a_record_that_matches_its_fit_receipt_is_accepted(
    sealed_root, fitted_record,
):
    """The check has to admit the record the fit that wrote the receipt made."""
    _assert_fit_bindings(sealed_root, [fitted_record], MANIFEST)


def test_a_prediction_map_copied_from_another_method_is_refused(
    sealed_root, fitted_record,
):
    """Round 10's probe: A's saved map in C's slot moved the share to 0.

    Nothing else in the record moves. The rows, the dataset hash, the code
    revision and the named configurations are the ones the fit wrote, so the row
    guard, the roster binding and the audit all still agree.
    """
    swapped = copy.deepcopy(fitted_record)
    swapped["test_predictions"]["liao-feat-only"] = copy.deepcopy(
        swapped["test_predictions"]["feat-only"])

    with pytest.raises(SystemExit, match="differs from its frozen-fit receipt"):
        _assert_fit_bindings(sealed_root, [swapped], MANIFEST)


def test_a_relabeled_selected_configuration_is_refused(
    sealed_root, fitted_record,
):
    """Renaming the model behind a prediction moved no number and no hash.

    It breaks the record's account of where its predictions came from, which is
    the claim the paper makes about them.
    """
    relabeled = copy.deepcopy(fitted_record)
    relabeled["selected_configs"]["liao-feat-only"]["selected_model"] = (
        "never-fitted")

    with pytest.raises(SystemExit, match="differs from its frozen-fit receipt"):
        _assert_fit_bindings(sealed_root, [relabeled], MANIFEST)


def test_a_record_without_a_receipt_is_refused(tmp_path, fitted_record):
    """A receipt made here from the record present here authenticates nothing.

    An unsealed record is refused rather than sealed on arrival, so a campaign
    fitted before this existed has to be refitted rather than grandfathered.
    """
    with pytest.raises(SystemExit, match="missing receipt from the frozen fit"):
        _assert_fit_bindings(tmp_path, [fitted_record], MANIFEST)


def test_a_receipt_written_under_another_freeze_is_refused(
    sealed_root, fitted_record,
):
    """The receipt names the freeze it was written under, and that is checked.

    Without the link a receipt from any run would satisfy any analysis, and the
    pre-fit commitment the manifest carries would reach the results through
    nothing.
    """
    other = dict(MANIFEST, manifest_sha256="b" * 64)

    with pytest.raises(SystemExit, match="differs from its frozen-fit receipt"):
        _assert_fit_bindings(sealed_root, [fitted_record], other)


def test_a_campaign_without_its_freeze_is_refused(tmp_path):
    """An absent manifest was ignored; every reported number rode on nothing."""
    with pytest.raises(SystemExit, match="requires its pre-fit freeze"):
        _verified_campaign_manifest(tmp_path)


def test_a_malformed_freeze_is_refused_rather_than_ignored(tmp_path):
    """A file that cannot be read is not a freeze that passed."""
    (tmp_path / "campaign-manifest.json").write_text("{malformed freeze")

    with pytest.raises(json.JSONDecodeError):
        _verified_campaign_manifest(tmp_path)


def test_a_freeze_recording_a_dirty_revision_is_refused(tmp_path):
    """A dirty revision cannot be checked out again, so it froze nothing."""
    _write_json(tmp_path / "campaign-manifest.json", {
        "schema_version": FREEZE_SCHEMA_VERSION,
        "code": {"clean": False, "revision": "abc123"},
    })

    with pytest.raises(SystemExit, match="clean frozen revision"):
        _verified_campaign_manifest(tmp_path)


def test_a_freeze_that_no_longer_matches_its_own_hash_is_refused(tmp_path):
    """`verify` reports drift; this is the layer that acts on the report."""
    _write_json(tmp_path / "campaign-manifest.json", {
        "schema_version": FREEZE_SCHEMA_VERSION,
        "code": {"clean": True, "revision": "abc123"},
        "manifest_sha256": "0" * 64,
    })

    with pytest.raises(SystemExit, match="differs from its recorded freeze"):
        _verified_campaign_manifest(tmp_path)


def test_an_explicit_code_revision_can_only_agree_with_the_tree():
    """The flag used to return before git ran, which is what made it an override.

    Passing the current HEAD then erased the `-dirty` suffix the tree had
    earned, and every restriction that reads the suffix went with it.
    """
    resolved = _code_revision(argparse.Namespace(code_revision=None))

    assert _code_revision(argparse.Namespace(code_revision=resolved)) == resolved
    with pytest.raises(SystemExit, match="disagrees with the actual working tree"):
        _code_revision(argparse.Namespace(
            code_revision=resolved.removesuffix("-dirty")
            if resolved.endswith("-dirty") else f"{resolved}-dirty"))
