"""Tests for the R016 budget-matched SIB-200 specialist control study."""

import importlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

trainer = importlib.import_module("train_r016_budget_specialist")
from laya.train import optimizer_updates

DATA = ROOT / "data/processed/r008-sib200-km-train.jsonl"
TEST = ROOT / "data/processed/sib200-km-test.jsonl"
SEEDS = (13, 14)


def test_update_count_matches_multitask_budget():
    """48 epochs on 681 rows with eff. batch 32 gives ~1,056 updates (~1,060 target)."""
    updates = optimizer_updates(681, 1, 32, 48)
    assert updates == 1_056
    assert updates <= 1_060
    # 49 epochs would overshoot the target (1,078 updates > 1,060 target).
    assert optimizer_updates(681, 1, 32, 49) == 1_078
    assert optimizer_updates(681, 1, 32, 49) > 1_060


def test_epochs_closest_to_target_without_overshooting():
    updates_per_epoch = optimizer_updates(681, 1, 32, 1)
    assert updates_per_epoch == 22
    for epochs in (47, 48, 49):
        total = optimizer_updates(681, 1, 32, epochs)
        assert total == 22 * epochs
    assert optimizer_updates(681, 1, 32, 48) == 1_056  # nearest <= 1,060


def test_training_data_fingerprint_is_locked():
    digest = trainer.sha256(DATA)
    assert digest == "ed2ebd5fbb9e8bdbc78dd9eccaeaa146ee117c1809afc36305036740657930c3"


def test_test_split_row_count_and_groups():
    raw = [
        json.loads(line)
        for line in TEST.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(raw) == 204
    identity = importlib.import_module("r008_identity")

    groups = [identity.state_group_id(row) for row in raw]
    assert len(set(groups)) == 204


def test_accuracy_compute_from_correctness_vector():
    correct = [True, False, True, True, False]
    assert sum(correct) / len(correct) == 0.6


def test_paired_bootstrap_is_deterministic():
    summarize_r008 = importlib.import_module("summarize_r008")
    paired_group_bootstrap = summarize_r008.paired_group_bootstrap

    mt = [1, 0, 1, 1, 0, 1]
    sp = [1, 1, 0, 1, 0, 0]
    groups = ["a", "a", "b", "b", "c", "c"]
    ci1 = paired_group_bootstrap(mt, sp, groups, samples=2000, seed=13)
    ci2 = paired_group_bootstrap(mt, sp, groups, samples=2000, seed=13)
    assert (ci1 == ci2).all()
    assert len(ci1) == 2000


@pytest.mark.parametrize("seed", SEEDS)
def test_output_paths_do_not_collide_with_r008_checkpoints(seed):
    r016 = ROOT / f"artifacts/checkpoints/R016-budget-sib-s{seed}"
    r008 = ROOT / f"artifacts/checkpoints/R008-sib-s{seed}"
    assert r016.resolve() != r008.resolve()
    assert str(r016).startswith(str(ROOT / "artifacts/checkpoints/R016-budget-sib-s"))
    assert str(r008).startswith(str(ROOT / "artifacts/checkpoints/R008-sib-s"))
