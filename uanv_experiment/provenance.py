"""Run-layout and checkpoint-policy guards for future full experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


PRIMARY_POLICY = "fixed_epoch_35"
PRIMARY_CHECKPOINT = "epoch_35_checkpoint.pth"
BEST_VAL_CHECKPOINT = "best_val_checkpoint.pth"

REQUIRED_RUN_FILENAMES = (
    "config.json",
    "command.txt",
    "run_manifest.json",
    "log.csv",
    PRIMARY_CHECKPOINT,
    BEST_VAL_CHECKPOINT,
    "epoch_35_checkpoint_sha256.txt",
    "best_val_checkpoint_sha256.txt",
    "primary_metrics_threshold_0.5.csv",
    "exploratory_threshold_metrics.csv",
    "run_status.json",
)


@dataclass(frozen=True)
class RunLayout:
    run_directory: Path

    def path(self, filename: str) -> Path:
        if filename not in REQUIRED_RUN_FILENAMES:
            raise KeyError(f"unregistered run artifact: {filename}")
        return self.run_directory / filename

    @property
    def primary_checkpoint(self) -> Path:
        return self.path(PRIMARY_CHECKPOINT)

    @property
    def best_val_checkpoint(self) -> Path:
        return self.path(BEST_VAL_CHECKPOINT)

    def as_dict(self) -> dict[str, str]:
        return {name: str(self.path(name)) for name in REQUIRED_RUN_FILENAMES}


def assert_run_directory_available(run_directory: Path) -> None:
    if run_directory.exists():
        raise FileExistsError(
            f"run directory already exists; refusing overwrite: {run_directory}. "
            "Choose a new run ID/version."
        )


def build_new_run_layout(run_directory: Path) -> RunLayout:
    assert_run_directory_available(run_directory)
    layout = RunLayout(run_directory=run_directory)
    paths = list(layout.as_dict().values())
    if len(paths) != len(set(paths)):
        raise AssertionError("run artifact path collision detected")
    if layout.primary_checkpoint == layout.best_val_checkpoint:
        raise AssertionError("primary and best-validation checkpoint paths must differ")
    return layout


def validate_run_manifest(manifest: dict[str, object], layout: RunLayout) -> None:
    if manifest.get("primary_checkpoint_policy") != PRIMARY_POLICY:
        raise AssertionError("manifest primary checkpoint policy is not fixed_epoch_35")
    if manifest.get("primary_checkpoint") != str(layout.primary_checkpoint):
        raise AssertionError("primary evaluator must reference epoch_35_checkpoint.pth only")
    if manifest.get("best_validation_checkpoint") != str(layout.best_val_checkpoint):
        raise AssertionError("best-validation checkpoint path mismatch")
    if manifest["primary_checkpoint"] == manifest["best_validation_checkpoint"]:
        raise AssertionError("best-validation checkpoint cannot be primary")
    if manifest.get("primary_threshold") != 0.5:
        raise AssertionError("primary threshold must be 0.5")
    if manifest.get("primary_comparison_operator") != ">=":
        raise AssertionError("primary comparison operator must be >=")
    if manifest.get("primary_aggregation") != "sample_macro":
        raise AssertionError("primary aggregation must be sample_macro")


def resolve_primary_checkpoint(manifest_path: Path) -> Path:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_directory = manifest_path.parent
    layout = RunLayout(run_directory)
    validate_run_manifest(manifest, layout)
    return layout.primary_checkpoint
