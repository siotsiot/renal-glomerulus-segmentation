"""Explicit four-model registry for the clean comparison."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Callable

import torch.nn as nn

from architectures import UANVPaperInspiredAttentionUNet, UNext, VanillaUNet
from config import COMMON_TRAINING_CONFIG


@dataclass(frozen=True)
class ModelSpec:
    label: str
    display_name: str
    factory: Callable[[], nn.Module]
    architecture_family: str
    legacy_brb: bool = False


MODEL_REGISTRY = {
    "vanilla_unet": ModelSpec(
        label="vanilla_unet",
        display_name="VanillaUNet",
        factory=lambda: VanillaUNet(input_channels=3, num_classes=1),
        architecture_family="VanillaUNet",
    ),
    "unext": ModelSpec(
        label="unext",
        display_name="UNext",
        factory=lambda: UNext(input_channels=3, num_classes=1, img_size=512, use_refine_bottleneck=False),
        architecture_family="UNext",
    ),
    "unext_legacy_brb": ModelSpec(
        label="unext_legacy_brb",
        display_name="UNext + legacy BRB",
        factory=lambda: UNext(input_channels=3, num_classes=1, img_size=512, use_refine_bottleneck=True),
        architecture_family="UNext",
        legacy_brb=True,
    ),
    "uanv_paper_inspired_attention_unet": ModelSpec(
        label="uanv_paper_inspired_attention_unet",
        display_name="UANVPaperInspiredAttentionUNet",
        factory=lambda: UANVPaperInspiredAttentionUNet(input_channels=3, num_classes=1),
        architecture_family="VanillaUNet",
    ),
}


def get_model(label: str) -> nn.Module:
    try:
        return MODEL_REGISTRY[label].factory()
    except KeyError as exc:
        raise KeyError(f"unknown model label {label!r}; expected one of {tuple(MODEL_REGISTRY)}") from exc


def model_config(label: str) -> dict[str, object]:
    if label not in MODEL_REGISTRY:
        raise KeyError(label)
    config = deepcopy(COMMON_TRAINING_CONFIG)
    config["model_label"] = label
    config["architecture"] = MODEL_REGISTRY[label].architecture_family
    config["use_refine_bottleneck"] = MODEL_REGISTRY[label].legacy_brb
    return config


def assert_common_conditions() -> dict[str, object]:
    configs = {label: model_config(label) for label in MODEL_REGISTRY}
    architecture_keys = {"model_label", "architecture", "use_refine_bottleneck"}
    common_keys = set(COMMON_TRAINING_CONFIG)
    mismatches = {}
    for key in sorted(common_keys):
        values = {label: config[key] for label, config in configs.items()}
        if len({repr(value) for value in values.values()}) != 1:
            mismatches[key] = values
    if mismatches:
        raise AssertionError(f"non-architecture conditions differ: {mismatches}")
    return {
        "model_count": len(configs),
        "labels": list(configs),
        "common_keys_checked": sorted(common_keys),
        "architecture_specific_keys": sorted(architecture_keys),
        "all_common_conditions_identical": True,
    }
