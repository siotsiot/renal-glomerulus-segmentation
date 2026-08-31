"""Frozen common settings for the four-model clean experiment."""

from __future__ import annotations

from copy import deepcopy


COMMON_TRAINING_CONFIG = {
    "input_channels": 3,
    "num_classes": 1,
    "input_h": 512,
    "input_w": 512,
    "batch_size": 2,
    "epochs": 35,
    "loss": "WeightedBCEDiceLoss",
    "bce_weight": 0.5,
    "dice_weight": 1.0,
    "optimizer": "Adam",
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "scheduler": "CosineAnnealingLR",
    "minimum_learning_rate": 0.00001,
    "augmentation": True,
    "kfolds": 5,
    "kfold_seed": 42,
    "primary_checkpoint_policy": "fixed_epoch_35",
    "primary_checkpoint_epoch": 35,
    "primary_checkpoint_filename": "epoch_35_checkpoint.pth",
    "primary_threshold": 0.5,
    "primary_comparison_operator": ">=",
    "primary_aggregation": "sample_macro",
    "primary_metrics_filename": "primary_metrics_threshold_0.5.csv",
    "secondary_checkpoint_policy": "best_validation_dice_exploratory_only",
    "secondary_checkpoint_filename": "best_val_checkpoint.pth",
    "secondary_threshold_sweep": "exploratory_only",
}


def baseline_config():
    config = deepcopy(COMMON_TRAINING_CONFIG)
    config["architecture"] = "VanillaUNet"
    return config


def attention_config():
    config = deepcopy(COMMON_TRAINING_CONFIG)
    config["architecture"] = "UANVPaperInspiredAttentionUNet"
    return config


def unext_config(use_refine_bottleneck: bool = False):
    config = deepcopy(COMMON_TRAINING_CONFIG)
    config["architecture"] = "UNext"
    config["use_refine_bottleneck"] = bool(use_refine_bottleneck)
    return config


def compare_training_conditions():
    baseline = baseline_config()
    attention = attention_config()
    return {
        key: {"baseline": baseline[key], "attention": attention[key], "identical": baseline[key] == attention[key]}
        for key in baseline
    }
