"""Checkpoint and loss helpers shared by offline encoder pretraining and HIL-RL.

The old ``TaskCNNEncoderV2`` semantic-query CNN lived here. It has been
removed: that pipeline failed on the pick-and-place task and is superseded by
the ``vit-grounded`` encoder. Only the generic, reusable pieces remain.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Dict, Optional

from flax import serialization
from flax.core import FrozenDict, freeze, unfreeze
import jax
import jax.numpy as jnp


def load_encoder_checkpoint(checkpoint_path: str | Path) -> tuple[dict, int]:
    """Load model parameters written by examples/encoder_training/train_encoder.py."""
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Encoder checkpoint does not exist: {checkpoint_path}"
        )
    payload = serialization.msgpack_restore(checkpoint_path.read_bytes())
    if payload.get("format_version") != 1 or "params" not in payload:
        raise ValueError(
            f"Unsupported encoder checkpoint format: {checkpoint_path}"
        )
    params = jax.tree_util.tree_map(jnp.asarray, payload["params"])
    return params, int(payload.get("epoch", -1))


def load_encoder_checkpoint_config(checkpoint_path: str | Path) -> dict:
    """Load the standalone trainer config stored beside a CNN checkpoint."""
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    config_path = checkpoint_path.parent / "config.json"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Encoder checkpoint does not exist: {checkpoint_path}"
        )
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Encoder checkpoint config does not exist: {config_path}"
        )
    config = json.loads(config_path.read_text())
    if config.get("checkpoint_format") != "flax_msgpack_v1":
        raise ValueError(
            "Unsupported encoder checkpoint config format at "
            f"{config_path}: {config.get('checkpoint_format')!r}"
        )
    return config


def replace_named_param_subtree(
    params,
    *,
    module_name: str,
    replacement: dict,
    allow_partial: bool = False,
):
    """Replace every exactly named module subtree after validating shapes."""
    was_frozen = isinstance(params, FrozenDict)
    mutable = unfreeze(params) if was_frozen else copy.deepcopy(params)
    replaced_paths = []

    def visit(node, path=()):
        if not isinstance(node, dict):
            return
        for key in list(node):
            value = node[key]
            current_path = (*path, key)
            if key == module_name:
                if allow_partial:
                    candidate = copy.deepcopy(value)
                    replacement_keys = set(value).intersection(replacement)
                    if not replacement_keys:
                        raise ValueError(
                            "No matching checkpoint parameters at "
                            f"{'/'.join(current_path)}."
                        )
                    for replacement_key in replacement_keys:
                        current_shapes = jax.tree_util.tree_map(
                            lambda item: item.shape,
                            value[replacement_key],
                        )
                        replacement_shapes = jax.tree_util.tree_map(
                            lambda item: item.shape,
                            replacement[replacement_key],
                        )
                        if current_shapes != replacement_shapes:
                            raise ValueError(
                                "Checkpoint shape mismatch at "
                                f"{'/'.join((*current_path, replacement_key))}: "
                                f"initialized={current_shapes}, "
                                f"checkpoint={replacement_shapes}"
                            )
                        candidate[replacement_key] = copy.deepcopy(
                            replacement[replacement_key]
                        )
                elif set(value).issubset(replacement):
                    candidate = {
                        replacement_key: replacement[replacement_key]
                        for replacement_key in value
                    }
                else:
                    candidate = replacement
                current_shapes = jax.tree_util.tree_map(
                    lambda item: item.shape,
                    value,
                )
                candidate_shapes = jax.tree_util.tree_map(
                    lambda item: item.shape,
                    candidate,
                )
                if current_shapes != candidate_shapes:
                    raise ValueError(
                        "Encoder checkpoint shape mismatch at "
                        f"{'/'.join(current_path)}: initialized={current_shapes}, "
                        f"checkpoint={candidate_shapes}"
                    )
                node[key] = copy.deepcopy(candidate)
                replaced_paths.append("/".join(current_path))
            else:
                visit(value, current_path)

    visit(mutable)
    if not replaced_paths:
        raise KeyError(f"Could not find parameter module named {module_name!r}.")
    return (freeze(mutable) if was_frozen else mutable), tuple(replaced_paths)


def mask_supervision_loss(
    predicted_logits: jax.Array,
    target_masks: jax.Array,
    *,
    dice_weight: float = 1.0,
    channel_weights: Optional[jax.Array] = None,
) -> Dict[str, jax.Array]:
    """Phase-aware BCE + Dice loss matching the original trainer."""
    if predicted_logits.shape[:2] != target_masks.shape[:2]:
        raise ValueError(
            "Prediction and target must have the same batch/channel shape: "
            f"{predicted_logits.shape} vs {target_masks.shape}"
        )
    target_masks = jnp.asarray(target_masks, dtype=jnp.float32)
    output_height, output_width = predicted_logits.shape[2:]
    input_height, input_width = target_masks.shape[2:]
    if input_height % output_height == 0 and input_width % output_width == 0:
        block_height = input_height // output_height
        block_width = input_width // output_width
        target_masks = target_masks.reshape(
            target_masks.shape[0],
            target_masks.shape[1],
            output_height,
            block_height,
            output_width,
            block_width,
        ).mean(axis=(3, 5))
    else:
        target_masks = jax.image.resize(
            target_masks,
            (
                target_masks.shape[0],
                target_masks.shape[1],
                output_height,
                output_width,
            ),
            method="linear",
        )
    target_masks = jnp.clip(target_masks, 0.0, 1.0)
    if channel_weights is None:
        channel_weights = jnp.ones(
            predicted_logits.shape[:2],
            dtype=predicted_logits.dtype,
        )
    else:
        channel_weights = jnp.asarray(
            channel_weights,
            dtype=predicted_logits.dtype,
        )
    normalizer = jnp.maximum(jnp.sum(channel_weights), 1.0)
    bce_elementwise = (
        jnp.maximum(predicted_logits, 0.0)
        - predicted_logits * target_masks
        + jnp.log1p(jnp.exp(-jnp.abs(predicted_logits)))
    )
    bce_per_channel = jnp.mean(bce_elementwise, axis=(2, 3))
    bce = jnp.sum(bce_per_channel * channel_weights) / normalizer

    probabilities = jax.nn.sigmoid(predicted_logits)
    intersection = jnp.sum(probabilities * target_masks, axis=(2, 3))
    denominator = jnp.sum(probabilities, axis=(2, 3)) + jnp.sum(
        target_masks,
        axis=(2, 3),
    )
    dice_per_channel = 1.0 - (2.0 * intersection + 1e-6) / (
        denominator + 1e-6
    )
    dice = jnp.sum(dice_per_channel * channel_weights) / normalizer
    return {
        "loss": bce + dice_weight * dice,
        "bce": bce,
        "dice": dice,
    }
