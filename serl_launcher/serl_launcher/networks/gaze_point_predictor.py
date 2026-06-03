from typing import Callable, Dict, List, Tuple

import jax
from jax import numpy as jnp
import flax.linen as nn
from flax.core import FrozenDict
from flax.core.frozen_dict import freeze, unfreeze
from flax.training import checkpoints
from flax.training.train_state import TrainState
import numpy as np
import optax

from serl_launcher.utils.train_utils import load_resnet10_encoder_params
from serl_launcher.vision.resnet_v1 import resnetv1_configs


class GazeHeatmapPredictor(nn.Module):
    """Predict a per-pixel gaze heatmap from image observations."""

    encoder_defs: Dict[str, nn.Module]
    image_keys: List[str]
    heatmap_size: Tuple[int, int]
    head_channels: int = 128

    def encode(self, observations, train=False):
        encoded = []
        for image_key in self.image_keys:
            encoded.append(self.encoder_defs[image_key](observations[image_key], train=train))
        return jnp.concatenate(encoded, axis=-1)

    @nn.compact
    def __call__(self, observations, train=False):
        encoded = self.encode(observations, train=train)

        x = nn.Conv(self.head_channels, (3, 3), padding="SAME", name="heatmap_conv_0")(encoded)
        x = nn.GroupNorm(num_groups=8, epsilon=1e-5, name="heatmap_norm_0")(x)
        x = nn.relu(x)
        x = nn.Conv(self.head_channels // 2, (3, 3), padding="SAME", name="heatmap_conv_1")(x)
        x = nn.GroupNorm(num_groups=8, epsilon=1e-5, name="heatmap_norm_1")(x)
        x = nn.relu(x)
        heatmap_logits = nn.Conv(1, (1, 1), padding="SAME", name="heatmap_logits")(x)
        batch_size = heatmap_logits.shape[0]
        target_h, target_w = self.heatmap_size
        heatmap_logits = jax.image.resize(
            heatmap_logits,
            (batch_size, target_h, target_w, 1),
            method="bilinear",
        )[..., 0]

        return {"heatmap_logits": heatmap_logits}


def heatmap_confidence_from_logits(
    logits: jnp.ndarray,
    peak_mass_pixels: int = 64,
    eps: float = 1e-8,
):
    """Estimate label-free gaze confidence from peak-localized heatmap mass."""
    batch_size = logits.shape[0]
    heatmap = jax.nn.sigmoid(logits).reshape(batch_size, -1)
    saliency = heatmap - jnp.min(heatmap, axis=-1, keepdims=True)
    saliency = jnp.maximum(saliency, 0.0)
    num_pixels = saliency.shape[-1]
    k = min(int(peak_mass_pixels), int(num_pixels))
    topk_values, _ = jax.lax.top_k(saliency, k)
    topk_mass = jnp.sum(topk_values, axis=-1)
    total_mass = jnp.sum(saliency, axis=-1)
    uniform_mass = float(k) / float(num_pixels)
    peak_mass_ratio = topk_mass / (total_mass + eps)
    calibrated = (peak_mass_ratio - uniform_mass) / max(1.0 - uniform_mass, eps)
    return jnp.clip(calibrated, 0.0, 1.0)


def _infer_heatmap_size(sample_observations: Dict[str, jnp.ndarray], image_keys: List[str]):
    sample_image = sample_observations[image_keys[0]]
    return int(sample_image.shape[-3]), int(sample_image.shape[-2])


def _build_spatial_encoders(image_keys: List[str], encoder_variant: str):
    if encoder_variant == "resnetv1-10-frozen":
        encoder_ctor = resnetv1_configs["resnetv1-10-frozen"]
        encoder_kwargs = {"pre_pooling": True}
    elif encoder_variant == "resnetv1-18-frozen":
        encoder_ctor = resnetv1_configs["resnetv1-18-frozen"]
        encoder_kwargs = {"pre_pooling": True}
    elif encoder_variant == "resnetv1-10":
        encoder_ctor = resnetv1_configs["resnetv1-10"]
        encoder_kwargs = {"pooling_method": "none", "bottleneck_dim": None}
    else:
        raise ValueError(
            f"Unsupported encoder_variant={encoder_variant}. "
            "Expected one of ['resnetv1-10-frozen', 'resnetv1-18-frozen', 'resnetv1-10']."
        )
    return {
        image_key: encoder_ctor(
            name=f"encoder_{image_key}",
            **encoder_kwargs,
        )
        for image_key in image_keys
    }


def _safe_merge(dst, src):
    if isinstance(dst, dict) and isinstance(src, dict):
        for key, value in src.items():
            if key in dst:
                dst[key] = _safe_merge(dst[key], value)
        return dst
    if isinstance(dst, np.ndarray) and isinstance(src, np.ndarray) and dst.shape == src.shape:
        return src
    return dst


def _load_backbone_params(params_tree, image_keys: List[str], encoder_variant: str):
    if encoder_variant == "resnetv1-18-frozen":
        print(
            "[warn] encoder_variant=resnetv1-18-frozen has no matching ResNet-18 pretrained "
            "weights wired up in this repo. The frozen ResNet-18 backbone will stay randomly "
            "initialized, so this run is only useful as a quick diagnostic."
        )
        return freeze(params_tree) if not isinstance(params_tree, FrozenDict) else params_tree

    encoder_params = load_resnet10_encoder_params()
    new_params = unfreeze(params_tree) if isinstance(params_tree, FrozenDict) else params_tree
    for image_key in image_keys:
        encoder_scope = None
        if "encoder_defs" in new_params:
            encoder_scope = new_params["encoder_defs"].get(f"encoder_{image_key}")
        if encoder_scope is None:
            encoder_scope = new_params.get(f"encoder_{image_key}")
        if encoder_scope is None:
            print(f"[warn] encoder scope not found under encoder_{image_key}")
            continue
        _safe_merge(encoder_scope, encoder_params)
    return freeze(new_params)


def create_gaze_point_predictor(
    key: jnp.ndarray,
    sample_observations: Dict[str, jnp.ndarray],
    image_keys: List[str],
    learning_rate: float = 1e-4,
    encoder_variant: str = "resnetv1-10",
) -> TrainState:
    heatmap_size = _infer_heatmap_size(sample_observations, image_keys)
    predictor_def = GazeHeatmapPredictor(
        encoder_defs=_build_spatial_encoders(image_keys, encoder_variant),
        image_keys=image_keys,
        heatmap_size=heatmap_size,
    )
    params = predictor_def.init(key, sample_observations)["params"]
    params = unfreeze(_load_backbone_params(params, image_keys, encoder_variant))
    return TrainState.create(
        apply_fn=predictor_def.apply,
        params=params,
        tx=optax.adam(learning_rate),
    )


def load_gaze_point_predictor_func(
    key: jnp.ndarray,
    sample_observations: Dict[str, jnp.ndarray],
    image_keys: List[str],
    checkpoint_path: str,
    encoder_variant: str = "resnetv1-10",
) -> Callable[[Dict[str, jnp.ndarray]], jnp.ndarray]:
    state = create_gaze_point_predictor(
        key,
        sample_observations,
        image_keys,
        encoder_variant=encoder_variant,
    )
    resolved_checkpoint = checkpoints.latest_checkpoint(checkpoint_path)
    if resolved_checkpoint is None:
        raise FileNotFoundError(
            "No gaze heatmap checkpoint was found. "
            f"checkpoint_path={checkpoint_path}. "
            "Please train the model first or pass the correct checkpoint directory."
        )
    print(f"[gaze-heatmap][load] checkpoint={resolved_checkpoint}")
    state = checkpoints.restore_checkpoint(resolved_checkpoint, target=state)

    def _predict(observations):
        outputs = state.apply_fn({"params": state.params}, observations, train=False)
        heatmap_logits = outputs["heatmap_logits"]
        gaze_conf = heatmap_confidence_from_logits(heatmap_logits)
        return {
            "gaze_heat": jax.nn.sigmoid(heatmap_logits),
            "gaze_conf": gaze_conf,
        }

    return jax.jit(_predict)
