import os
from typing import Iterable

import jax
import jax.numpy as jnp
import numpy as np


def latest_image(obs, image_key):
    image = np.asarray(obs[image_key]) if image_key in obs else None
    if image is None:
        return None
    if image.ndim == 4:
        image = image[-1]
    if image.ndim != 3:
        return None
    if image.shape[-1] > 3:
        image = image[..., -3:]
    return image


def select_gaze_image_key(obs, image_keys: Iterable[str], preferred_key="front_camera"):
    """Return the RGB image key used by the gaze predictor."""
    if preferred_key in image_keys and latest_image(obs, preferred_key) is not None:
        return preferred_key

    # TODO: If a future task uses gaze on a camera other than front_camera,
    # pass that camera key explicitly instead of relying on this fallback.
    for image_key in image_keys:
        if "tactile" in image_key:
            continue
        if latest_image(obs, image_key) is not None:
            return image_key
    return None


def infer_heatmap_shape(obs, image_keys, preferred_key="front_camera"):
    image_key = select_gaze_image_key(obs, image_keys, preferred_key)
    if image_key is None:
        return (1, 1)
    image = latest_image(obs, image_key)
    return int(image.shape[0]), int(image.shape[1])


def load_gaze_predictor(
    obs,
    image_keys,
    checkpoint_path,
    *,
    preferred_key="front_camera",
    encoder_variant="resnetv1-10",
    log_fn=print,
):
    """Load the frozen gaze heatmap predictor used to generate critic aux targets."""
    if not checkpoint_path:
        return None

    image_key = select_gaze_image_key(obs, image_keys, preferred_key)
    if image_key is None:
        log_fn(
            "Could not load gaze predictor: no RGB camera image was found in "
            f"image_keys={image_keys}."
        )
        return None

    image = latest_image(obs, image_key)
    from serl_launcher.networks.gaze_point_predictor import load_gaze_point_predictor_func

    sample_observations = {
        image_key: np.zeros(
            (1, image.shape[0], image.shape[1], image.shape[2]),
            dtype=np.float32,
        )
    }
    log_fn(
        "Loading frozen gaze predictor "
        f"checkpoint={checkpoint_path} image_key={image_key} encoder={encoder_variant}"
    )
    predictor_func = load_gaze_point_predictor_func(
        key=np.asarray([0, 0], dtype=np.uint32),
        sample_observations=sample_observations,
        image_keys=[image_key],
        checkpoint_path=os.path.abspath(checkpoint_path),
        encoder_variant=encoder_variant,
    )
    return predictor_func, image_key


def compute_gaze_heatmap_fields(obs, gaze_predictor, gaze_heatmap_shape):
    """Generate the gaze heatmap stored in replay transitions for CGL training."""
    gaze_heatmap = np.zeros(gaze_heatmap_shape, dtype=np.float32)
    if gaze_predictor is None:
        return {"gaze_heatmap": gaze_heatmap}

    gaze_predictor_func, image_key = gaze_predictor
    image = latest_image(obs, image_key)
    if image is None:
        return {"gaze_heatmap": gaze_heatmap}

    outputs = gaze_predictor_func({image_key: image[None].astype(np.float32)})
    gaze_heatmap = np.asarray(outputs["gaze_heat"][0], dtype=np.float32)
    if gaze_heatmap.shape != gaze_heatmap_shape:
        gaze_heatmap = np.asarray(
            jax.image.resize(
                jnp.asarray(gaze_heatmap)[None, ..., None],
                (1, *gaze_heatmap_shape, 1),
                method="bilinear",
            )[0, ..., 0],
            dtype=np.float32,
        )
    return {"gaze_heatmap": gaze_heatmap}


def gaze_xy_norm_from_heatmap(gaze_heatmap):
    gaze_heatmap = np.asarray(gaze_heatmap)
    while gaze_heatmap.ndim > 2:
        gaze_heatmap = gaze_heatmap[0]
    if gaze_heatmap.ndim != 2 or gaze_heatmap.size == 0:
        return None
    if not np.isfinite(gaze_heatmap).all() or float(np.max(gaze_heatmap)) <= 0.0:
        return None

    y, x = np.unravel_index(int(np.argmax(gaze_heatmap)), gaze_heatmap.shape)
    height, width = gaze_heatmap.shape
    x_norm = 0.0 if width <= 1 else float(x) / float(width - 1)
    y_norm = 0.0 if height <= 1 else float(y) / float(height - 1)
    return x_norm, y_norm


def update_env_gaze_prediction_overlay(env, gaze_heatmap, gaze_predictor):
    """Draw the frozen gaze predictor peak on the env RGB display, if supported."""
    try:
        set_overlay = env.unwrapped.set_gaze_prediction_overlay
    except Exception:
        return

    if gaze_predictor is None:
        set_overlay(xy_norm=None)
        return

    _, image_key = gaze_predictor
    xy_norm = gaze_xy_norm_from_heatmap(gaze_heatmap)
    if xy_norm is None:
        set_overlay(image_key=image_key, xy_norm=None)
        return
    set_overlay(image_key=image_key, xy_norm=xy_norm)


def ensure_optional_transition_fields(transition):
    transition = dict(transition)
    transition.setdefault("grasp_penalty", np.float32(0.0))
    transition.setdefault("robot_arm_penalty", np.float32(0.0))
    return transition


def ensure_gaze_transition_fields(transition, gaze_heatmap_shape):
    transition = dict(transition)
    transition.setdefault(
        "gaze_heatmap",
        np.zeros(gaze_heatmap_shape, dtype=np.float32),
    )
    return transition


def add_or_compute_gaze_transition_fields(transition, gaze_predictor, gaze_heatmap_shape):
    transition = ensure_optional_transition_fields(transition)
    if "gaze_heatmap" not in transition:
        transition.update(
            compute_gaze_heatmap_fields(
                transition["observations"],
                gaze_predictor,
                gaze_heatmap_shape,
            )
        )
    return ensure_gaze_transition_fields(transition, gaze_heatmap_shape)
