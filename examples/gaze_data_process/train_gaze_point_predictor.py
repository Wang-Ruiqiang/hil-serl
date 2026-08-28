#!/usr/bin/env python3
from __future__ import annotations

import glob
import os
import pickle as pkl
import sys
from pathlib import Path

import jax
from jax import numpy as jnp
import numpy as np
from absl import app, flags
from flax.training import checkpoints
from tqdm import tqdm


project_root = next(
    p for p in Path(__file__).resolve().parents if (p / "serl_launcher").exists()
)
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "serl_launcher"))
sys.path.insert(0, str(project_root / "serl_robot_infra"))
sys.path.insert(0, str(project_root / "examples"))

from serl_launcher.networks.gaze_point_predictor import create_gaze_point_predictor


FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", "tennis_ball_pick", "Experiment name used to choose image keys.")
flags.DEFINE_boolean("enable_tactile", False, "Use the task config's tactile classifier keys if available.")
flags.DEFINE_string("image_keys", "", "Comma-separated override. Empty means use task config classifier_keys.")
flags.DEFINE_string(
    "train_data_dir",
    str((Path(__file__).resolve().parent / "gaze_cls_data" / "train")),
    "Directory with training pkl shards.",
)
flags.DEFINE_string(
    "val_data_dir",
    str((Path(__file__).resolve().parent / "gaze_cls_data" / "val")),
    "Directory with validation pkl shards.",
)
flags.DEFINE_string(
    "checkpoint_dir",
    str((Path(__file__).resolve().parent / "gaze_heatmap_ckpt")),
    "Directory for gaze heatmap checkpoints.",
)
flags.DEFINE_integer("image_width", 128, "Input image width.")
flags.DEFINE_integer("image_height", 128, "Input image height.")
flags.DEFINE_integer("num_epochs", 30, "Number of training epochs.")
flags.DEFINE_integer("steps_per_epoch", 100, "Training batches per epoch.")
flags.DEFINE_integer("batch_size", 128, "Batch size.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("ckpt_every", 10, "Save checkpoint every N epochs; 0 means only final.")
flags.DEFINE_float("learning_rate", 1e-4, "Adam learning rate.")
flags.DEFINE_float("heatmap_sigma_px", 4.0, "Gaussian sigma in output-pixel units.")
flags.DEFINE_string(
    "encoder_variant",
    "resnetv1-10",
    "Encoder backbone variant: resnetv1-10-frozen, resnetv1-18-frozen, or resnetv1-10.",
)
flags.DEFINE_integer(
    "target_window", 0,
    "Frames on each side whose gaze is pooled into this frame's target. 0 keeps "
    "the single-point Gaussian. Above 0 the target becomes the distribution of "
    "where the operator looked around this moment, which is multi-modal when "
    "the eyes move between the hand and the ball -- a single point cannot "
    "represent that, and forcing the model to predict one is what caps the "
    "single-point model at 14 px.",
)
flags.DEFINE_integer(
    "target_max_frame_gap", 3,
    "Largest frame-id step allowed when walking out to a neighbour. Filtering "
    "leaves holes in the frame numbering (2.1% of steps, up to 20 frames), and "
    "pooling across a hole would mix in gaze from a different moment.",
)
flags.DEFINE_float(
    "target_decay", 0.0,
    "Exponential weight decay per frame of temporal distance. 0 weights every "
    "frame in the window equally.",
)
flags.DEFINE_integer(
    "aug_shift_px", 8,
    "Random translation applied to training images, in input pixels. The gaze "
    "target is shifted with the image. 0 disables. Without this the model fits "
    "7865 frames from a fixed camera and val stops improving after ~2 epochs.",
)
flags.DEFINE_float("aug_brightness", 0.25, "Random brightness scale range (+/-). 0 disables.")
flags.DEFINE_float("aug_contrast", 0.25, "Random contrast scale range (+/-). 0 disables.")
flags.DEFINE_boolean("debug_pred_logs", False, "Print extra prediction debug logs.")
flags.DEFINE_boolean("debug_grad_logs", False, "Print extra gradient debug logs.")
flags.DEFINE_integer("overfit_num_samples", 0, "If > 0, train on a fixed small subset for overfit debugging.")
flags.DEFINE_boolean(
    "overfit_use_train_subset_for_val",
    True,
    "When overfit_num_samples > 0, reuse the same subset for validation.",
)


def _load_samples(data_dir: str):
    samples = []
    for path in sorted(glob.glob(os.path.join(data_dir, "*.pkl"))):
        with open(path, "rb") as f:
            shard = pkl.load(f)
        samples.extend(shard)
    if not samples:
        raise FileNotFoundError(f"No pkl samples found in {data_dir}")
    return samples

def _valid_counts(samples):
    return {"valid": len(samples), "invalid": 0}


def _resolve_image_keys(train_samples):
    if FLAGS.image_keys.strip():
        image_keys = [key.strip() for key in FLAGS.image_keys.split(",") if key.strip()]
    else:
        try:
            from experiments.mappings import NEW_MAPPING

            if FLAGS.exp_name not in NEW_MAPPING:
                raise ValueError(f"Unknown exp_name={FLAGS.exp_name}. Available: {sorted(NEW_MAPPING)}")
            config = NEW_MAPPING[FLAGS.exp_name]()
            config.get_environment(
                fake_env=True,
                save_video=False,
                classifier=False,
                enable_tactile=FLAGS.enable_tactile,
            )
            image_keys = list(config.classifier_keys)
        except Exception as exc:
            image_keys = ["front_camera"]
            print(
                "[warn] could not load experiment classifier_keys "
                f"for exp_name={FLAGS.exp_name}; using {image_keys}. Error: {exc}"
            )

    available = set(train_samples[0]["observations"].keys())
    missing = [key for key in image_keys if key not in available]
    if missing:
        raise KeyError(
            f"Requested image_keys={image_keys}, but pkl observations only contain {sorted(available)}. "
            "Pass --image_keys=front_camera or regenerate the pkl with the required keys."
        )
    return image_keys


def _build_gaze_index(samples):
    """(recording, episode) -> frame_id -> gaze xy, for temporal pooling."""
    index = {}
    for sample in samples:
        key = (sample["recording_root"], int(sample["episode_index"]))
        index.setdefault(key, {})[int(sample["frame_id"])] = np.asarray(
            sample["gaze_xy"], dtype=np.float32)
    return index


def _neighbour_points(sample, index, window: int, max_gap: int, decay: float):
    """Gaze points pooled from this frame and its temporal neighbours.

    Walks outward one frame at a time and stops at a hole larger than
    `max_gap`, so a gap left by filtering never joins two unrelated moments.
    """
    centre = np.asarray(sample["gaze_xy"], dtype=np.float32)
    if window <= 0:
        return centre[None], np.ones(1, np.float32)

    episode = index.get((sample["recording_root"], int(sample["episode_index"])), {})
    frame_id = int(sample["frame_id"])
    points, weights = [centre], [1.0]
    for direction in (-1, 1):
        previous = frame_id
        found = 0
        candidate = frame_id + direction
        while found < window and abs(candidate - frame_id) <= window + max_gap:
            if candidate in episode:
                if abs(candidate - previous) > max_gap:
                    break
                found += 1
                points.append(episode[candidate])
                weights.append(float(np.exp(-decay * found)) if decay > 0 else 1.0)
                previous = candidate
            candidate += direction
    return np.stack(points), np.asarray(weights, np.float32)


def _select_overfit_subset(samples, subset_size: int, seed: int):
    if subset_size <= 0:
        return list(samples)
    subset_size = min(subset_size, len(samples))
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(samples))[:subset_size]
    return [samples[int(idx)] for idx in indices]


def _xy_to_heatmaps(xy: np.ndarray, height: int, width: int, sigma_px: float):
    """Gaussian target, normalised to sum to 1 so it is a spatial distribution.

    The loss below is a KL against a softmax over the map. Peak-1 Gaussians and
    an MSE through a sigmoid -- what this used to do -- let the background
    dominate: with sigma 4 on a 128x128 grid the Gaussian covers ~0.6% of the
    pixels, so a near-constant output already scores 0.0030 and the trained
    model only reached 0.0027. Its predicted peak-to-mean ratio was 11.6 where
    the target's is 162, i.e. an almost flat map whose argmax wanders.
    """
    xy = np.asarray(xy, dtype=np.float32)
    yy, xx = np.mgrid[0:height, 0:width]
    yy = yy[None, :, :]
    xx = xx[None, :, :]
    cx = xy[:, 0:1, None] * float(width - 1)
    cy = xy[:, 1:2, None] * float(height - 1)
    sigma = max(float(sigma_px), 1e-3)
    heatmaps = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma * sigma))
    heatmaps /= np.maximum(heatmaps.sum(axis=(1, 2), keepdims=True), 1e-8)
    return heatmaps.astype(np.float32)


def _points_to_heatmaps(points, weights, height, width, sigma_px):
    """Sum of Gaussians over a padded (batch, max_points, 2) point set."""
    yy, xx = np.mgrid[0:height, 0:width]
    yy = yy[None, None]
    xx = xx[None, None]
    cx = points[..., 0:1, None] * float(width - 1)
    cy = points[..., 1:2, None] * float(height - 1)
    cx = cx.reshape(points.shape[0], points.shape[1], 1, 1)
    cy = cy.reshape(points.shape[0], points.shape[1], 1, 1)
    sigma = max(float(sigma_px), 1e-3)
    blobs = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma * sigma))
    blobs *= weights[..., None, None]
    heatmaps = blobs.sum(axis=1)
    heatmaps /= np.maximum(heatmaps.sum(axis=(1, 2), keepdims=True), 1e-8)
    return heatmaps.astype(np.float32)


def _augment(observations, points, xy, image_keys, rng):
    """Shift and re-expose the training images; move the gaze target with them.

    No horizontal flip: the scene is asymmetric (the basket sits on one side),
    so a mirrored frame is not a scene this camera can ever produce.
    """
    first = observations[image_keys[0]]
    batch_n, height, width = first.shape[0], first.shape[1], first.shape[2]
    xy = xy.copy()
    points = points.copy()

    if FLAGS.aug_shift_px > 0:
        shift = FLAGS.aug_shift_px
        dx = rng.randint(-shift, shift + 1, size=batch_n)
        dy = rng.randint(-shift, shift + 1, size=batch_n)
        for image_key in image_keys:
            images = observations[image_key]
            out = np.empty_like(images)
            for i in range(batch_n):
                # Replicate-pad then crop, so shifting never invents black bars.
                padded = np.pad(images[i], ((shift, shift), (shift, shift), (0, 0)), mode="edge")
                y0, x0 = shift - dy[i], shift - dx[i]
                out[i] = padded[y0:y0 + height, x0:x0 + width]
            observations[image_key] = out
        # Every pooled point moves with the image, not just the centre one.
        sx = (dx / max(1.0, width - 1)).astype(np.float32)
        sy = (dy / max(1.0, height - 1)).astype(np.float32)
        xy[:, 0] = np.clip(xy[:, 0] + sx, 0.0, 1.0)
        xy[:, 1] = np.clip(xy[:, 1] + sy, 0.0, 1.0)
        points[..., 0] = np.clip(points[..., 0] + sx[:, None], 0.0, 1.0)
        points[..., 1] = np.clip(points[..., 1] + sy[:, None], 0.0, 1.0)

    if FLAGS.aug_brightness > 0 or FLAGS.aug_contrast > 0:
        for image_key in image_keys:
            images = observations[image_key]
            if FLAGS.aug_contrast > 0:
                scale = rng.uniform(1 - FLAGS.aug_contrast, 1 + FLAGS.aug_contrast,
                                    size=(batch_n, 1, 1, 1)).astype(np.float32)
                mean = images.mean(axis=(1, 2, 3), keepdims=True)
                images = (images - mean) * scale + mean
            if FLAGS.aug_brightness > 0:
                offset = rng.uniform(-FLAGS.aug_brightness, FLAGS.aug_brightness,
                                     size=(batch_n, 1, 1, 1)).astype(np.float32) * 255.0
                images = images + offset
            observations[image_key] = np.clip(images, 0.0, 255.0)
    return observations, points, xy


def _to_batch(samples, image_keys: list[str], rng=None, index=None):
    observations = {
        image_key: np.asarray(
            [sample["observations"][image_key] for sample in samples],
            dtype=np.float32,
        )
        for image_key in image_keys
    }
    xy = np.asarray([sample["gaze_xy"] for sample in samples], dtype=np.float32)

    # Pool the neighbourhood into a padded point set. Padded rows carry weight
    # 0, so they contribute nothing to the rasterised target.
    gathered = [
        _neighbour_points(sample, index or {}, FLAGS.target_window,
                          FLAGS.target_max_frame_gap, FLAGS.target_decay)
        for sample in samples
    ]
    max_points = max(len(pts) for pts, _ in gathered)
    points = np.zeros((len(samples), max_points, 2), np.float32)
    weights = np.zeros((len(samples), max_points), np.float32)
    for i, (pts, wts) in enumerate(gathered):
        points[i, :len(pts)] = pts
        weights[i, :len(wts)] = wts

    if rng is not None:
        observations, points, xy = _augment(observations, points, xy, image_keys, rng)

    heatmap = _points_to_heatmaps(points, weights, FLAGS.image_height,
                                  FLAGS.image_width, FLAGS.heatmap_sigma_px)
    return {
        "observations": observations,
        "xy": xy,
        "points": points,
        "weights": weights,
        "heatmap": heatmap,
    }


def _train_batches(samples, batch_size: int, seed: int, image_keys: list[str], index=None):
    rng = np.random.RandomState(seed)
    while True:
        idxs = rng.choice(len(samples), size=batch_size, replace=len(samples) < batch_size)
        # Augmentation is training-only; validation must stay a fixed target.
        yield _to_batch([samples[int(idx)] for idx in idxs], image_keys, rng, index)


def _val_batches(samples, batch_size: int, image_keys: list[str], index=None):
    for start in range(0, len(samples), batch_size):
        yield _to_batch(samples[start : start + batch_size], image_keys, None, index)


def _batch_observations(batch):
    return batch["observations"]


def _heatmap_to_xy_jnp(heatmap: jnp.ndarray):
    batch_size, height, width = heatmap.shape
    flat_idx = jnp.argmax(heatmap.reshape(batch_size, height * width), axis=-1)
    y = flat_idx // width
    x = flat_idx % width
    return jnp.stack(
        [
            x.astype(jnp.float32) / max(1.0, float(width - 1)),
            y.astype(jnp.float32) / max(1.0, float(height - 1)),
        ],
        axis=-1,
    )


def _point_nll(probs_np, xy, height, width):
    """-log p(true gaze pixel) under the predicted distribution.

    This is the metric that means something once the target is a distribution.
    `xy_err` compares the argmax against one point, so a model that correctly
    places mass on two plausible targets scores worse on it than a model that
    always guesses the same one -- even though it is the better model.
    """
    n = probs_np.shape[0]
    col = np.clip(np.round(xy[:, 0] * (width - 1)).astype(int), 0, width - 1)
    row = np.clip(np.round(xy[:, 1] * (height - 1)).astype(int), 0, height - 1)
    return -np.log(np.maximum(probs_np[np.arange(n), row, col], 1e-12))


def _mass_within(probs_np, xy, height, width, radius_px):
    """Predicted probability inside a disc around the true gaze point."""
    n = probs_np.shape[0]
    yy, xx = np.mgrid[0:height, 0:width]
    cx = xy[:, 0, None, None] * (width - 1)
    cy = xy[:, 1, None, None] * (height - 1)
    disc = ((xx[None] - cx) ** 2 + (yy[None] - cy) ** 2) <= radius_px ** 2
    return (probs_np * disc).sum(axis=(1, 2))


def eval_gaze_heatmap_epoch(state, samples, batch_size: int, image_keys: list[str],
                            index=None):
    xy_err_sum = 0.0
    xy_err_px_sum = 0.0
    heatmap_loss_sum = 0.0
    nll_sum = 0.0
    mass_sum = 0.0
    count = 0
    for batch in _val_batches(samples, batch_size, image_keys, index):
        outputs = state.apply_fn(
            {"params": state.params},
            _batch_observations(batch),
            train=False,
        )
        logits = outputs["heatmap_logits"]
        batch_n = logits.shape[0]
        log_probs = jax.nn.log_softmax(logits.reshape(batch_n, -1), axis=-1)
        probs = jnp.exp(log_probs).reshape(logits.shape)
        xy_pred = np.asarray(_heatmap_to_xy_jnp(probs))
        y_xy = batch["xy"]
        target = batch["heatmap"].reshape(batch_n, -1)
        heatmap_loss = -(target * np.asarray(log_probs)).sum(axis=-1)
        abs_err = np.linalg.norm(xy_pred - y_xy, axis=-1)
        abs_err_px = abs_err * float(max(FLAGS.image_width, FLAGS.image_height))
        probs_np = np.asarray(probs)
        nll = _point_nll(probs_np, y_xy, FLAGS.image_height, FLAGS.image_width)
        mass = _mass_within(probs_np, y_xy, FLAGS.image_height, FLAGS.image_width, 12.0)
        heatmap_loss_sum += float(heatmap_loss.sum())
        xy_err_sum += float(abs_err.sum())
        xy_err_px_sum += float(abs_err_px.sum())
        nll_sum += float(nll.sum())
        mass_sum += float(mass.sum())
        count += int(y_xy.shape[0])

    return (
        heatmap_loss_sum / max(1, count),
        xy_err_sum / max(1, count),
        xy_err_px_sum / max(1, count),
        nll_sum / max(1, count),
        mass_sum / max(1, count),
    )


def main(_):
    train_samples = _load_samples(FLAGS.train_data_dir)
    val_samples = _load_samples(FLAGS.val_data_dir)
    if not train_samples:
        raise ValueError("No training samples were loaded.")
    if not val_samples:
        raise ValueError("No validation samples were loaded.")

    if FLAGS.overfit_num_samples > 0:
        train_samples = _select_overfit_subset(
            train_samples,
            FLAGS.overfit_num_samples,
            FLAGS.seed,
        )
        if FLAGS.overfit_use_train_subset_for_val:
            val_samples = list(train_samples)
        else:
            val_samples = _select_overfit_subset(
                val_samples,
                FLAGS.overfit_num_samples,
                FLAGS.seed + 1,
            )
        print(
            f"[overfit] enabled subset_size={len(train_samples)} "
            f"use_train_subset_for_val={FLAGS.overfit_use_train_subset_for_val}"
        )

    image_keys = _resolve_image_keys(train_samples)
    print("[data] train:", len(train_samples), dict(_valid_counts(train_samples)))
    print("[data] val:  ", len(val_samples), dict(_valid_counts(val_samples)))
    print("[config] exp_name:", FLAGS.exp_name)
    print("[config] image_keys:", image_keys)
    print("[config] encoder_variant:", FLAGS.encoder_variant)
    print("[config] heatmap_sigma_px:", FLAGS.heatmap_sigma_px)
    print(f"[config] augmentation: shift={FLAGS.aug_shift_px}px "
          f"brightness={FLAGS.aug_brightness} contrast={FLAGS.aug_contrast}")

    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, init_key = jax.random.split(rng)
    state = create_gaze_point_predictor(
        init_key,
        {
            image_key: np.zeros((1, FLAGS.image_height, FLAGS.image_width, 3), np.float32)
            for image_key in image_keys
        },
        image_keys=image_keys,
        learning_rate=FLAGS.learning_rate,
        encoder_variant=FLAGS.encoder_variant,
    )

    train_index = _build_gaze_index(train_samples)
    val_index = _build_gaze_index(val_samples)
    if FLAGS.target_window > 0:
        probe = _to_batch(train_samples[:64], image_keys, None, train_index)
        pooled = (probe["weights"] > 0).sum(axis=1)
        print(f"[config] target: pooled distribution, window=+/-{FLAGS.target_window} "
              f"frames, max_gap={FLAGS.target_max_frame_gap}, decay={FLAGS.target_decay}")
        print(f"[config] points per target: mean {pooled.mean():.1f} "
              f"min {pooled.min()} max {pooled.max()} (of {2*FLAGS.target_window+1} possible)")
    else:
        print("[config] target: single-point Gaussian")

    train_iter = _train_batches(
        train_samples,
        FLAGS.batch_size,
        FLAGS.seed,
        image_keys,
        train_index,
    )

    @jax.jit
    def train_step(state, batch, key):
        y_heatmap = batch["heatmap"]
        y_xy = batch["xy"]

        def loss_fn(params):
            outputs = state.apply_fn(
                {"params": params},
                _batch_observations(batch),
                rngs={"dropout": key},
                train=True,
            )
            logits = outputs["heatmap_logits"]
            batch_n = logits.shape[0]
            log_probs = jax.nn.log_softmax(logits.reshape(batch_n, -1), axis=-1)
            target = y_heatmap.reshape(batch_n, -1)
            # KL(target || prediction), constant entropy term dropped.
            heatmap_loss = -(target * log_probs).sum(axis=-1).mean()
            probs = jnp.exp(log_probs).reshape(logits.shape)
            return heatmap_loss, (heatmap_loss, probs)

        (total_loss, (heatmap_loss, probs)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        new_state = state.apply_gradients(grads=grads)
        xy_pred = _heatmap_to_xy_jnp(probs)
        xy_err = jnp.linalg.norm(xy_pred - y_xy, axis=-1).mean()
        xy_err_px = xy_err * float(max(FLAGS.image_width, FLAGS.image_height))
        return new_state, total_loss, heatmap_loss, xy_err, xy_err_px

    best_val = {"xy_err": float("inf"), "nll": float("inf"), "epoch": -1}
    ckpt_dir = Path(FLAGS.checkpoint_dir).expanduser().resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in tqdm(range(FLAGS.num_epochs)):
        xy_err_sum = 0.0
        xy_err_px_sum = 0.0
        total_sum = 0.0
        heatmap_loss_sum = 0.0

        for step in range(FLAGS.steps_per_epoch):
            batch = next(train_iter)
            rng, step_key = jax.random.split(rng)
            state, total_loss, heatmap_loss, xy_err, xy_err_px = train_step(
                state,
                batch,
                step_key,
            )
            total_sum += float(total_loss)
            heatmap_loss_sum += float(heatmap_loss)
            xy_err_sum += float(xy_err)
            xy_err_px_sum += float(xy_err_px)

            if step % 100 == 0:
                print(
                    f"[gaze-heatmap] epoch={epoch + 1} step={step + 1}/{FLAGS.steps_per_epoch} "
                    f"total={float(total_loss):.4f} heatmap_loss={float(heatmap_loss):.4f} "
                    f"xy_err={float(xy_err):.4f} xy_err_px={float(xy_err_px):.2f}"
                )

        denom = max(1, FLAGS.steps_per_epoch)
        train_xy_err = xy_err_sum / denom
        train_xy_err_px = xy_err_px_sum / denom
        print(
            f"[gaze-heatmap][train] epoch={epoch + 1} "
            f"total={total_sum / denom:.4f} heatmap_loss={heatmap_loss_sum / denom:.4f} "
            f"xy_err={train_xy_err:.4f} xy_err_px={train_xy_err_px:.2f}"
        )

        val_heatmap_loss, val_xy_err, val_xy_err_px, val_nll, val_mass = (
            eval_gaze_heatmap_epoch(
                state, val_samples, FLAGS.batch_size, image_keys, val_index,
            )
        )
        print(
            f"[gaze-heatmap][val]   epoch={epoch + 1} "
            f"heatmap_loss={val_heatmap_loss:.4f} xy_err={val_xy_err:.4f} "
            f"xy_err_px={val_xy_err_px:.2f} nll={val_nll:.4f} mass@12px={val_mass:.4f}"
        )

        # Save only on improvement, so `checkpoints.latest_checkpoint(ckpt_dir)`
        # -- what the RL loader calls -- resolves to the best epoch rather than
        # the last one. Periodic snapshots go to a subdirectory where they
        # cannot shadow it.
        # Select on whatever the model is actually being trained to do: the
        # point error is the wrong yardstick for a distributional target.
        score = val_nll if FLAGS.target_window > 0 else val_xy_err
        best_score = best_val["nll"] if FLAGS.target_window > 0 else best_val["xy_err"]
        if score < best_score:
            best_val = {"xy_err": float(val_xy_err), "nll": float(val_nll),
                        "epoch": epoch + 1}
            checkpoints.save_checkpoint(
                str(ckpt_dir), state, step=epoch + 1, overwrite=False, keep=3,
            )
            metric = "nll" if FLAGS.target_window > 0 else "xy_err"
            print(f"[gaze-heatmap][ckpt] new best val_{metric}={score:.4f} "
                  f"(xy_err={val_xy_err:.4f} nll={val_nll:.4f}) "
                  f"-> saved epoch {epoch + 1}")

        if FLAGS.ckpt_every and (epoch + 1) % FLAGS.ckpt_every == 0:
            periodic = ckpt_dir / "periodic"
            periodic.mkdir(parents=True, exist_ok=True)
            checkpoints.save_checkpoint(
                str(periodic), state, step=epoch + 1, overwrite=False, keep=100,
            )

    print("[gaze-heatmap][summary]")
    print(f"best val xy_err={best_val['xy_err']:.4f} nll={best_val['nll']:.4f} "
          f"epoch={best_val['epoch']}")
    resolved = checkpoints.latest_checkpoint(str(ckpt_dir))
    print(f"[gaze-heatmap][ckpt] loader will resolve to: {resolved}")


if __name__ == "__main__":
    app.run(main)
