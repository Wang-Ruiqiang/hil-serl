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
flags.DEFINE_integer("num_epochs", 20, "Number of training epochs.")
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


def _select_overfit_subset(samples, subset_size: int, seed: int):
    if subset_size <= 0:
        return list(samples)
    subset_size = min(subset_size, len(samples))
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(samples))[:subset_size]
    return [samples[int(idx)] for idx in indices]


def _xy_to_heatmaps(xy: np.ndarray, height: int, width: int, sigma_px: float):
    xy = np.asarray(xy, dtype=np.float32)
    yy, xx = np.mgrid[0:height, 0:width]
    yy = yy[None, :, :]
    xx = xx[None, :, :]
    cx = xy[:, 0:1, None] * float(width - 1)
    cy = xy[:, 1:2, None] * float(height - 1)
    sigma = max(float(sigma_px), 1e-3)
    heatmaps = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma * sigma))
    return heatmaps.astype(np.float32)


def _to_batch(samples, image_keys: list[str]):
    observations = {
        image_key: np.asarray(
            [sample["observations"][image_key] for sample in samples],
            dtype=np.float32,
        )
        for image_key in image_keys
    }
    xy = np.asarray([sample["gaze_xy"] for sample in samples], dtype=np.float32)
    heatmap = _xy_to_heatmaps(xy, FLAGS.image_height, FLAGS.image_width, FLAGS.heatmap_sigma_px)
    return {
        "observations": observations,
        "xy": xy,
        "heatmap": heatmap,
    }


def _train_batches(samples, batch_size: int, seed: int, image_keys: list[str]):
    rng = np.random.RandomState(seed)
    while True:
        idxs = rng.choice(len(samples), size=batch_size, replace=len(samples) < batch_size)
        yield _to_batch([samples[int(idx)] for idx in idxs], image_keys)


def _val_batches(samples, batch_size: int, image_keys: list[str]):
    for start in range(0, len(samples), batch_size):
        yield _to_batch(samples[start : start + batch_size], image_keys)


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


def eval_gaze_heatmap_epoch(state, samples, batch_size: int, image_keys: list[str]):
    xy_err_sum = 0.0
    heatmap_loss_sum = 0.0
    count = 0
    for batch in _val_batches(samples, batch_size, image_keys):
        outputs = state.apply_fn(
            {"params": state.params},
            _batch_observations(batch),
            train=False,
        )
        logits = outputs["heatmap_logits"]
        probs = jax.nn.sigmoid(logits)
        probs_np = np.asarray(probs)
        xy_pred = np.asarray(_heatmap_to_xy_jnp(probs))
        y_xy = batch["xy"]
        heatmap_loss = ((probs_np - batch["heatmap"]) ** 2).mean(axis=(1, 2))
        abs_err = np.abs(xy_pred - y_xy).sum(axis=-1)
        heatmap_loss_sum += float(heatmap_loss.sum())
        xy_err_sum += float(abs_err.sum())
        count += int(y_xy.shape[0])

    return heatmap_loss_sum / max(1, count), xy_err_sum / max(1, count)


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

    train_iter = _train_batches(
        train_samples,
        FLAGS.batch_size,
        FLAGS.seed,
        image_keys,
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
            probs = jax.nn.sigmoid(logits)
            heatmap_loss = ((probs - y_heatmap) ** 2).mean()
            return heatmap_loss, (heatmap_loss, probs)

        (total_loss, (heatmap_loss, probs)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        new_state = state.apply_gradients(grads=grads)
        xy_pred = _heatmap_to_xy_jnp(probs)
        xy_err = jnp.abs(xy_pred - y_xy).sum(axis=-1).mean()
        return new_state, total_loss, heatmap_loss, xy_err

    best_val = {"xy_err": float("inf"), "epoch": -1}
    ckpt_dir = Path(FLAGS.checkpoint_dir).expanduser().resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in tqdm(range(FLAGS.num_epochs)):
        xy_err_sum = 0.0
        total_sum = 0.0
        heatmap_loss_sum = 0.0

        for step in range(FLAGS.steps_per_epoch):
            batch = next(train_iter)
            rng, step_key = jax.random.split(rng)
            state, total_loss, heatmap_loss, xy_err = train_step(
                state,
                batch,
                step_key,
            )
            total_sum += float(total_loss)
            heatmap_loss_sum += float(heatmap_loss)
            xy_err_sum += float(xy_err)

            if step % 100 == 0:
                print(
                    f"[gaze-heatmap] epoch={epoch + 1} step={step + 1}/{FLAGS.steps_per_epoch} "
                    f"total={float(total_loss):.4f} heatmap_loss={float(heatmap_loss):.4f} "
                    f"xy_err={float(xy_err):.4f}"
                )

        denom = max(1, FLAGS.steps_per_epoch)
        train_xy_err = xy_err_sum / denom
        print(
            f"[gaze-heatmap][train] epoch={epoch + 1} "
            f"total={total_sum / denom:.4f} heatmap_loss={heatmap_loss_sum / denom:.4f} "
            f"xy_err={train_xy_err:.4f}"
        )

        val_heatmap_loss, val_xy_err = eval_gaze_heatmap_epoch(
            state,
            val_samples,
            FLAGS.batch_size,
            image_keys,
        )
        print(
            f"[gaze-heatmap][val]   epoch={epoch + 1} "
            f"heatmap_loss={val_heatmap_loss:.4f} xy_err={val_xy_err:.4f}"
        )

        if val_xy_err < best_val["xy_err"]:
            best_val = {"xy_err": float(val_xy_err), "epoch": epoch + 1}

        if (FLAGS.ckpt_every and (epoch + 1) % FLAGS.ckpt_every == 0) or epoch + 1 == FLAGS.num_epochs:
            checkpoints.save_checkpoint(
                str(ckpt_dir),
                state,
                step=epoch + 1,
                overwrite=False,
                keep=100,
            )
            print(f"[gaze-heatmap][ckpt] saved epoch {epoch + 1} -> {ckpt_dir}")

    print("[gaze-heatmap][summary]")
    print(f"best val xy_err={best_val['xy_err']:.4f} epoch={best_val['epoch']}")


if __name__ == "__main__":
    app.run(main)
