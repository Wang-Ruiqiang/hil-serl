#!/usr/bin/env python3

import os
import pickle as pkl
import sys
from pathlib import Path

import jax
import numpy as np
from absl import app, flags


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from serl_launcher.networks.reward_classifier import load_classifier_func


FLAGS = flags.FLAGS

flags.DEFINE_string(
    "data_dir",
    "",
    "Classifier pkl directory. Defaults to examples/reward_classifier/classifier_data_pick.",
)
flags.DEFINE_string(
    "checkpoint_dir",
    "",
    "Classifier checkpoint directory. Defaults to examples/reward_classifier/classifier_ckpt_ball_pick.",
)
flags.DEFINE_integer("batch_size", 256, "Evaluation batch size.")
flags.DEFINE_float(
    "threshold",
    0.95,
    "Success probability threshold. 0.95 matches train_reward_classifier.py accuracy logging.",
)
flags.DEFINE_multi_string(
    "image_key",
    None,
    "Image keys used by the classifier. Defaults to keys found in the pkl, excluding state.",
)
flags.DEFINE_integer(
    "max_per_class",
    0,
    "Optional max samples per class for a quick smoke test. 0 means use all samples.",
)


def _default_data_dir():
    return SCRIPT_DIR / "classifier_data_pick"


def _default_checkpoint_dir():
    return SCRIPT_DIR / "classifier_ckpt_ball_pick"


def _load_pickle_stream(path):
    data = []
    with path.open("rb") as f:
        while True:
            try:
                data.extend(pkl.load(f))
            except EOFError:
                break
    return data


def _load_labeled_data(data_dir):
    data_dir = Path(data_dir).expanduser()
    success_paths = sorted(data_dir.glob("*success*.pkl"))
    failure_paths = sorted(data_dir.glob("*failure*.pkl"))
    if not success_paths:
        raise FileNotFoundError(f"No success pkl files found in {data_dir}")
    if not failure_paths:
        raise FileNotFoundError(f"No failure pkl files found in {data_dir}")

    samples = []
    labels = []
    for path in success_paths:
        loaded = _load_pickle_stream(path)
        samples.extend(loaded)
        labels.extend([1] * len(loaded))
    for path in failure_paths:
        loaded = _load_pickle_stream(path)
        samples.extend(loaded)
        labels.extend([0] * len(loaded))
    return samples, np.asarray(labels, dtype=np.int32), success_paths, failure_paths


def _infer_image_keys(observation):
    if FLAGS.image_key:
        return list(FLAGS.image_key)
    preferred_order = ("front_camera", "wrist_camera", "tactile_data", "front_classifier")
    keys = [key for key in preferred_order if key in observation]
    keys.extend(
        key
        for key in observation
        if key not in keys and key != "state" and getattr(observation[key], "ndim", 0) >= 3
    )
    if not keys:
        raise ValueError(f"Could not infer image keys from observation keys: {list(observation.keys())}")
    return keys


def _batch_observations(samples, image_keys):
    observations = {}
    for key in image_keys:
        images = np.stack([sample["observations"][key] for sample in samples], axis=0)
        if images.ndim == 4:
            # The classifier encoder is built with enable_stacking=True and expects
            # (batch, history, height, width, channels). Recorded frames store HWC,
            # so add a singleton history dimension for offline evaluation.
            images = images[:, None, ...]
        observations[key] = images
    return observations


def _predict_probs(classifier_func, samples, image_keys):
    probs = []
    for start in range(0, len(samples), FLAGS.batch_size):
        batch_samples = samples[start : start + FLAGS.batch_size]
        batch_obs = _batch_observations(batch_samples, image_keys)
        logits = classifier_func(batch_obs)
        batch_probs = np.asarray(jax.nn.sigmoid(logits)).reshape(-1)
        probs.append(batch_probs)
    return np.concatenate(probs, axis=0)


def _summarize_split(name, probs):
    if len(probs) == 0:
        print(f"[{name}] count=0")
        return
    print(
        f"[{name}] count={len(probs)} "
        f"mean={probs.mean():.4f} std={probs.std():.4f} "
        f"min={probs.min():.4f} p05={np.percentile(probs, 5):.4f} "
        f"p50={np.percentile(probs, 50):.4f} p95={np.percentile(probs, 95):.4f} "
        f"max={probs.max():.4f}"
    )


def _print_hard_examples(samples, labels, probs, image_keys):
    success_idx = np.where(labels == 1)[0]
    failure_idx = np.where(labels == 0)[0]
    if len(success_idx):
        low_success = success_idx[np.argsort(probs[success_idx])[:5]]
        print("[hard] lowest success probabilities:")
        for idx in low_success:
            print(f"  idx={idx} prob={probs[idx]:.4f} keys={image_keys}")
    if len(failure_idx):
        high_failure = failure_idx[np.argsort(-probs[failure_idx])[:5]]
        print("[hard] highest failure probabilities:")
        for idx in high_failure:
            print(f"  idx={idx} prob={probs[idx]:.4f} keys={image_keys}")


def main(_):
    data_dir = Path(FLAGS.data_dir).expanduser() if FLAGS.data_dir else _default_data_dir()
    checkpoint_dir = (
        Path(FLAGS.checkpoint_dir).expanduser()
        if FLAGS.checkpoint_dir
        else _default_checkpoint_dir()
    )
    data_dir = data_dir.resolve()
    checkpoint_dir = checkpoint_dir.resolve()

    samples, labels, success_paths, failure_paths = _load_labeled_data(data_dir)
    if FLAGS.max_per_class > 0:
        success_idx = np.where(labels == 1)[0][: FLAGS.max_per_class]
        failure_idx = np.where(labels == 0)[0][: FLAGS.max_per_class]
        keep_idx = np.concatenate([success_idx, failure_idx], axis=0)
        samples = [samples[int(idx)] for idx in keep_idx]
        labels = labels[keep_idx]
    image_keys = _infer_image_keys(samples[0]["observations"])
    sample_observations = _batch_observations(samples[:1], image_keys)

    print(f"[source] data_dir={data_dir}")
    print(f"[source] checkpoint_dir={checkpoint_dir}")
    print(f"[source] success_files={len(success_paths)} failure_files={len(failure_paths)}")
    print(f"[source] samples={len(samples)} success={int(labels.sum())} failure={int((1 - labels).sum())}")
    print(f"[source] image_keys={image_keys}")

    classifier_func = load_classifier_func(
        key=jax.random.PRNGKey(0),
        sample=sample_observations,
        image_keys=image_keys,
        checkpoint_path=str(checkpoint_dir),
    )
    probs = _predict_probs(classifier_func, samples, image_keys)
    preds = (probs >= FLAGS.threshold).astype(np.int32)

    accuracy = np.mean(preds == labels)
    tp = int(np.sum((preds == 1) & (labels == 1)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))
    success_recall = tp / max(tp + fn, 1)
    failure_reject = tn / max(tn + fp, 1)

    print(f"[metric] threshold={FLAGS.threshold:.3f}")
    print(f"[metric] accuracy={accuracy:.4f}")
    print(f"[metric] success_recall={success_recall:.4f} failure_reject={failure_reject:.4f}")
    print(f"[metric] tp={tp} tn={tn} fp={fp} fn={fn}")
    _summarize_split("success_prob", probs[labels == 1])
    _summarize_split("failure_prob", probs[labels == 0])
    _print_hard_examples(samples, labels, probs, image_keys)


if __name__ == "__main__":
    app.run(main)
