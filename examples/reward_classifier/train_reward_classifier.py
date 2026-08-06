#!/usr/bin/env python3

import glob
import os
import pickle as pkl
import sys
from collections import OrderedDict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = SCRIPT_DIR.parent
REPO_ROOT = EXAMPLES_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(EXAMPLES_DIR))

import flax.linen as nn
import gymnasium as gym
import jax
import numpy as np
import optax
from absl import app, flags
from flax.training import checkpoints
from jax import numpy as jnp
from tqdm import tqdm

from serl_launcher.data.data_store import ReplayBuffer
from serl_launcher.networks.reward_classifier import create_classifier
from serl_launcher.utils.train_utils import concat_batches
from serl_launcher.vision.data_augmentations import batched_random_crop


FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", "flip_object", "Name of experiment.")
flags.DEFINE_integer("num_epochs", 50, "Number of training epochs.")
flags.DEFINE_integer("batch_size", 256, "Batch size.")
flags.DEFINE_enum("classifier_task", "flip_object", ["", "pick", "place", "lid_grip", "bottle_twist", "tube_pick", "tube_insertion", "flip_object"], "Classifier stage.")
flags.DEFINE_integer("is_pick_task", 1, "Deprecated tennis-ball pick flag.")
flags.DEFINE_integer("is_place_task", 0, "Deprecated tennis-ball place flag.")
flags.DEFINE_integer("is_tube_pick", 0, "Deprecated tube-pick flag.")
flags.DEFINE_integer("enable_tactile", 0, "Whether tactile_data is included.")
flags.DEFINE_multi_string("image_key", None, "Override classifier image keys.")
flags.DEFINE_string("data_dir", "", "Directory containing *success*.pkl and *failure*.pkl.")
flags.DEFINE_string("checkpoint_dir", "", "Output checkpoint directory.")


def _classifier_task():
    if FLAGS.classifier_task:
        return FLAGS.classifier_task
    if FLAGS.exp_name == "twist_bottle_cap":
        return "bottle_twist"
    if FLAGS.exp_name == "lid_grip":
        return "lid_grip"
    if FLAGS.exp_name == "tube_insertion":
        return "tube_pick" if FLAGS.is_tube_pick else "tube_insertion"
    if FLAGS.exp_name == "tennis_ball_pick":
        return "place" if FLAGS.is_place_task else "pick"
    return FLAGS.exp_name


def _data_dir_name():
    task = _classifier_task()
    suffix = "_no_tactile" if not FLAGS.enable_tactile else ""
    return f"classifier_data_{task}{suffix}"


def _checkpoint_dir_name():
    task = _classifier_task()
    tactile = bool(FLAGS.enable_tactile)
    if FLAGS.exp_name == "tennis_ball_pick":
        name = "classifier_ckpt_ball_place" if task == "place" else "classifier_ckpt_ball_pick"
        return os.path.join("tennis_ball_pick_classifier", f"{name}{'' if tactile else '_no_tactile'}")
    if FLAGS.exp_name == "tube_insertion":
        name = "classifier_ckpt_tube_pick" if task == "tube_pick" else "classifier_ckpt_tube_insertion"
        return os.path.join("tube_insertion_classifier", f"{name}{'' if tactile else '_no_tactile'}")
    if FLAGS.exp_name == "twist_bottle_cap":
        name = "classifier_ckpt_bottle_twist"
        return f"{name}{'' if tactile else '_no_tactile'}"
    if FLAGS.exp_name == "lid_grip":
        name = "classifier_ckpt_lid_grip"
        return f"{name}{'' if tactile else '_no_tactile'}"
    if FLAGS.exp_name == "flip_object":
        return os.path.join("flip_object_classifier", f"classifier_ckpt_flip_object{'' if tactile else '_no_tactile'}")
    return f"classifier_ckpt_{task}{'' if tactile else '_no_tactile'}"


def _resolve_examples_path(path, default_name):
    if path:
        return os.path.abspath(os.path.expanduser(path))
    return str(EXAMPLES_DIR / default_name)


def _resolve_default_data_dir():
    if FLAGS.data_dir:
        return os.path.abspath(os.path.expanduser(FLAGS.data_dir))
    default_name = _data_dir_name()
    reward_classifier_dir = SCRIPT_DIR / default_name
    legacy_dir = EXAMPLES_DIR / default_name
    return reward_classifier_dir if os.path.exists(reward_classifier_dir) else legacy_dir


def _load_pickle_stream(path):
    data = []
    with open(path, "rb") as f:
        while True:
            try:
                data.extend(pkl.load(f))
            except EOFError:
                break
    return data


def _infer_image_keys(observation):
    if FLAGS.image_key:
        return list(FLAGS.image_key)
    preferred_order = ("front_camera", "wrist_camera", "tactile_data", "front_classifier")
    image_keys = [key for key in preferred_order if key in observation]
    image_keys.extend(
        key
        for key, value in observation.items()
        if key not in image_keys
        and key != "state"
        and getattr(np.asarray(value), "ndim", 0) >= 3
    )
    if not image_keys:
        raise ValueError(f"Could not infer image keys from observation keys: {list(observation.keys())}")
    return image_keys


def _space_from_array(value, *, add_history_dim=False):
    value = np.asarray(value)
    shape = value.shape
    if add_history_dim:
        shape = (1, *shape)
    if value.dtype == np.uint8:
        return gym.spaces.Box(
            low=np.zeros(shape, dtype=np.uint8),
            high=np.full(shape, 255, dtype=np.uint8),
            dtype=np.uint8,
        )
    dtype = np.float32 if np.issubdtype(value.dtype, np.floating) else value.dtype
    return gym.spaces.Box(
        low=np.full(shape, -np.inf, dtype=dtype),
        high=np.full(shape, np.inf, dtype=dtype),
        dtype=dtype,
    )


def _make_spaces_from_transition(transition):
    obs_spaces = OrderedDict()
    for key, value in transition["observations"].items():
        value = np.asarray(value)
        add_history_dim = value.ndim in (1, 3)
        obs_spaces[key] = _space_from_array(value, add_history_dim=add_history_dim)
    action = np.asarray(transition["actions"], dtype=np.float32)
    action_space = gym.spaces.Box(
        low=np.full(action.shape, -np.inf, dtype=np.float32),
        high=np.full(action.shape, np.inf, dtype=np.float32),
        dtype=np.float32,
    )
    return gym.spaces.Dict(obs_spaces), action_space


def _load_classifier_data(data_dir):
    success_paths = glob.glob(os.path.join(data_dir, "*success*.pkl"))
    failure_paths = glob.glob(os.path.join(data_dir, "*failure*.pkl"))
    if not success_paths:
        raise FileNotFoundError(f"No success pkl files found in {data_dir}")
    if not failure_paths:
        raise FileNotFoundError(f"No failure pkl files found in {data_dir}")
    success_data = []
    for path in success_paths:
        success_data.extend(_load_pickle_stream(path))
    failure_data = []
    for path in failure_paths:
        failure_data.extend(_load_pickle_stream(path))
    if not success_data:
        raise ValueError(f"No success transitions loaded from {data_dir}")
    if not failure_data:
        raise ValueError(f"No failure transitions loaded from {data_dir}")
    return success_data, failure_data, success_paths, failure_paths


def main(_):
    devices = jax.local_devices()
    sharding = jax.sharding.PositionalSharding(devices)

    data_dir = _resolve_default_data_dir()
    success_data, failure_data, success_paths, failure_paths = _load_classifier_data(data_dir)
    image_keys = _infer_image_keys(success_data[0]["observations"])
    observation_space, action_space = _make_spaces_from_transition(success_data[0])

    print(f"[source] data_dir={data_dir}")
    print(f"[source] classifier_task={_classifier_task()}")
    print(f"[source] success_files={len(success_paths)} failure_files={len(failure_paths)}")
    print(f"[source] success={len(success_data)} failure={len(failure_data)}")
    print(f"[source] image_keys={image_keys}")
    print(f"[source] state_shape={observation_space['state'].shape}")

    pos_buffer = ReplayBuffer(
        observation_space=observation_space,
        action_space=action_space,
        capacity=max(len(success_data), 1),
        include_label=True,
    )
    for transition in success_data:
        transition["labels"] = 1
        pos_buffer.insert(transition)

    neg_buffer = ReplayBuffer(
        observation_space=observation_space,
        action_space=action_space,
        capacity=max(len(failure_data), 1),
        include_label=True,
    )
    for transition in failure_data:
        transition["labels"] = 0
        neg_buffer.insert(transition)

    pos_iterator = pos_buffer.get_iterator(
        sample_args={"batch_size": FLAGS.batch_size // 2},
        device=sharding.replicate(),
    )
    neg_iterator = neg_buffer.get_iterator(
        sample_args={"batch_size": FLAGS.batch_size // 2},
        device=sharding.replicate(),
    )

    rng = jax.random.PRNGKey(0)
    rng, key = jax.random.split(rng)
    sample = concat_batches(next(pos_iterator), next(neg_iterator), axis=0)
    classifier = create_classifier(key, sample["observations"], image_keys)

    def data_augmentation_fn(rng, observations):
        for pixel_key in image_keys:
            num_batch_dims = 2 if observations[pixel_key].ndim == 5 else 1
            observations = observations.copy(
                add_or_replace={
                    pixel_key: batched_random_crop(
                        observations[pixel_key],
                        rng,
                        padding=4,
                        num_batch_dims=num_batch_dims,
                    )
                }
            )
        return observations

    @jax.jit
    def train_step(state, batch, key):
        def loss_fn(params):
            logits = state.apply_fn(
                {"params": params},
                batch["observations"],
                rngs={"dropout": key},
                train=True,
            )
            return optax.sigmoid_binary_cross_entropy(logits, batch["labels"]).mean()

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        logits = state.apply_fn(
            {"params": state.params},
            batch["observations"],
            train=False,
            rngs={"dropout": key},
        )
        train_accuracy = jnp.mean((nn.sigmoid(logits) >= 0.95) == batch["labels"])
        return state.apply_gradients(grads=grads), loss, train_accuracy

    for epoch in tqdm(range(FLAGS.num_epochs)):
        batch = concat_batches(next(pos_iterator), next(neg_iterator), axis=0)
        rng, key = jax.random.split(rng)
        batch = batch.copy(
            add_or_replace={
                "observations": data_augmentation_fn(key, batch["observations"]),
                "labels": batch["labels"][..., None],
            }
        )
        rng, key = jax.random.split(rng)
        classifier, train_loss, train_accuracy = train_step(classifier, batch, key)
        print(f"Epoch: {epoch + 1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

    checkpoint_dir = _resolve_examples_path(FLAGS.checkpoint_dir, _checkpoint_dir_name())
    checkpoints.save_checkpoint(
        checkpoint_dir,
        classifier,
        step=FLAGS.num_epochs,
        overwrite=True,
    )
    print(f"[done] checkpoint_dir={checkpoint_dir}")


if __name__ == "__main__":
    app.run(main)
