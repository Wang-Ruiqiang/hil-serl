import glob
import os, sys
import pickle as pkl
import jax
from jax import numpy as jnp
import flax.linen as nn
from flax.training import checkpoints
import numpy as np
import optax
from tqdm import tqdm
from absl import app, flags

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
EXAMPLES_DIR = os.path.join(REPO_ROOT, "examples")
ROBOT_INFRA_DIR = os.path.join(REPO_ROOT, "serl_robot_infra")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, EXAMPLES_DIR)
sys.path.insert(0, ROBOT_INFRA_DIR)

from serl_launcher.data.data_store import ReplayBuffer
from serl_launcher.utils.train_utils import concat_batches
from serl_launcher.vision.data_augmentations import batched_random_crop
from serl_launcher.networks.reward_classifier import create_classifier


from experiments.mappings import NEW_MAPPING


FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", "tennis_ball_pick", "Name of experiment corresponding to folder.")
flags.DEFINE_integer("num_epochs", 50, "Number of training epochs.")
flags.DEFINE_integer("batch_size", 256, "Batch size.")
flags.DEFINE_integer("is_pick_task", 1, "evaluate pick or place task.")
flags.DEFINE_integer("is_place_task", 0, "evaluate pick or place task.")
flags.DEFINE_integer("is_tube_pick", 0, "evaluate pick or place task.")
flags.DEFINE_integer("enable_tactile", 1, "Whether to include tactile_data in observations.")
flags.DEFINE_string(
    "data_dir",
    "",
    "Classifier data directory. Defaults to the task-specific directory next to this script.",
)
flags.DEFINE_string(
    "checkpoint_dir",
    "",
    "Classifier checkpoint output directory. Defaults to the task-specific directory next to this script.",
)


def _task_data_dir_name():
    if FLAGS.exp_name == "twist_bottle_cap":
        return "classifier_data_bottle_twist" if FLAGS.enable_tactile else "classifier_data_bottle_twist_no_tactile"
    if FLAGS.exp_name == "lid_grip":
        return "classifier_data_lid_grip" if FLAGS.enable_tactile else "classifier_data_lid_grip_no_tactile"
    if FLAGS.exp_name == "tube_insertion":
        if FLAGS.is_tube_pick:
            return "classifier_data_tube_pick" if FLAGS.enable_tactile else "classifier_data_tube_pick_no_tactile"
        return "classifier_data_tube_insertion" if FLAGS.enable_tactile else "classifier_data_tube_insertion_no_tactile"
    if FLAGS.exp_name == "tennis_ball_pick":
        if FLAGS.is_place_task:
            return "classifier_data_place" if FLAGS.enable_tactile else "classifier_data_place_no_tactile"
        return "classifier_data_pick" if FLAGS.enable_tactile else "classifier_data_pick_no_tactile"
    raise ValueError(f"Unsupported exp_name: {FLAGS.exp_name}")


def _task_checkpoint_dir_name():
    if FLAGS.exp_name == "twist_bottle_cap":
        return "classifier_ckpt_bottle_twist" if FLAGS.enable_tactile else "classifier_ckpt_bottle_twist_no_tactile"
    if FLAGS.exp_name == "lid_grip":
        return "classifier_ckpt_lid_grip" if FLAGS.enable_tactile else "classifier_ckpt_lid_grip_no_tactile"
    if FLAGS.exp_name == "tube_insertion":
        if FLAGS.is_tube_pick:
            return "classifier_ckpt_tube_pick" if FLAGS.enable_tactile else "classifier_ckpt_tube_pick_no_tactile"
        return "classifier_ckpt_tube_insertion" if FLAGS.enable_tactile else "classifier_ckpt_tube_insertion_no_tactile"
    if FLAGS.exp_name == "tennis_ball_pick":
        if FLAGS.is_place_task:
            return "classifier_ckpt_ball_place" if FLAGS.enable_tactile else "classifier_ckpt_ball_place_no_tactile"
        return "classifier_ckpt_ball_pick" if FLAGS.enable_tactile else "classifier_ckpt_ball_pick_no_tactile"
    raise ValueError(f"Unsupported exp_name: {FLAGS.exp_name}")


def _resolve_script_relative_path(path, default_name):
    if path:
        return os.path.abspath(os.path.expanduser(path))
    return os.path.join(SCRIPT_DIR, default_name)


def main(_):
    assert FLAGS.exp_name in NEW_MAPPING, 'Experiment folder not found.'
    config = NEW_MAPPING[FLAGS.exp_name]()
    env = config.get_environment(fake_env=True, save_video=False, classifier=False, enable_tactile=FLAGS.enable_tactile)

    devices = jax.local_devices()
    sharding = jax.sharding.PositionalSharding(devices)
    
    # stack_observation_space = space_stack(observation_space, 1)
    
    # Create buffer for positive transitions
    # print("env.action_space shape= ", env.action_space.shape)
    # print("env.observation_space shape= ", env.observation_space["state"].shape)
    pos_buffer = ReplayBuffer(
        observation_space=env.observation_space,
        action_space=env.action_space,
        capacity=10000,
        include_label=True,
    )
    
    data_dir = _resolve_script_relative_path(FLAGS.data_dir, _task_data_dir_name())
    success_paths = glob.glob(os.path.join(data_dir, "*success*.pkl"))
    failure_paths = glob.glob(os.path.join(data_dir, "*failure*.pkl"))
    print(f"[source] data_dir={data_dir}")
    print(f"[source] success_files={len(success_paths)} failure_files={len(failure_paths)}")
    if not success_paths:
        raise FileNotFoundError(f"No success pkl files found in {data_dir}")
    if not failure_paths:
        raise FileNotFoundError(f"No failure pkl files found in {data_dir}")

    for path in success_paths:
        success_data = []
        with open(path, "rb") as f:
            while True:
                try:
                    success_data.extend(pkl.load(f))
                except EOFError:  # 读取完毕
                        break
            
            for trans in success_data:
                trans["labels"] = 1
                # print("trans keys= ",trans["observations"].keys())
                # print("trans tactile_data shape=", trans["observations"]["tactile_data"].shape)
                pos_buffer.insert(trans)
            
    pos_iterator = pos_buffer.get_iterator(
        sample_args={
            "batch_size": FLAGS.batch_size // 2,
        },
        device=sharding.replicate(),
    )
    
    # Create buffer for negative transitions
    neg_buffer = ReplayBuffer(
        observation_space=env.observation_space,
        action_space=env.action_space,
        capacity=10000,
        include_label=True,
    )

    for path in failure_paths:
         failure_data = []
         with open(path, "rb") as f:
            while True:
                try:
                    failure_data.extend(pkl.load(f))
                except EOFError:  # 读取完毕
                    break
            for trans in failure_data:
                trans["labels"] = 0
                neg_buffer.insert(trans)
            
    neg_iterator = neg_buffer.get_iterator(
        sample_args={
            "batch_size": FLAGS.batch_size // 2,
        },
        device=sharding.replicate(),
    )

    print(f"failed buffer size: {len(neg_buffer)}")
    print(f"success buffer size: {len(pos_buffer)}")

    rng = jax.random.PRNGKey(0)
    rng, key = jax.random.split(rng)
    pos_sample = next(pos_iterator)
    neg_sample = next(neg_iterator)
    sample = concat_batches(pos_sample, neg_sample, axis=0)
    print("config.classifier_keys = ", config.classifier_keys)

    rng, key = jax.random.split(rng)
    classifier = create_classifier(key, 
                                   sample["observations"], 
                                   config.classifier_keys,
                                   )

    def data_augmentation_fn(rng, observations):
        for pixel_key in config.classifier_keys:
            observations = observations.copy(
                add_or_replace={
                    pixel_key: batched_random_crop(
                        observations[pixel_key], rng, padding=4, num_batch_dims=2
                    )
                }
            )
        return observations

    @jax.jit
    def train_step(state, batch, key):
        def loss_fn(params):
            logits = state.apply_fn(
                {"params": params}, batch["observations"], rngs={"dropout": key}, train=True
            )
            return optax.sigmoid_binary_cross_entropy(logits, batch["labels"]).mean()

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params)
        logits = state.apply_fn(
            {"params": state.params}, batch["observations"], train=False, rngs={"dropout": key}
        )
        train_accuracy = jnp.mean((nn.sigmoid(logits) >= 0.95) == batch["labels"])

        return state.apply_gradients(grads=grads), loss, train_accuracy

    for epoch in tqdm(range(FLAGS.num_epochs)):
        # Sample equal number of positive and negative examples
        pos_sample = next(pos_iterator)
        neg_sample = next(neg_iterator)
        # Merge and create labels
        batch = concat_batches(
            pos_sample, neg_sample, axis=0
        )
        rng, key = jax.random.split(rng)
        obs = data_augmentation_fn(key, batch["observations"])
        batch = batch.copy(
            add_or_replace={
                "observations": obs,
                "labels": batch["labels"][..., None],
            }
        )
            
        rng, key = jax.random.split(rng)
        classifier, train_loss, train_accuracy = train_step(classifier, batch, key)

        print(
            f"Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}"
        )

    checkpoint_dir = _resolve_script_relative_path(
        FLAGS.checkpoint_dir,
        _task_checkpoint_dir_name(),
    )
    checkpoints.save_checkpoint(
        checkpoint_dir,
        classifier,
        step=FLAGS.num_epochs,
        overwrite=True,
    )
    print(f"[done] checkpoint_dir={checkpoint_dir}")
    # env.close()
    

if __name__ == "__main__":
    app.run(main)
