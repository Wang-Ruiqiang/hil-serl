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
import gymnasium as gym
from gymnasium.spaces import flatten_space, flatten

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../serl_robot_infra'))
sys.path.insert(0, project_root)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../serl_launcher'))
sys.path.insert(0, project_root)

from serl_launcher.data.data_store import ReplayBuffer
from serl_launcher.utils.train_utils import concat_batches
from serl_launcher.vision.data_augmentations import batched_random_crop
from serl_launcher.networks.reward_classifier import create_classifier


from experiments.mappings import NEW_MAPPING


FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", "twist_bottle_cap", "Name of experiment corresponding to folder.")
flags.DEFINE_integer("num_epochs", 50, "Number of training epochs.")
flags.DEFINE_integer("batch_size", 256, "Batch size.")
flags.DEFINE_integer("is_bottle_twist", 1, "evaluate pick or place task.")
flags.DEFINE_integer("is_ball_pick", 0, "evaluate pick or place task.")
flags.DEFINE_integer("is_pick_task", 0, "evaluate pick or place task.")
flags.DEFINE_integer("is_pick_and_place_task", 0, "evaluate pick or place task.")
flags.DEFINE_integer("enable_tactile", 0, "evaluate pick or place task.")


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
    
    if FLAGS.is_bottle_twist:
        success_paths = glob.glob(os.path.join(os.getcwd(), "classifier_data_bottle_twist", "*success*.pkl"))
    elif FLAGS.is_ball_pick:
        if FLAGS.is_pick_and_place_task:
            success_paths = glob.glob(os.path.join(os.getcwd(), "classifier_data", "*success*.pkl"))
        elif FLAGS.is_pick_task:
            success_paths = glob.glob(os.path.join(os.getcwd(), "classifier_data_pick", "*success*.pkl"))
        else:
            success_paths = glob.glob(os.path.join(os.getcwd(), "classifier_data_place", "*success*.pkl"))

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

    if FLAGS.is_bottle_twist:
        failure_paths = glob.glob(os.path.join(os.getcwd(), "classifier_data_bottle_twist", "*failure*.pkl"))
    elif FLAGS.is_ball_pick:
        if FLAGS.is_pick_and_place_task:
            failure_paths = glob.glob(os.path.join(os.getcwd(), "classifier_data", "*failure*.pkl"))
        elif FLAGS.is_pick_task:
            failure_paths = glob.glob(os.path.join(os.getcwd(), "classifier_data_pick", "*failure*.pkl"))
        else:
            failure_paths = glob.glob(os.path.join(os.getcwd(), "classifier_data_place", "*failure*.pkl"))

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
                                   image_key_weights=config.classifier_key_weights,
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

    if FLAGS.is_bottle_twist:
        checkpoints.save_checkpoint(
            os.path.join(os.getcwd(), "classifier_ckpt_bottle_twist/"),
            classifier,
            step=FLAGS.num_epochs,
            overwrite=True,
        )
    elif FLAGS.is_ball_pick:
        if FLAGS.is_pick_and_place_task:
            checkpoints.save_checkpoint(
                os.path.join(os.getcwd(), "classifier_ckpt_ball_pick/"),
                classifier,
                step=FLAGS.num_epochs,
                overwrite=True,
            )
        elif FLAGS.is_pick_task:
            checkpoints.save_checkpoint(
                os.path.join(os.getcwd(), "classifier_ckpt_pick/"),
                classifier,
                step=FLAGS.num_epochs,
                overwrite=True,
            )
        else:
            checkpoints.save_checkpoint(
                os.path.join(os.getcwd(), "classifier_ckpt_place/"),
                classifier,
                step=FLAGS.num_epochs,
                overwrite=True,
            )
    # env.close()
    

if __name__ == "__main__":
    app.run(main)