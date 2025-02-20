import glob
import os
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

from serl_launcher.data.data_store import ReplayBuffer
from serl_launcher.utils.train_utils import concat_batches
from serl_launcher.vision.data_augmentations import batched_random_crop
from serl_launcher.networks.reward_classifier import create_classifier

from experiments.mappings import NEW_MAPPING


FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", "tennis_ball_pick", "Name of experiment corresponding to folder.")
flags.DEFINE_integer("num_epochs", 150, "Number of training epochs.")
flags.DEFINE_integer("batch_size", 256, "Batch size.")

# camera_keys = ["front_camera", "side_camera"]
# classifier_keys = ["front_camera", "side_camera"]
# observation_space = gym.spaces.Dict(
#     {
#         "state": gym.spaces.Dict(
#             {
#                 "tcp_pos": gym.spaces.Box(
#                     -np.inf, np.inf, shape=(3,)
#                 ),
#                 "tcp_ori": gym.spaces.Box(
#                     -np.inf, np.inf, shape=(4,)
#                 ),
#                 "gripper_pose": gym.spaces.Box(-np.inf, np.inf, shape=(16,)),
#             }
#         ),
#         "images": gym.spaces.Dict(
#             {key: gym.spaces.Box(0, 255, shape=(480, 640, 3), dtype=np.uint8) 
#                         for key in camera_keys}
#         ),
#     }
# )

# action_space = gym.spaces.Box(
#         np.ones((23,), dtype=np.float32) * -1,
#         np.ones((23,), dtype=np.float32),
#     )

# def space_stack(space: gym.Space, repeat: int):
#     if isinstance(space, gym.spaces.Box):
#         return gym.spaces.Box(
#             low=np.repeat(space.low[None], repeat, axis=0),
#             high=np.repeat(space.high[None], repeat, axis=0),
#             dtype=space.dtype,
#         )
#     elif isinstance(space, gym.spaces.Discrete):
#         return gym.spaces.MultiDiscrete([space.n] * repeat)
#     elif isinstance(space, gym.spaces.Dict):
#         return gym.spaces.Dict(
#             {k: space_stack(v, repeat) for k, v in space.spaces.items()}
#         )
#     else:
#         raise TypeError()
    

def main(_):
    assert FLAGS.exp_name in NEW_MAPPING, 'Experiment folder not found.'
    config = NEW_MAPPING[FLAGS.exp_name]()
    env = config.get_environment(fake_env=True, save_video=False, classifier=False)

    devices = jax.local_devices()
    sharding = jax.sharding.PositionalSharding(devices)
    
    # stack_observation_space = space_stack(observation_space, 1)
    
    # Create buffer for positive transitions
    print("ReplayBuffer module:", ReplayBuffer.__module__)
    print("ReplayBuffer doc:", ReplayBuffer.__doc__)
    pos_buffer = ReplayBuffer(
        observation_space=env.observation_space,
        action_space=env.action_space,
        capacity=10000,
        include_label=True,
    )

    success_paths = glob.glob(os.path.join(os.getcwd(), "classifier_data", "*success*.pkl"))
    for path in success_paths:
        success_data = pkl.load(open(path, "rb"))
        for trans in success_data:
            # if "images" in trans['observations'].keys():
            #     continue
            trans["labels"] = 1
            # trans['actions'] = env.action_space.sample()

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
    failure_paths = glob.glob(os.path.join(os.getcwd(), "classifier_data", "*failure*.pkl"))
    for path in failure_paths:
        failure_data = pkl.load(
            open(path, "rb")
        )
        for trans in failure_data:
            # if "images" in trans['observations'].keys():
            #     continue
            trans["labels"] = 0
            # trans['actions'] = env.action_space.sample()

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
        train_accuracy = jnp.mean((nn.sigmoid(logits) >= 0.5) == batch["labels"])

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

    checkpoints.save_checkpoint(
        os.path.join(os.getcwd(), "classifier_ckpt/"),
        classifier,
        step=FLAGS.num_epochs,
        overwrite=True,
    )
    

if __name__ == "__main__":
    app.run(main)