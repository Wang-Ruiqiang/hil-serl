#!/usr/bin/env python3

import glob
import time
import sys
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from absl import app, flags
from flax.training import checkpoints
import os
import pickle as pkl
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../serl_robot_infra'))
sys.path.insert(0, project_root)

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import threading
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState

from serl_launcher.agents.continuous.bc import BCAgent

from serl_launcher.utils.launcher import (
    make_bc_agent,
    make_trainer_config,
    make_wandb_logger,
)

# print(sys.path)
from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore

from experiments.mappings import NEW_MAPPING
from experiments.config import DefaultTrainingConfig
from examples.utils import read_utils


FLAGS = flags.FLAGS

flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
flags.DEFINE_string("bc_checkpoint_path", None, "Path to save checkpoints.")
flags.DEFINE_integer("eval_n_trajs", 1000, "Number of trajectories to evaluate.")
flags.DEFINE_integer("train_steps", 20000, "Number of pretraining steps.")
flags.DEFINE_bool("save_video", False, "Save video of the evaluation.")
flags.DEFINE_multi_string("demo_path", None, "Path to the demo data.")

robot_urdf_path = "/home/ruiqiang/workspaces/HK_TACEXO_WANG/hil-serl/examples/urdf/denso_robot_with_ati_4.urdf"

flags.DEFINE_boolean(
    "debug", False, "Debug mode."
)  # debug mode will disable wandb logging


devices = jax.local_devices()
num_devices = len(devices)
sharding = jax.sharding.PositionalSharding(devices)


def print_green(x):
    return print("\033[92m {}\033[00m".format(x))


def print_yellow(x):
    return print("\033[93m {}\033[00m".format(x))


##############################################################################

def eval(
    env,
    bc_agent: BCAgent,
    sampling_rng,
):
    """
    This is the actor loop, which runs when "--actor" is set to True.
    """
    print("evaluating")
    success_counter = 0
    time_list = []
    data = read_utils.read_data(robot_urdf_path, True)
    data_count = 0
    
    obs, _ = env.reset()
    done = False
    start_time = time.time()
    while not done:
        rng, key = jax.random.split(sampling_rng)
        actions = bc_agent.sample_actions(observations=obs, seed=key)
        actions = np.asarray(jax.device_get(actions))
        print("state obs = ", obs["state"])
        
        obs = data[data_count]["observations"]
        print("obs state = ", obs["state"][:3])
        # print("obs state = ", obs[data_count]["observations"]["state"])
        input("debug")
        sampling_rng, key = jax.random.split(sampling_rng)
        # actions_read = data[data_count]["actions"]

        next_obs, reward, done, truncated, info = env.step(actions)
        obs = next_obs
        if done:
            if reward:
                dt = time.time() - start_time
                time_list.append(dt)
                print(dt)
            success_counter += reward
            print(reward)
        data_count += 1
        if data_count >= len(data):
            print("eval failed")
            break


##############################################################################


def train(
    bc_agent: BCAgent,
    bc_replay_buffer,
    config: DefaultTrainingConfig,
    wandb_logger=None,
):

    bc_replay_iterator = bc_replay_buffer.get_iterator(
        sample_args={
            "batch_size": config.batch_size,
            "pack_obs_and_next_obs": False,
        },
        device=sharding.replicate(), 
    )
    
    # Pretrain BC policy to get started
    for step in tqdm.tqdm(
        range(FLAGS.train_steps),
        dynamic_ncols=True,
        desc="bc_pretraining",
    ):
        batch = next(bc_replay_iterator)
        bc_agent, bc_update_info = bc_agent.update(batch)
        if step % config.log_period == 0 and wandb_logger:
            wandb_logger.log({"bc": bc_update_info}, step=step)
        # if step > FLAGS.train_steps - 100 and step % 10 == 0:
        #     checkpoints.save_checkpoint(
        #         os.path.abspath(FLAGS.bc_checkpoint_path), bc_agent.state, step=step, keep=5
        #     )

        if (
            step > 0
            and config.checkpoint_period
            and step % config.checkpoint_period == 0
        ):
            checkpoints.save_checkpoint(
                os.path.abspath(FLAGS.bc_checkpoint_path), bc_agent.state, step=step, keep=100
            )
    print_green("bc pretraining done and saved checkpoint")


##############################################################################


def main(_):
    config: DefaultTrainingConfig = NEW_MAPPING[FLAGS.exp_name]()

    assert config.batch_size % num_devices == 0
    assert FLAGS.exp_name in NEW_MAPPING, "Experiment folder not found."
    eval_mode = FLAGS.eval_n_trajs > 0
 
    env = config.get_environment(
        fake_env=not eval_mode,
        save_video=FLAGS.save_video,
        classifier=True,
    )
    env = RecordEpisodeStatistics(env)

    if not eval_mode:
        assert not os.path.isdir(
            os.path.join(FLAGS.bc_checkpoint_path, f"checkpoint_{FLAGS.train_steps}")
        )

        bc_replay_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys,
        )

        # set up wandb and logging
        wandb_logger = make_wandb_logger(
            project="hil-serl",
            description=FLAGS.exp_name,
            debug=FLAGS.debug,
        )


        all_actions = []  # 存储所有动作
        assert FLAGS.demo_path is not None
        for path in FLAGS.demo_path:
            with open(path, "rb") as f:
                transitions = []
                while True:
                    try:
                        transitions.extend(pkl.load(f))  # 读取并扩展列表
                    except EOFError:
                        break  # 读取结束
                for transition in transitions:
                    bc_replay_buffer.insert(transition)
                    # print("transition keys = ", transition.keys())
                    # print("transition[observation]state = ", transition["observations"]["state"][:3])
                    # print("transition[actions] = ", transition["actions"][:3])
                    all_actions.append(transition["actions"])
                    # input("debug")
        print_green(f"bc_replay_buffer size: {len(bc_replay_buffer)}")



        bc_replay_iterator = bc_replay_buffer.get_iterator(
            sample_args={
                "batch_size": config.batch_size,
                "pack_obs_and_next_obs": False,
            },
            device=sharding.replicate(), 
        )
        all_actions = []
        for _ in range(len(bc_replay_buffer) // config.batch_size):
            batch = next(bc_replay_iterator)
            all_actions.append(batch["actions"])  # 应该已经是(batch_size, 7)
        all_actions = np.concatenate(all_actions, axis=0)
        action_mean = np.mean(all_actions[:,:3], axis=0)
        action_std = np.std(all_actions[:,:3], axis=0) + 1e-6  # 防止除以0
        print("action_mean = ", action_mean)
        print("action_std = ", action_std)

        bc_agent: BCAgent = make_bc_agent(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            action_mean=action_mean,
            action_std=action_std
        )

        # replicate agent across devices
        # need the jnp.array to avoid a bug where device_put doesn't recognize primitives
        bc_agent: BCAgent = jax.device_put(
            jax.tree_map(jnp.array, bc_agent), sharding.replicate()
    )   

        # learner loop
        print_green("starting learner loop")
        train(
            bc_agent=bc_agent,
            bc_replay_buffer=bc_replay_buffer,
            wandb_logger=wandb_logger,
            config=config,
        )

    else:
        rng = jax.random.PRNGKey(FLAGS.seed)
        sampling_rng = jax.device_put(rng, sharding.replicate())

        bc_agent: BCAgent = make_bc_agent(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
        )

        # replicate agent across devices
        # need the jnp.array to avoid a bug where device_put doesn't recognize primitives
        bc_agent: BCAgent = jax.device_put(
            jax.tree_map(jnp.array, bc_agent), sharding.replicate()
        )

        bc_ckpt = checkpoints.restore_checkpoint(
             os.path.abspath(FLAGS.bc_checkpoint_path),
            bc_agent.state,
        )
        bc_agent = bc_agent.replace(state=bc_ckpt)

        print_green("starting actor loop")
        eval(
            env=env,
            bc_agent=bc_agent,
            sampling_rng=sampling_rng,
        )


if __name__ == "__main__":
    app.run(main)
