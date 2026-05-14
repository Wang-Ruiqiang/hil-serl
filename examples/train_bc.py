#!/usr/bin/env python3

import glob
import time
import os, sys, threading, queue, termios, tty, select
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
flags.DEFINE_integer("eval_n_trajs", 0, "Number of trajectories to evaluate.")
flags.DEFINE_integer("eval_checkpoint_step", 60000, "Step to evaluate the checkpoint.")
flags.DEFINE_integer("train_steps", 2000000, "Number of pretraining steps.")
flags.DEFINE_bool("save_video", False, "Save video of the evaluation.")
flags.DEFINE_multi_string("demo_path", None, "Path to the demo data.")
flags.DEFINE_integer("enable_tactile", 1, "evaluate pick or place task.")

robot_urdf_path = "/home/wrq/workspaces/HK_TACEXO_WANG/hil-serl/examples/urdf/denso_robot_with_ati_4.urdf"

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

class KeyReader(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.q = queue.Queue()
        self._stop = threading.Event()
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)  # 立即读取，无需回车

    def run(self):
        try:
            while not self._stop.is_set():
                if sys.stdin in select.select([sys.stdin], [], [], 0.01)[0]:
                    ch = sys.stdin.read(1)
                    self.q.put(ch)
        finally:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)

    def get_key_nowait(self):
        try:
            return self.q.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        self._stop.set()


# def eval(
#     env,
#     bc_agent: BCAgent,
#     sampling_rng,
# ):
#     """
#     This is the actor loop, which runs when "--actor" is set to True.
#     """
#     print("evaluating")
#     ckpt = checkpoints.restore_checkpoint(
#         os.path.abspath(FLAGS.bc_checkpoint_path),
#         bc_agent.state,
#         step=eval_checkpoint_step,
#     )

#     print_green(f"Loaded previous checkpoint at step {eval_checkpoint_step}.")

#     bc_agent = bc_agent.replace(state=ckpt)

#     success_counter = 0
#     time_list = []
#     # data, _= read_utils.read_data(robot_urdf_path, True)
#     data_count = 0
    
#     obs, _ = env.reset()
#     done = False
#     start_time = time.time()
#     while not done:
#         sampling_rng, key = jax.random.split(sampling_rng)

#         print("obs state = ", obs["state"])
#         # obs = data[data_count]["observations"]

#         # print("obs_read state = ", obs["state"])
        
#         actions = bc_agent.sample_actions(observations=obs, seed=key)
#         actions = np.asarray(jax.device_get(actions))
#         actions = np.array(actions, copy=True)
        
#         # ori_index = [3, 0, 1, 2]
#         # tcp_ori = actions[3:7]
#         # actions[3:7] = tcp_ori[ori_index]
#         # actions[:3], actions[3:7] = kinematics_utils.apply_transformation(actions[:3], actions[3:7], palm_lower2denso_end_tf)
        
#         # actions_read = data[data_count]["actions"]

#         next_obs, reward, done, truncated, info = env.step(actions)
#         obs = next_obs
#         if done:
#             if reward:
#                 dt = time.time() - start_time
#                 time_list.append(dt)
#                 print(dt)
#             success_counter += reward
#             print(reward)
#         data_count += 1
#         # if data_count >= len(data):
#         #     print("eval failed")
#         #     break


def eval(env, bc_agent: BCAgent, sampling_rng):
    try:
        print("in eval mode")
        mode = "S1_INFERENCE"
        success_counter = 0
        intervention_label = 0
        time_list = []
        print_green(f"Loaded previous checkpoint at step {FLAGS.eval_checkpoint_step}.")
        # ckpt = checkpoints.restore_checkpoint(
        #     os.path.abspath(FLAGS.bc_checkpoint_path),
        #     bc_agent.state,
        #     step=FLAGS.eval_checkpoint_step,
        # )
        # agent = bc_agent.replace(state=ckpt)

        if FLAGS.exp_name == "tennis_ball_place" or FLAGS.exp_name == "twist_bottle_cap":
            print_green("Loaded previous checkpoint at step 32000.")
            ckpt_pick = checkpoints.restore_checkpoint(
                os.path.abspath(FLAGS.bc_checkpoint_path_pick),
                agent.state,
                step=32000,
            )
            agent_pick = agent.replace(state=ckpt_pick)
        
        obs, _ = env.reset()
        key_reader = KeyReader()
        key_reader.start()
        ckpt_step = FLAGS.eval_checkpoint_step
        done_by_manual = False
        for episode in range(FLAGS.eval_n_trajs):
            done = False
            start_time = time.time()
            
            print_green(f"Loaded previous checkpoint at step {ckpt_step}.")
            ckpt = checkpoints.restore_checkpoint(
                os.path.abspath(FLAGS.bc_checkpoint_path),
                bc_agent.state,
                step=ckpt_step,
            )
            agent = bc_agent.replace(state=ckpt)
            
            while not done:
                sampling_rng, key = jax.random.split(sampling_rng)
                if (FLAGS.exp_name == "tennis_ball_place" or FLAGS.exp_name == "twist_bottle_cap") and mode == "S1_INFERENCE":
                    # -------- 阶段1：只用 agent_s1 做推理，不写入训练 buffer --------
                    actions = agent_pick.sample_actions(
                        observations=jax.device_put(obs),
                        argmax=True,    
                        seed=key
                    )
                    actions = np.asarray(jax.device_get(actions)).copy()

                    next_obs, reward, done, truncated, info = env.step(actions)
                    obs = next_obs
                    if "is_pick" in info:
                        is_pick_task = info["is_pick"]
                    else:
                        is_pick_task = True

                    # ==== 判定任务1完成（你可替换为自己的条件）====
                    if not is_pick_task:
                        print_green("pick task done--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
                        mode = "S2_TRAIN"
                        # 可在此清零统计（可选）
                        intervention_count = 0
                        intervention_steps = 0
                        already_intervened = False
                        continue
                    else:
                        # 任务1未完成就继续 S1 推理
                        continue

                print_green(f"obs[state] =  {obs['state']}")
                actions = agent.sample_actions(
                    observations=jax.device_put(obs),
                    argmax=False,
                    seed=key
                )
                
                # actions = np.asarray(jax.device_get(actions))
                actions = np.asarray(jax.device_get(actions)).copy()
                actions[..., 3:6] = 0.0

                print("actions = ", actions)

                next_obs, reward, done, truncated, info = env.step(actions)
                obs = next_obs
                key = key_reader.get_key_nowait()
                while key is not None:
                    if key == '1':
                        done = True
                        info = dict(info)
                        done_by_manual = True
                        info['succeed'] = True
                    key = key_reader.get_key_nowait()
                    
                if "intervene_action" in info:
                    intervention_label = 1
                if done:
                    if reward:
                        dt = time.time() - start_time
                        time_list.append(dt)
                        print(dt)
                    if not intervention_label or not done_by_manual:
                        success_counter += 1
                    print(f"{success_counter}/{episode + 1}")
                    intervention_label = 0
                    ckpt_step += 20000
                    done_by_manual = False

                    if FLAGS.exp_name == "tennis_ball_pick" or FLAGS.exp_name == "tennis_ball_place" or FLAGS.exp_name == "lid_grip":
                        env.unwrapped.stop_cur_command()
                    if FLAGS.exp_name == "tube_insertion":
                        env.open_hand(steps=20, step_time=0.05)
                        time.sleep(1.5)
                    elif FLAGS.exp_name == "tennis_ball_pick":
                        env.move_up()
                    if FLAGS.save_video:
                        env.unwrapped.save_video_recording(episode)
                    mode = "S1_INFERENCE"
                    input("reset env")
                    obs, _ = env.reset()

        print(f"success rate: {success_counter / FLAGS.eval_n_trajs}")
        print(f"average time: {np.mean(time_list)}")
        return  # after done eval, return and exit
    
    except KeyboardInterrupt:
        pass
    finally:
        # env.save_all_data_on_exit()
        # env.close()
        return

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
    config = NEW_MAPPING[FLAGS.exp_name]()

    assert config.batch_size % num_devices == 0
    assert FLAGS.exp_name in NEW_MAPPING, "Experiment folder not found."
    eval_mode = FLAGS.eval_n_trajs > 0
 
    env = config.get_environment(
        fake_env=not eval_mode,
        save_video=FLAGS.save_video,
        classifier=True,
        enable_tactile=FLAGS.enable_tactile
    )
    env = RecordEpisodeStatistics(env)

    action_mean = [ 0.65431754, -0.19202452, 0.11915473]
    action_std =  [0.09939767, 0.20337829, 0.08846061 ]

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
            project="bc_ball_pick-4-6",
            description=FLAGS.exp_name,
            debug=FLAGS.debug,
        )


        # all_actions = []  # 存储所有动作
        assert FLAGS.demo_path is not None
        for path in FLAGS.demo_path:
            # with open(path, "rb") as f:
            #     transitions = []
            #     while True:
            #         try:
            #             transitions.extend(pkl.load(f))  # 读取并扩展列表
            #         except EOFError:
            #             break  # 读取结束
            #     for transition in transitions:
            #         bc_replay_buffer.insert(transition)
             with open(path, "rb") as f:
                transitions = pkl.load(f)
                for transition in transitions:
                    bc_replay_buffer.insert(transition)

        if FLAGS.bc_checkpoint_path is not None and os.path.exists(
            os.path.join(FLAGS.bc_checkpoint_path, "demo_buffer")
        ):
            for file in glob.glob(
                os.path.join(FLAGS.bc_checkpoint_path, "demo_buffer/*.pkl")
            ):
                with open(file, "rb") as f:
                    transitions = pkl.load(f)
                    for transition in transitions:
                        bc_replay_buffer.insert(transition)
        print_green(f"bc_replay_buffer size: {len(bc_replay_buffer)}")

        # all_actions = np.array(all_actions)
        # action_mean = np.mean(all_actions[:,:3], axis=0)
        # action_std = np.std(all_actions[:,:3], axis=0) + 1e-6  # 防止除以0
        # print("sample_obs=env.observation_space.sample() = ", env.observation_space.sample()["state"].shape)
        # print("env.action_space.sample(), = ", env.action_space.sample().shape)
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
            jax.tree_util.tree_map(jnp.array, bc_agent), sharding.replicate()
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
        rng, sampling_rng = jax.random.split(rng)
        # print("rng = ", rng)
        # print("sampling_rng = ", sampling_rng)

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
            jax.tree_util.tree_map(jnp.array, bc_agent), sharding.replicate()
        )

        bc_ckpt = checkpoints.restore_checkpoint(
            os.path.abspath(FLAGS.bc_checkpoint_path),
            bc_agent.state,
        )
        bc_agent = bc_agent.replace(state=bc_ckpt)

        sampling_rng = jax.device_put(sampling_rng, sharding.replicate())
        print_green("starting actor loop")
        # eval(
        #     env=env,
        #     bc_agent=bc_agent,
        #     sampling_rng=sampling_rng,
        # )
        eval(env=env, bc_agent=bc_agent, sampling_rng=sampling_rng)


if __name__ == "__main__":
    app.run(main)
