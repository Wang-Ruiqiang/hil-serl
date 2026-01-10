#!/usr/bin/env python3

import os, sys, threading, queue, termios, tty, select
import glob
import time
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from absl import app, flags
from flax.training import checkpoints
import copy
import pickle as pkl
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from natsort import natsorted

# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../serl_launcher'))
# sys.path.insert(0, project_root)

# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../serl_robot_infra'))
# sys.path.insert(0, project_root)
from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.agents.continuous.sac_hybrid_single import SACAgentHybridSingleArm
from serl_launcher.agents.continuous.sac_hybrid_dual import SACAgentHybridDualArm
from serl_launcher.utils.timer_utils import Timer
from serl_launcher.utils.train_utils import concat_batches

from agentlace.trainer import TrainerServer, TrainerClient
from agentlace.data.data_store import QueuedDataStore

from serl_launcher.utils.launcher import (
    make_sac_pixel_agent,
    make_trainer_config,
    make_wandb_logger,
)
from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore
from experiments.mappings import NEW_MAPPING
from examples.utils import read_utils

FLAGS = flags.FLAGS

flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_boolean("learner", False, "Whether this is a learner.")
flags.DEFINE_boolean("actor", False, "Whether this is an actor.")
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
flags.DEFINE_multi_string("demo_path", None, "Path to the demo data.")
flags.DEFINE_string("checkpoint_path", None, "Path to save checkpoints.")
flags.DEFINE_string("checkpoint_path_pick", None, "Path to save pick checkpoints.")
flags.DEFINE_integer("eval_checkpoint_step", 0, "Step to evaluate the checkpoint.")
flags.DEFINE_integer("eval_n_trajs", 20, "Number of trajectories to evaluate.")
flags.DEFINE_boolean("save_video", False, "Save video.")
flags.DEFINE_integer("enable_tactile", 0, "evaluate pick or place task.")

flags.DEFINE_boolean(
    "debug", False, "Debug mode."
)  # debug mode will disable wandb logging


devices = jax.local_devices()
num_devices = len(devices)
sharding = jax.sharding.PositionalSharding(devices)
is_end = False

def print_green(x):
    return print("\033[92m {}\033[00m".format(x))

def print_red(x):
    return print("\033[91m {}\033[00m".format(x))

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
##############################################################################


def actor(agent, data_store, intvn_data_store, env, sampling_rng, agent_pick=None):
    """
    This is the actor loop, which runs when "--actor" is set to True.
    """
    
    if FLAGS.eval_checkpoint_step:
        try:
            print("in eval mode")
            mode = "S1_INFERENCE"
            success_counter = 0
            intervention_label = 0
            time_list = []
            print_green(f"Loaded previous checkpoint at step {FLAGS.eval_checkpoint_step}.")
            ckpt = checkpoints.restore_checkpoint(
                os.path.abspath(FLAGS.checkpoint_path),
                agent.state,
                step=FLAGS.eval_checkpoint_step,
            )
            agent = agent.replace(state=ckpt)

            # if FLAGS.exp_name == "tennis_ball_place":
            #     print_green("Loaded previous checkpoint at step 100000.")
            #     ckpt_pick = checkpoints.restore_checkpoint(
            #         os.path.abspath(FLAGS.checkpoint_path_pick),
            #         agent.state,
            #         step=100000,
            #     )
            #     agent_pick = agent.replace(state=ckpt_pick)
            
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
                    os.path.abspath(FLAGS.checkpoint_path),
                    agent.state,
                    step=ckpt_step,
                )
                agent = agent.replace(state=ckpt)
                
                while not done:

                    sampling_rng, key = jax.random.split(sampling_rng)
                    if FLAGS.exp_name == "tennis_ball_place" and mode == "S1_INFERENCE":
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
                        if FLAGS.save_video:
                            env.unwrapped.save_video_recording(episode)
                        print(f"{success_counter}/{episode + 1}")
                        intervention_label = 0
                        ckpt_step += 1000
                        done_by_manual = False
                        if FLAGS.exp_name == "tube_insertion":
                            env.open_hand(steps=20, step_time=0.05)
                            time.sleep(1.5)
                        elif FLAGS.exp_name == "tennis_ball_pick":
                            env.move_up()
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
        
        
        
    start_step = 0

    if FLAGS.checkpoint_path and os.path.exists(FLAGS.checkpoint_path):
        buffer_dir = os.path.join(FLAGS.checkpoint_path, "buffer")
        buffer_pkls = []

        if os.path.exists(buffer_dir):
            buffer_pkls = natsorted(glob.glob(os.path.join(buffer_dir, "transitions_*.pkl")))

        if len(buffer_pkls) > 0:
            last_pkl = os.path.basename(buffer_pkls[-1])
            start_step = int(last_pkl.replace("transitions_", "").replace(".pkl", "")) + 1
        else:
            # 有 ckpt 但还没有 buffer
            start_step = 0

    datastore_dict = {
        "actor_env": data_store,
        "actor_env_intvn": intvn_data_store,
    }

    client = TrainerClient(
        "actor_env",
        FLAGS.ip,
        make_trainer_config(),
        data_stores=datastore_dict,
        wait_for_server=True,
        timeout_ms=3000,
    )

    # Function to update the agent with new params
    def update_params(params):
        nonlocal agent
        agent = agent.replace(state=agent.state.replace(params=params))

    client.recv_network_callback(update_params)

    transitions = []
    demo_transitions = []
    obs, _ = env.reset()
    done = False

    # training loop
    timer = Timer()
    running_return = 0.0
    already_intervened = False
    intervention_count = 0
    intervention_steps = 0
    mode = "S1_INFERENCE"
    pick_steps = 0

    pbar = tqdm.tqdm(range(start_step, config.max_steps), dynamic_ncols=True)
    for step in pbar:
        if FLAGS.exp_name == "tennis_ball_place":
            step = step - pick_steps
        if step > 0 and config.buffer_period > 0 and step % config.buffer_period == 0:
            # dump to pickle file
            buffer_path = os.path.join(FLAGS.checkpoint_path, "buffer")
            demo_buffer_path = os.path.join(FLAGS.checkpoint_path, "demo_buffer")
            if not os.path.exists(buffer_path):
                os.makedirs(buffer_path)
            if not os.path.exists(demo_buffer_path):
                os.makedirs(demo_buffer_path)
            with open(os.path.join(buffer_path, f"transitions_{step}.pkl"), "wb") as f:
                pkl.dump(transitions, f)
                transitions = []
            with open(
                os.path.join(demo_buffer_path, f"transitions_{step}.pkl"), "wb"
            ) as f:
                pkl.dump(demo_transitions, f)
                demo_transitions = []
        
        sampling_rng, key = jax.random.split(sampling_rng)
        if FLAGS.exp_name == "tennis_ball_place" and mode == "S1_INFERENCE":
            # -------- 阶段1：只用 agent_s1 做推理，不写入训练 buffer --------
            pick_steps += 1
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

        
        timer.tick("total")
        with timer.context("sample_actions"):
            print_green(f"obs[state] =  {obs['state']}")
            # if step < config.random_steps:
            #     print("random actions")
            #     actions = env.action_space.sample()
            # else:
            sampling_rng, key = jax.random.split(sampling_rng)
            # print("obs shape = ", obs["state"].shape)
            actions_sample = agent.sample_actions(
                observations=obs,
                seed=key,
            )
            actions = np.asarray(jax.device_get(actions_sample)).copy()
            actions[..., 3:6] = 0.0
            print("actions sampled= ", actions)
            # if actions[..., 6] < 0.0:
            #     random_action = np.random.uniform(0.0, 0.30)
            #     actions[..., 6] = random_action
            
        # Step environment
        with timer.context("step_env"):
            next_obs, reward, done, truncated, info = env.step(actions)
            # print("reward = ", reward)

            # print_red(f"next_obs[state] =  {next_obs['state']}")

            # override the action with the intervention action
            if "intervene_action" in info:
                actions = info.pop("intervene_action")
                # print("intervene_action = ", actions)
                intervention_steps += 1
                if not already_intervened:
                    intervention_count += 1
                already_intervened = True
            else:
                already_intervened = False
            
            # if "is_pick" in info:
            #     is_pick = info["is_pick"]
            # else:
            #     is_pick = True
            state = obs["state"][0]
            if FLAGS.exp_name == "twist_bottle_cap":
                if state[2] < 0.22 and (0.6 < state[0] < 0.8) and (-0.13 < state[1] < -0.05):
                    actions[:3] = np.clip(actions[:3], -0.4, 0.4)
            print("actions = ", actions)
            # print("reward = ", reward)
            # print("done = ", done)
            # input("step done, press to continue")
            running_return += reward
            transition = dict(
                observations=obs,
                next_observations=next_obs,
                actions=actions,
                rewards=reward,
                masks=1.0 - done,
                dones=done,
            )
            # if 'grasp_penalty' in info:
            #     transition['grasp_penalty']= info['grasp_penalty']
            data_store.insert(transition)
            transitions.append(copy.deepcopy(transition))
            if already_intervened:
                intvn_data_store.insert(transition)
                demo_transitions.append(copy.deepcopy(transition))

            obs = next_obs
            # if done and is_pick:
            #     print_green("pick task done--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
            if done:
                print_green(f" task done = {done}")
                info["episode"]["intervention_count"] = intervention_count
                info["episode"]["intervention_steps"] = intervention_steps
                stats = {"environment": info}  # send stats to the learner to log
                client.request("send-stats", stats)
                pbar.set_description(f"last return: {running_return}")
                running_return = 0.0
                intervention_count = 0
                intervention_steps = 0
                already_intervened = False
                client.update()
                mode = "S1_INFERENCE"
                if FLAGS.exp_name == "tube_insertion":
                    env.open_hand(steps=20, step_time=0.05)
                    time.sleep(1.5)
                elif FLAGS.exp_name == "tennis_ball_pick":
                    env.move_up()
                input("reset env")
                obs, _ = env.reset()
                
        timer.tock("total")
        if step % config.log_period == 0:
            stats = {"timer": timer.get_average_times()}
            client.request("send-stats", stats)


##############################################################################

def learner(rng, agent, replay_buffer, demo_buffer, wandb_logger=None):
    """
    The learner loop, which runs when "--learner" is set to True.
    """
    start_step = (
        int(os.path.basename(checkpoints.latest_checkpoint(os.path.abspath(FLAGS.checkpoint_path)))[11:])
        + 1
        if FLAGS.checkpoint_path and os.path.exists(FLAGS.checkpoint_path)
        else 0
    )

    # start_step = 0
    step = start_step

    def stats_callback(type: str, payload: dict) -> dict:
        """Callback for when server receives stats request."""
        assert type == "send-stats", f"Invalid request type: {type}"
        if wandb_logger is not None:
            wandb_logger.log(payload, step=step)
        return {}  # not expecting a response

    # Create server
    server = TrainerServer(make_trainer_config(), request_callback=stats_callback)
    server.register_data_store("actor_env", replay_buffer)
    server.register_data_store("actor_env_intvn", demo_buffer)
    server.start(threaded=True)
    print_green(f"online buffer size: {len(replay_buffer)}")

    # Loop to wait until replay_buffer is filled
    pbar = tqdm.tqdm(
        total=config.training_starts,
        initial=len(replay_buffer),
        desc="Filling up replay buffer",
        position=0,
        leave=True,
    )
    while len(replay_buffer) < config.training_starts:
        pbar.update(len(replay_buffer) - pbar.n)  # Update progress bar
        time.sleep(1)
    pbar.update(len(replay_buffer) - pbar.n)  # Update progress bar
    pbar.close()

    # send the initial network to the actor
    server.publish_network(agent.state.params)
    print_green("sent initial network to actor")

    # 50/50 sampling from RLPD, half from demo and half from online experience
    replay_iterator = replay_buffer.get_iterator(
        sample_args={
            "batch_size": config.batch_size // 2,
            "pack_obs_and_next_obs": True,
        },
        device=sharding.replicate(),
    )
    demo_iterator = demo_buffer.get_iterator(
        sample_args={
            "batch_size": config.batch_size // 2,
            "pack_obs_and_next_obs": True,
        },
        device=sharding.replicate(),
    )

    # wait till the replay buffer is filled with enough data
    timer = Timer()
    
    if isinstance(agent, SACAgent):
        train_critic_networks_to_update = frozenset({"critic"})
        train_networks_to_update = frozenset({"critic", "actor", "temperature"})
    else:
        train_critic_networks_to_update = frozenset({"critic", "grasp_critic"})
        train_networks_to_update = frozenset({"critic", "grasp_critic", "actor", "temperature"})

    for step in tqdm.tqdm(
        range(start_step, config.max_steps), dynamic_ncols=True, desc="learner"
    ):
        # run n-1 critic updates and 1 critic + actor update.
        # This makes training on GPU faster by reducing the large batch transfer time from CPU to GPU
        for critic_step in range(config.cta_ratio - 1):
            with timer.context("sample_replay_buffer"):
                batch = next(replay_iterator)
                demo_batch = next(demo_iterator)
                batch = concat_batches(batch, demo_batch, axis=0)

            with timer.context("train_critics"):
                agent, critics_info = agent.update(
                    batch,
                    networks_to_update=train_critic_networks_to_update,
                )

        with timer.context("train"):
            batch = next(replay_iterator)
            demo_batch = next(demo_iterator)
            batch = concat_batches(batch, demo_batch, axis=0)
            agent, update_info = agent.update(
                batch,
                networks_to_update=train_networks_to_update,
            )
        # publish the updated network
        if step > 0 and step % (config.steps_per_update) == 0:
            agent = jax.block_until_ready(agent)
            server.publish_network(agent.state.params)

        if step % config.log_period == 0 and wandb_logger:
            wandb_logger.log(update_info, step=step)
            wandb_logger.log({"timer": timer.get_average_times()}, step=step)

        if (
            step > 0
            and config.checkpoint_period
            and step % config.checkpoint_period == 0
        ) or is_end:
            checkpoints.save_checkpoint(
                os.path.abspath(FLAGS.checkpoint_path), agent.state, step=step, keep=1000
            )

#############################################################################


def main(_):
    global config
    config = NEW_MAPPING[FLAGS.exp_name]()

    assert config.batch_size % num_devices == 0
    # seed
    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, sampling_rng = jax.random.split(rng)

    assert FLAGS.exp_name in NEW_MAPPING, "Experiment folder not found."
    env = config.get_environment(
        fake_env=FLAGS.learner,
        save_video=FLAGS.save_video,
        classifier=True,
        enable_tactile=FLAGS.enable_tactile
    )
    env = RecordEpisodeStatistics(env)

    # rng, sampling_rng = jax.random.split(rng)
    agent: SACAgent = make_sac_pixel_agent(
        seed=FLAGS.seed,
        sample_obs=env.observation_space.sample(),
        sample_action=env.action_space.sample(),
        image_keys=config.image_keys,
        encoder_type=config.encoder_type,
        discount=config.discount,
        state_weights=config.state_weights,
        image_weights=config.image_weights,
    )
    
    include_grasp_penalty = False
    
    # replicate agent across devices
    # need the jnp.array to avoid a bug where device_put doesn't recognize primitives
    agent = jax.device_put(
        jax.tree_map(jnp.array, agent), sharding.replicate()
    )
    
    latest_ckpt = None
    if FLAGS.checkpoint_path is not None and os.path.exists(FLAGS.checkpoint_path):
        # input("Checkpoint path already exists. Press Enter to resume training.")
        latest_ckpt = checkpoints.latest_checkpoint(FLAGS.checkpoint_path)
    
    if latest_ckpt is not None:
        ckpt = checkpoints.restore_checkpoint(
            os.path.abspath(FLAGS.checkpoint_path),
            agent.state,
        )
        agent = agent.replace(state=ckpt)
        # print_green(f"Loaded previous checkpoint at step {step}.")
        ckpt_number = os.path.basename(
            checkpoints.latest_checkpoint(os.path.abspath(FLAGS.checkpoint_path))
        )[11:]
        print_green(f"Loaded previous checkpoint at step {ckpt_number}.")
        
    include_grasp_penalty = False
    agent_pick = None
    if FLAGS.exp_name == "tennis_ball_place":
        agent_pick: SACAgent = make_sac_pixel_agent(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
            state_weights=config.state_weights,
        )
        # replicate agent across devices
        # need the jnp.array to avoid a bug where device_put doesn't recognize primitives
    
        agent_pick = jax.device_put(
            jax.tree_map(jnp.array, agent_pick), sharding.replicate()
        )
        
        latest_ckpt = None
        if FLAGS.checkpoint_path_pick is not None and os.path.exists(FLAGS.checkpoint_path_pick):
            # input("Checkpoint path already exists. Press Enter to resume training.")
            latest_ckpt = checkpoints.latest_checkpoint(FLAGS.checkpoint_path_pick)

        if latest_ckpt is not None:
            ckpt = checkpoints.restore_checkpoint(
                os.path.abspath(FLAGS.checkpoint_path_pick),
                agent_pick.state,
            )
            agent_pick = agent_pick.replace(state=ckpt)
            # print_green(f"Loaded previous checkpoint at step {step}.")
            ckpt_number = os.path.basename(
                checkpoints.latest_checkpoint(os.path.abspath(FLAGS.checkpoint_path_pick))
            )[11:]
            print_green(f"Loaded previous checkpoint_pick at step {ckpt_number}.")
        
    # if FLAGS.checkpoint_path is not None and os.path.exists(os.path.join(FLAGS.checkpoint_path, "checkpoint*")):

    def create_replay_buffer_and_wandb_logger():
        replay_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys,
            include_grasp_penalty=include_grasp_penalty,
        )
        # set up wandb and logging
        wandb_logger = make_wandb_logger(
            project="twist_bottle_cap-ablation-1-7",
            # project="tube-insertion-ablation-12-27",
            description=FLAGS.exp_name,
            debug=FLAGS.debug,
        )
        return replay_buffer, wandb_logger

    if FLAGS.learner:
        sampling_rng = jax.device_put(sampling_rng, device=sharding.replicate())
        replay_buffer, wandb_logger = create_replay_buffer_and_wandb_logger()
        demo_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys,
            include_grasp_penalty=include_grasp_penalty,
        )

        assert FLAGS.demo_path is not None
        for path in FLAGS.demo_path:
            with open(path, "rb") as f:
                transitions = []
                while True:
                    try:
                        transitions.extend(pkl.load(f))  # 读取并扩展列表
                    except EOFError:
                        break  # 读取结束
                print("len trans = ", len(transitions))
                for transition in transitions:
                    # if 'infos' in transition and 'grasp_penalty' in transition['infos']:
                    #     transition['grasp_penalty'] = transition['infos']['grasp_penalty']
                    demo_buffer.insert(transition)
        print_green(f"demo buffer size: {len(demo_buffer)}")
        print_green(f"online buffer size: {len(replay_buffer)}")

        if FLAGS.checkpoint_path is not None and os.path.exists(
            os.path.join(FLAGS.checkpoint_path, "buffer")
        ):
            for file in glob.glob(os.path.join(FLAGS.checkpoint_path, "buffer/*.pkl")):
                with open(file, "rb") as f:
                    transitions = pkl.load(f)
                    for transition in transitions:
                        replay_buffer.insert(transition)
            print_green(
                f"Loaded previous buffer data. Replay buffer size: {len(replay_buffer)}"
            )

        if FLAGS.checkpoint_path is not None and os.path.exists(
            os.path.join(FLAGS.checkpoint_path, "demo_buffer")
        ):
            for file in glob.glob(
                os.path.join(FLAGS.checkpoint_path, "demo_buffer/*.pkl")
            ):
                with open(file, "rb") as f:
                    transitions = pkl.load(f)
                    for transition in transitions:
                        demo_buffer.insert(transition)
            print_green(
                f"Loaded previous demo buffer data. Demo buffer size: {len(demo_buffer)}"
            )

        # learner loop
        print_green("starting learner loop")
        learner(
            sampling_rng,
            agent,
            replay_buffer,
            demo_buffer=demo_buffer,
            wandb_logger=wandb_logger,
        )

    elif FLAGS.actor:
        sampling_rng = jax.device_put(sampling_rng, sharding.replicate())
        data_store = QueuedDataStore(50000)  # the queue size on the actor
        intvn_data_store = QueuedDataStore(50000)

        # actor loop
        print_green("starting actor loop")
        actor(
            agent,
            data_store,
            intvn_data_store,
            env,
            sampling_rng,
            agent_pick,
        )

    else:
        raise NotImplementedError("Must be either a learner or an actor")

if __name__ == "__main__":
    app.run(main)
