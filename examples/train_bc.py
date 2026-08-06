#!/usr/bin/env python3

import glob
import os
import sys
import time

import cv2
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from absl import app, flags
from flax.training import checkpoints
import pickle as pkl
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from natsort import natsorted
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../serl_robot_infra'))
sys.path.insert(0, project_root)

from serl_launcher.agents.continuous.bc import BCAgent

from serl_launcher.utils.launcher import (
    make_bc_agent,
    make_wandb_logger,
)

from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore

from experiments.mappings import NEW_MAPPING
from experiments.config import DefaultTrainingConfig
from examples.utils.runtime import (
    KeyReader,
    MULTI_STAGE_EXP_NAMES,
    STOP_COMMAND_EXP_NAMES,
    print_green,
)


FLAGS = flags.FLAGS

DEFAULT_DEMO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "bc_data"))

flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
flags.DEFINE_string("bc_checkpoint_path", None, "Path to save checkpoints.")
flags.DEFINE_string("bc_checkpoint_path_pick", None, "Path to the stage-1 pick checkpoint.")
flags.DEFINE_integer("eval_n_trajs", 0, "Number of trajectories to evaluate.")
flags.DEFINE_integer("eval_checkpoint_step", 0, "Step to evaluate the checkpoint.")
flags.DEFINE_integer(
    "eval_checkpoint_period",
    0,
    "If > 0, advance eval checkpoint step by this period after each trajectory. Defaults to 0 to reuse the same checkpoint.",
)
flags.DEFINE_integer("train_steps", 5000000, "Number of pretraining steps.")
flags.DEFINE_integer(
    "checkpoint_period",
    -1,
    "BC checkpoint save period. Uses the task config value when set to -1; disables periodic saving when set to 0.",
)
flags.DEFINE_bool("save_video", False, "Save video of the evaluation.")
flags.DEFINE_multi_string(
    "demo_path",
    [DEFAULT_DEMO_PATH],
    "Path to demo pkl file(s) or directories.",
)
flags.DEFINE_integer("enable_tactile", 1, "evaluate pick or place task.")
flags.DEFINE_string(
    "wandb_description",
    None,
    "WandB run name. Defaults to bc-<exp_name>-<month>-<day>.",
)

flags.DEFINE_boolean(
    "debug", False, "Debug mode."
)  # debug mode will disable wandb logging


devices = jax.local_devices()
num_devices = len(devices)
sharding = jax.sharding.PositionalSharding(devices)


##############################################################################


def _iter_demo_files(paths):
    demo_files = []
    for path in paths or []:
        path = os.path.expanduser(path)
        if os.path.isdir(path):
            path_pkls = glob.glob(os.path.join(path, "*.pkl"))
            if not path_pkls and os.path.isdir(os.path.join(path, "demo_buffer")):
                path = os.path.join(path, "demo_buffer")
                path_pkls = glob.glob(os.path.join(path, "*.pkl"))
            demo_files.extend(path_pkls)
        else:
            matches = glob.glob(path)
            demo_files.extend(matches if matches else [path])
    return natsorted(set(demo_files))


def _load_pickle_stream(path):
    transitions = []
    with open(path, "rb") as f:
        while True:
            try:
                data = pkl.load(f)
            except EOFError:
                break
            transitions.extend(data)
    return transitions


def _resize_image_to_shape(image, target_shape):
    if image.shape == target_shape:
        return image

    target_h, target_w = target_shape[-3], target_shape[-2]
    if image.ndim == 4:
        resized = [
            cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
            for frame in image
        ]
        image = np.stack(resized, axis=0)
    elif image.ndim == 3:
        image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)
    else:
        raise ValueError(f"Unsupported image shape {image.shape}; expected 3D or 4D image.")

    return image.astype(np.uint8, copy=False)


def _normalize_transition_images(transition, observation_space, image_keys):
    transition = transition.copy()
    for obs_name in ("observations", "next_observations"):
        transition[obs_name] = transition[obs_name].copy()
        for image_key in image_keys:
            if image_key not in transition[obs_name]:
                continue
            target_shape = observation_space[image_key].shape
            transition[obs_name][image_key] = _resize_image_to_shape(
                transition[obs_name][image_key],
                target_shape,
            )
    return transition


def _count_demo_episodes(transitions):
    num_episodes = 0
    num_successes = 0
    episode_return = 0.0
    for transition in transitions:
        episode_return += float(transition.get("rewards", 0.0))
        if transition.get("dones", False):
            num_episodes += 1
            if episode_return > 0.0:
                num_successes += 1
            episode_return = 0.0
    return num_episodes, num_successes


def _dated_run_name():
    return f"bc-{FLAGS.exp_name}-{int(time.strftime('%m'))}-{int(time.strftime('%d'))}"


def _checkpoint_period(config):
    return config.checkpoint_period if FLAGS.checkpoint_period < 0 else FLAGS.checkpoint_period


def _save_bc_checkpoint(bc_agent, step):
    checkpoints.save_checkpoint(
        os.path.abspath(FLAGS.bc_checkpoint_path),
        bc_agent.state,
        step=step,
        keep=100,
    )


def _is_multi_stage_task():
    return FLAGS.exp_name in MULTI_STAGE_EXP_NAMES


def _sample_bc_action(agent, obs, key, *, argmax):
    actions = agent.sample_actions(
        observations=jax.device_put(obs),
        argmax=argmax,
        seed=key,
    )
    actions = np.asarray(jax.device_get(actions)).copy()
    actions[..., 3:6] = 0.0
    return actions


def _restore_bc_checkpoint(agent, path, *, step=None, label="BC checkpoint"):
    assert path is not None, f"{label} path is required."
    ckpt = checkpoints.restore_checkpoint(os.path.abspath(path), agent.state, step=step)
    print_green(f"Loaded {label}{'' if step is None else f' at step {step}'}: {path}")
    return agent.replace(state=ckpt)


def _latest_checkpoint_step(path):
    if path is None or not os.path.exists(path):
        return 0

    latest_ckpt = checkpoints.latest_checkpoint(os.path.abspath(path))
    if latest_ckpt is None:
        return 0

    basename = os.path.basename(latest_ckpt)
    if not basename.startswith("checkpoint_"):
        return 0
    return int(basename.replace("checkpoint_", ""))


def _run_stage1_until_complete(agent_pick, env, obs, key):
    actions = _sample_bc_action(agent_pick, obs, key, argmax=True)
    next_obs, reward, done, truncated, info = env.step(actions)
    is_pick_task = info.get("is_pick", True)
    if not is_pick_task:
        print_green("stage-1 pick task done")
        return next_obs, True
    return next_obs, False


def _reset_eval_env(env, episode):
    if FLAGS.exp_name in STOP_COMMAND_EXP_NAMES:
        env.unwrapped.stop_cur_command()
    if FLAGS.exp_name == "tube_insertion":
        env.open_hand(steps=20, step_time=0.05)
        time.sleep(1.5)
    elif FLAGS.exp_name == "tennis_ball_pick":
        env.move_up()
    if FLAGS.save_video:
        env.unwrapped.save_video_recording(episode)
    input("reset env")
    return env.reset()[0]


def eval(env, bc_agent: BCAgent, sampling_rng):
    key_reader = None
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

        agent_pick = None
        if _is_multi_stage_task():
            assert (
                FLAGS.bc_checkpoint_path_pick is not None
            ), "bc_checkpoint_path_pick is required for multi-stage eval."
            agent_pick = _restore_bc_checkpoint(
                bc_agent,
                FLAGS.bc_checkpoint_path_pick,
                step=32000,
                label="stage-1 BC checkpoint",
            )
        
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
                if _is_multi_stage_task() and mode == "S1_INFERENCE":
                    obs, stage1_done = _run_stage1_until_complete(agent_pick, env, obs, key)
                    if stage1_done:
                        mode = "S2_TRAIN"
                    continue

                print_green(f"obs[state] =  {obs['state']}")
                actions = _sample_bc_action(agent, obs, key, argmax=False)

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
                    if FLAGS.eval_checkpoint_period > 0:
                        ckpt_step += FLAGS.eval_checkpoint_period
                    done_by_manual = False

                    mode = "S1_INFERENCE"
                    obs = _reset_eval_env(env, episode)

        print(f"success rate: {success_counter / FLAGS.eval_n_trajs}")
        print(f"average time: {np.mean(time_list)}")
        return  # after done eval, return and exit
    
    except KeyboardInterrupt:
        pass
    finally:
        if key_reader is not None:
            key_reader.stop()

##############################################################################


def train(
    bc_agent: BCAgent,
    bc_replay_buffer,
    config: DefaultTrainingConfig,
    start_step=0,
    wandb_logger=None,
):
    checkpoint_period = _checkpoint_period(config)

    bc_replay_iterator = bc_replay_buffer.get_iterator(
        sample_args={
            "batch_size": config.batch_size,
            "pack_obs_and_next_obs": False,
        },
        device=sharding.replicate(), 
    )
    
    # Pretrain BC policy to get started
    for step in tqdm.tqdm(
        range(start_step, FLAGS.train_steps),
        initial=start_step,
        total=FLAGS.train_steps,
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
            and checkpoint_period
            and step % checkpoint_period == 0
        ):
            _save_bc_checkpoint(bc_agent, step)
    _save_bc_checkpoint(bc_agent, FLAGS.train_steps)
    print_green(f"saved final BC checkpoint at step {FLAGS.train_steps}")
    print_green("bc pretraining done")


##############################################################################


def main(_):
    assert FLAGS.exp_name in NEW_MAPPING, "Experiment folder not found."
    config = NEW_MAPPING[FLAGS.exp_name]()

    assert config.batch_size % num_devices == 0
    eval_mode = FLAGS.eval_n_trajs > 0
    run_name = FLAGS.wandb_description or _dated_run_name()
    wandb_project = _dated_run_name()
    bc_checkpoint_path = FLAGS.bc_checkpoint_path or run_name
 
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
        bc_replay_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys,
        )

        # set up wandb and logging
        wandb_logger = make_wandb_logger(
            project=wandb_project,
            description=run_name,
            debug=FLAGS.debug,
        )
        FLAGS.bc_checkpoint_path = bc_checkpoint_path
        print_green(f"BC checkpoint path: {FLAGS.bc_checkpoint_path}")
        print_green(f"WandB project: {wandb_project}")
        print_green(f"WandB run name: {run_name}")


        # all_actions = []  # 存储所有动作
        assert FLAGS.demo_path is not None
        demo_files = _iter_demo_files(FLAGS.demo_path)
        assert demo_files, f"No demo pkl files found from --demo_path={FLAGS.demo_path}"
        total_transitions = 0
        total_episodes = 0
        total_successes = 0
        for path in demo_files:
            transitions = _load_pickle_stream(path)
            num_episodes, num_successes = _count_demo_episodes(transitions)
            total_transitions += len(transitions)
            total_episodes += num_episodes
            total_successes += num_successes
            print_green(
                f"Loaded {len(transitions)} transitions, "
                f"{num_successes}/{num_episodes} successful demos from {path}"
            )
            for transition in transitions:
                transition = _normalize_transition_images(
                    transition,
                    env.observation_space,
                    config.image_keys,
                )
                bc_replay_buffer.insert(transition)
        print_green(
            f"Loaded demo summary: {total_transitions} transitions, "
            f"{total_successes}/{total_episodes} successful demos"
        )

        if FLAGS.bc_checkpoint_path is not None and os.path.exists(
            os.path.join(FLAGS.bc_checkpoint_path, "demo_buffer")
        ):
            for file in _iter_demo_files([os.path.join(FLAGS.bc_checkpoint_path, "demo_buffer")]):
                transitions = _load_pickle_stream(file)
                num_episodes, num_successes = _count_demo_episodes(transitions)
                print_green(
                    f"Loaded resume demo buffer: {len(transitions)} transitions, "
                    f"{num_successes}/{num_episodes} successful demos from {file}"
                )
                for transition in transitions:
                    transition = _normalize_transition_images(
                        transition,
                        env.observation_space,
                        config.image_keys,
                    )
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

        start_step = _latest_checkpoint_step(FLAGS.bc_checkpoint_path)
        if start_step > 0:
            bc_agent = _restore_bc_checkpoint(
                bc_agent,
                FLAGS.bc_checkpoint_path,
                step=start_step,
                label="BC resume checkpoint",
            )
            start_step += 1
            print_green(f"Resuming BC training from step {start_step}.")

        assert (
            start_step < FLAGS.train_steps
        ), f"Latest checkpoint step {start_step - 1} is already >= train_steps={FLAGS.train_steps}."

        # learner loop
        print_green("starting learner loop")
        train(
            bc_agent=bc_agent,
            bc_replay_buffer=bc_replay_buffer,
            start_step=start_step,
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
