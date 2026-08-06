#!/usr/bin/env python3

import glob
import json
import os
import re
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

from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.utils.timer_utils import Timer
from serl_launcher.utils.train_utils import concat_batches

from agentlace.trainer import TrainerServer, TrainerClient
from agentlace.data.data_store import QueuedDataStore

from serl_launcher.utils.launcher import (
    make_sac_pixel_agent,
    make_sac_pixel_agent_hybrid_single_arm,
    make_trainer_config,
    make_wandb_logger,
)
from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore
from experiments.mappings import NEW_MAPPING
from examples.utils.runtime import (
    KeyReader,
    MULTI_STAGE_EXP_NAMES,
    STOP_COMMAND_EXP_NAMES,
    print_green,
    print_yellow,
)

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
flags.DEFINE_integer("stage1_checkpoint_step", 32000, "Checkpoint step for the first-stage pick policy.")
flags.DEFINE_integer("eval_n_trajs", 20, "Number of trajectories to evaluate.")
flags.DEFINE_integer(
    "eval_checkpoint_step_interval",
    0,
    "If > 0, load the next checkpoint after each eval episode by increasing eval_checkpoint_step by this interval.",
)
flags.DEFINE_boolean("save_video", True, "Save video.")
flags.DEFINE_integer("enable_tactile", 1, "evaluate pick or place task.")
flags.DEFINE_integer(
    "eval_max_episode_steps",
    0,
    "Maximum eval steps per episode. 0 relies on the task env MAX_EPISODE_LENGTH.",
)

flags.DEFINE_boolean(
    "debug", False, "Debug mode."
)  # debug mode will disable wandb logging


devices = jax.local_devices()
num_devices = len(devices)
sharding = jax.sharding.PositionalSharding(devices)
is_end = False

##############################################################################


def is_multi_stage_task():
    return FLAGS.exp_name in MULTI_STAGE_EXP_NAMES


def _latest_checkpoint_step(path):
    if path is None or not os.path.exists(path):
        return 0
    latest = checkpoints.latest_checkpoint(os.path.abspath(path))
    if latest is None:
        return 0
    basename = os.path.basename(latest)
    if not basename.startswith("checkpoint_"):
        return 0
    return int(basename.replace("checkpoint_", ""))


def _date_from_checkpoint_path(path):
    if path is None:
        return None
    basename = os.path.basename(os.path.normpath(path))
    match = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})", basename)
    if match is None:
        return None
    month = int(match.group(2))
    day = int(match.group(3))
    return f"{month}-{day}"


def _wandb_project_date():
    if _latest_checkpoint_step(FLAGS.checkpoint_path) > 0:
        checkpoint_date = _date_from_checkpoint_path(FLAGS.checkpoint_path)
        if checkpoint_date is not None:
            return checkpoint_date
    return time.strftime("%m-%d").lstrip("0").replace("-0", "-")


def restore_agent_checkpoint(agent, path, *, step=None, label="checkpoint"):
    assert path is not None, f"{label} path is required."
    ckpt = checkpoints.restore_checkpoint(
        os.path.abspath(path),
        agent.state,
        step=step,
    )
    print_green(f"Loaded {label}{'' if step is None else f' at step {step}'}: {path}")
    return agent.replace(state=ckpt)


def sample_policy_action(agent, obs, key, *, argmax):
    actions = agent.sample_actions(
        observations=jax.device_put(obs),
        argmax=argmax,
        seed=key,
    )
    actions = np.asarray(jax.device_get(actions)).copy()
    actions[..., 3:6] = 0.0
    return actions


def run_stage1_until_complete(agent_pick, env, obs, key):
    actions = sample_policy_action(agent_pick, obs, key, argmax=True)
    next_obs, reward, done, truncated, info = env.step(actions)
    is_pick_task = info.get("is_pick", True)
    if not is_pick_task:
        print_green("stage-1 pick task done")
        return next_obs, True
    return next_obs, False


def reset_after_episode(env, episode=None):
    if FLAGS.exp_name in STOP_COMMAND_EXP_NAMES:
        env.unwrapped.stop_cur_command()
    if _is_eval_mode() and FLAGS.save_video and episode is not None:
        env.unwrapped.save_video_recording(episode)
    if FLAGS.exp_name == "tube_insertion":
        env.open_hand(steps=20, step_time=0.05)
        time.sleep(1.5)
    elif FLAGS.exp_name == "tennis_ball_pick":
        env.move_up()
    input("reset env")
    return env.reset()[0]


def _reached_eval_max_episode_steps(eval_episode_steps):
    return (
        FLAGS.eval_max_episode_steps > 0
        and eval_episode_steps >= FLAGS.eval_max_episode_steps
    )


def _save_eval_video(env, ckpt_step, episode):
    if not (_is_eval_mode() and FLAGS.save_video):
        return None
    video_id = f"ckpt_{ckpt_step}_episode_{episode}"
    env.unwrapped.save_video_recording(video_id)
    return os.path.abspath(os.path.join("videos", video_id))


def _is_eval_mode():
    return bool(FLAGS.actor and FLAGS.eval_checkpoint_step)


def _record_eval_result(
    records,
    *,
    ckpt_step,
    episode,
    reward,
    duration,
    episode_steps,
    video_dir,
    termination_reason,
):
    reward_value = float(np.asarray(reward).item())
    records.append(
        {
            "checkpoint_step": int(ckpt_step),
            "episode": int(episode),
            "success": bool(reward_value),
            "reward": reward_value,
            "duration_sec": float(duration),
            "episode_steps": int(episode_steps),
            "termination_reason": termination_reason,
            "video_dir": video_dir,
        }
    )


def _write_eval_summary(records, *, interrupted):
    if not FLAGS.save_video:
        return None

    summary_dir = os.path.abspath("videos")
    os.makedirs(summary_dir, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    md_path = os.path.join(summary_dir, f"eval_summary_{timestamp}.md")
    json_path = os.path.join(summary_dir, f"eval_summary_{timestamp}.json")

    success_records = [record for record in records if record["success"]]
    success_rate = len(success_records) / len(records) if records else 0.0
    success_durations = [record["duration_sec"] for record in success_records]
    avg_success_time = float(np.mean(success_durations)) if success_durations else None

    summary = {
        "exp_name": FLAGS.exp_name,
        "checkpoint_path": FLAGS.checkpoint_path,
        "checkpoint_path_pick": FLAGS.checkpoint_path_pick,
        "stage1_checkpoint_step": FLAGS.stage1_checkpoint_step,
        "eval_checkpoint_step": FLAGS.eval_checkpoint_step,
        "eval_checkpoint_step_interval": FLAGS.eval_checkpoint_step_interval,
        "eval_max_episode_steps": FLAGS.eval_max_episode_steps,
        "interrupted": interrupted,
        "total_episodes": len(records),
        "success_count": len(success_records),
        "success_rate": success_rate,
        "average_success_duration_sec": avg_success_time,
        "records": records,
    }

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    lines = [
        "# Eval Summary\n\n",
        f"- exp_name: `{FLAGS.exp_name}`\n",
        f"- checkpoint_path: `{FLAGS.checkpoint_path}`\n",
        f"- eval_checkpoint_step: `{FLAGS.eval_checkpoint_step}`\n",
        f"- eval_checkpoint_step_interval: `{FLAGS.eval_checkpoint_step_interval}`\n",
        f"- eval_max_episode_steps: `{FLAGS.eval_max_episode_steps}`\n",
        f"- interrupted: `{interrupted}`\n",
        f"- success: `{len(success_records)}/{len(records)}`\n",
        f"- success_rate: `{success_rate:.2%}`\n",
        "- average_success_duration_sec: "
        f"`{avg_success_time:.2f}`\n" if avg_success_time is not None else "- average_success_duration_sec: `nan`\n",
        "\n## Successful Episodes\n\n",
        "| checkpoint | episode | duration_sec | steps | video_dir |\n",
        "|---:|---:|---:|---:|---|\n",
    ]
    for record in success_records:
        lines.append(
            f"| {record['checkpoint_step']} | {record['episode']} | "
            f"{record['duration_sec']:.2f} | {record['episode_steps']} | "
            f"`{record['video_dir']}` |\n"
        )

    lines.extend(
        [
            "\n## All Episodes\n\n",
            "| checkpoint | episode | success | reward | duration_sec | steps | termination | video_dir |\n",
            "|---:|---:|:---:|---:|---:|---:|---|---|\n",
        ]
    )
    for record in records:
        lines.append(
            f"| {record['checkpoint_step']} | {record['episode']} | "
            f"{str(record['success']).lower()} | {record['reward']:.1f} | "
            f"{record['duration_sec']:.2f} | {record['episode_steps']} | "
            f"{record['termination_reason']} | `{record['video_dir']}` |\n"
        )

    with open(md_path, "w") as f:
        f.writelines(lines)

    print_green(f"Saved eval summary: {md_path}")
    print_green(f"Saved eval summary json: {json_path}")
    return md_path


def iter_transition_files(paths):
    files = []
    for path in paths or []:
        path = os.path.expanduser(path)
        if os.path.isdir(path):
            files.extend(glob.glob(os.path.join(path, "*.pkl")))
        else:
            matches = glob.glob(path)
            files.extend(matches if matches else [path])
    return natsorted(set(files))


def load_transition_stream(path):
    transitions = []
    with open(path, "rb") as f:
        while True:
            try:
                transitions.extend(pkl.load(f))
            except EOFError:
                break
    return transitions


def ensure_penalty_keys(transition):
    transition = transition.copy()
    transition.setdefault("grasp_penalty", np.float32(0.0))
    transition.setdefault("robot_arm_penalty", np.float32(0.0))
    return transition


def actor(agent, data_store, intvn_data_store, env, sampling_rng, agent_pick=None):
    """
    This is the actor loop, which runs when "--actor" is set to True.
    """
    
    if FLAGS.eval_checkpoint_step:
        key_reader = None
        eval_records = []
        interrupted = False
        try:
            print("in eval mode")
            eval_steps = [FLAGS.eval_checkpoint_step]
            if FLAGS.eval_checkpoint_step_interval > 0:
                latest_step = _latest_checkpoint_step(FLAGS.checkpoint_path)
                if latest_step < FLAGS.eval_checkpoint_step:
                    print_yellow(
                        f"Latest checkpoint step {latest_step} is smaller than "
                        f"eval_checkpoint_step {FLAGS.eval_checkpoint_step}."
                    )
                else:
                    eval_steps = list(
                        range(
                            FLAGS.eval_checkpoint_step,
                            latest_step + 1,
                            FLAGS.eval_checkpoint_step_interval,
                        )
                    )
                    print_green(
                        f"Evaluating checkpoints from {FLAGS.eval_checkpoint_step} "
                        f"to {latest_step} every {FLAGS.eval_checkpoint_step_interval} steps."
                    )

            if is_multi_stage_task():
                stage1_template = agent_pick if agent_pick is not None else agent
                agent_pick = restore_agent_checkpoint(
                    stage1_template,
                    FLAGS.checkpoint_path_pick,
                    step=FLAGS.stage1_checkpoint_step,
                    label="stage-1 checkpoint",
                )
            
            key_reader = KeyReader()
            key_reader.start()
            done_by_manual = False
            total_success_counter = 0
            total_episode_counter = 0
            total_time_list = []
            for ckpt_index, ckpt_step in enumerate(eval_steps):
                try:
                    agent = restore_agent_checkpoint(
                        agent,
                        FLAGS.checkpoint_path,
                        step=ckpt_step,
                        label="eval checkpoint",
                    )
                except ValueError as exc:
                    print_yellow(f"Skipping checkpoint_{ckpt_step}: {exc}")
                    break
                print_green(f"Evaluating checkpoint step {ckpt_step}")
                success_counter = 0
                ckpt_episode_counter = 0
                time_list = []
                n_episodes = 1 if FLAGS.eval_checkpoint_step_interval > 0 else FLAGS.eval_n_trajs
                for episode in range(n_episodes):
                    obs = reset_after_episode(env) if episode > 0 or ckpt_index > 0 else env.reset()[0]
                    done = False
                    mode = "S1_INFERENCE" if is_multi_stage_task() else "S2_TRAIN"
                    intervention_label = 0
                    episode_steps = 0
                    start_time = time.time()

                    while not done:

                        sampling_rng, key = jax.random.split(sampling_rng)
                        if is_multi_stage_task() and mode == "S1_INFERENCE":
                            obs, stage1_done = run_stage1_until_complete(agent_pick, env, obs, key)
                            episode_steps += 1
                            if _reached_eval_max_episode_steps(episode_steps):
                                reward = 0
                                print_yellow(
                                    f"Reached eval max episode length {FLAGS.eval_max_episode_steps}."
                                )
                                video_dir = _save_eval_video(env, ckpt_step, episode)
                                _record_eval_result(
                                    eval_records,
                                    ckpt_step=ckpt_step,
                                    episode=episode,
                                    reward=reward,
                                    duration=time.time() - start_time,
                                    episode_steps=episode_steps,
                                    video_dir=video_dir,
                                    termination_reason="max_length",
                                )
                                ckpt_episode_counter += 1
                                total_episode_counter += 1
                                print(f"checkpoint_{ckpt_step}: {success_counter}/{episode + 1}")
                                done_by_manual = False
                                break
                            if stage1_done:
                                mode = "S2_TRAIN"
                            continue

                        print_green(f"obs[state] =  {obs['state']}")
                        actions = sample_policy_action(agent, obs, key, argmax=False)

                        print("actions = ", actions)

                        next_obs, reward, done, truncated, info = env.step(actions)
                        episode_steps += 1
                        obs = next_obs
                        key = key_reader.get_key_nowait()
                        while key is not None:
                            if key == '1':
                                done = True
                                reward = 1
                                info = dict(info)
                                done_by_manual = True
                                info['succeed'] = True
                            elif key == '2':
                                done = True
                                reward = 0
                                info = dict(info)
                                done_by_manual = True
                                info['succeed'] = False
                            key = key_reader.get_key_nowait()

                        max_length_done = _reached_eval_max_episode_steps(episode_steps)
                        if max_length_done and not done:
                            done = True
                            reward = 0
                            print_yellow(
                                f"Reached eval max episode length {FLAGS.eval_max_episode_steps}."
                            )
                            
                        if "intervene_action" in info:
                            intervention_label = 1
                        if done:
                            reward_value = float(np.asarray(reward).item())
                            dt = time.time() - start_time
                            if reward_value:
                                time_list.append(dt)
                                total_time_list.append(dt)
                                print(dt)
                            success_counter += int(bool(reward_value))
                            total_success_counter += int(bool(reward_value))
                            ckpt_episode_counter += 1
                            total_episode_counter += 1
                            print(f"checkpoint_{ckpt_step}: {success_counter}/{episode + 1}")
                            intervention_label = 0
                            termination_reason = "success" if reward_value else "env_done"
                            if truncated:
                                termination_reason = "truncated"
                            if max_length_done:
                                termination_reason = "max_length"
                            if done_by_manual and reward_value:
                                termination_reason = "manual_success"
                            elif done_by_manual:
                                termination_reason = "manual_failure"
                            video_dir = _save_eval_video(env, ckpt_step, episode)
                            _record_eval_result(
                                eval_records,
                                ckpt_step=ckpt_step,
                                episode=episode,
                                reward=reward_value,
                                duration=dt,
                                episode_steps=episode_steps,
                                video_dir=video_dir,
                                termination_reason=termination_reason,
                            )
                            done_by_manual = False

                if ckpt_episode_counter:
                    print(f"checkpoint_{ckpt_step} success rate: {success_counter / ckpt_episode_counter}")
                else:
                    print(f"checkpoint_{ckpt_step} success rate: no completed eval episodes")
                print(f"checkpoint_{ckpt_step} average time: {np.mean(time_list) if time_list else float('nan')}")

            if total_episode_counter:
                print(f"success rate: {total_success_counter / total_episode_counter}")
            else:
                print("success rate: no completed eval episodes")
            print(f"average time: {np.mean(total_time_list) if total_time_list else float('nan')}")
            return  # after done eval, return and exit
        
        except KeyboardInterrupt:
            interrupted = True
            print_yellow("Eval interrupted by Ctrl+C.")
        finally:
            _write_eval_summary(eval_records, interrupted=interrupted)
            if key_reader is not None:
                key_reader.stop()
            print_green("Resetting env before eval exit.")
            try:
                env.reset()
            except Exception as exc:
                print_yellow(f"Failed to reset env before eval exit: {exc}")
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
    demo_count = 216

    key_reader = KeyReader()
    key_reader.start()
    pbar = tqdm.tqdm(range(start_step, config.max_steps), dynamic_ncols=True)
    try:
        for step in pbar:
            if is_multi_stage_task():
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
            if is_multi_stage_task() and mode == "S1_INFERENCE":
                pick_steps += 1
                obs, stage1_done = run_stage1_until_complete(agent_pick, env, obs, key)
                if stage1_done:
                    mode = "S2_TRAIN"
                    intervention_count = 0
                    intervention_steps = 0
                    already_intervened = False
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
                actions = sample_policy_action(agent, obs, key, argmax=True)
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

                manual_success_done = False
                manual_failure_done = False
                key_input = key_reader.get_key_nowait()
                while key_input is not None:
                    if key_input == "1":
                        done = True
                        reward = 1
                        manual_success_done = True
                        info = dict(info)
                        info["succeed"] = True
                        print_yellow("Manual episode success requested by 1.")
                    elif key_input == "2":
                        done = True
                        reward = 0
                        manual_failure_done = True
                        info = dict(info)
                        info["succeed"] = False
                        print_yellow("Manual episode failure requested by 2.")
                    key_input = key_reader.get_key_nowait()
                
                # if "is_pick" in info:
                #     is_pick = info["is_pick"]
                # else:
                #     is_pick = True
                state = obs["state"][0]
                if FLAGS.exp_name in {"twist_bottle_cap", "lid_grip"}:
                    # if state[2] < 0.22 and (0.6 < state[0] < 0.8) and (-0.13 < state[1] < -0.05):
                    #     actions[:3] = np.clip(actions[:3], -0.4, 0.4)
                    if state[2] < 0.24 and (0.6 < state[0] < 0.8) and (-0.2 < state[1] < -0.1):
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
                transition["grasp_penalty"] = info.get("grasp_penalty", np.float32(0.0))
                transition["robot_arm_penalty"] = info.get("robot_arm_penalty", np.float32(0.0))
                
                print("info['robot_arm_penalty'] = ", info.get('robot_arm_penalty', 0))
                print("info['grasp_penalty'] = ", info.get('grasp_penalty', 0))
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
                    info.setdefault("episode", {})
                    info["episode"]["intervention_count"] = intervention_count
                    info["episode"]["intervention_steps"] = intervention_steps
                    info["episode"]["manual_success"] = manual_success_done
                    info["episode"]["manual_failure"] = manual_failure_done
                    stats = {"environment": info}  # send stats to the learner to log
                    client.request("send-stats", stats)
                    pbar.set_description(f"last return: {running_return}")
                    running_return = 0.0
                    intervention_count = 0
                    intervention_steps = 0
                    already_intervened = False
                    client.update()
                    mode = "S1_INFERENCE"
                    demo_count += 1
                    obs = reset_after_episode(env)
                    
            timer.tock("total")
            if step % config.log_period == 0:
                stats = {"timer": timer.get_average_times()}
                client.request("send-stats", stats)
    finally:
        key_reader.stop()


##############################################################################

def learner(rng, agent, replay_buffer, demo_buffer, wandb_logger=None):
    """
    The learner loop, which runs when "--learner" is set to True.
    """
    latest_step = _latest_checkpoint_step(FLAGS.checkpoint_path)
    start_step = latest_step + 1 if latest_step > 0 else 0

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
    assert FLAGS.exp_name in NEW_MAPPING, "Experiment folder not found."
    config = NEW_MAPPING[FLAGS.exp_name]()

    assert config.batch_size % num_devices == 0
    # seed
    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, sampling_rng = jax.random.split(rng)

    env = config.get_environment(
        fake_env=FLAGS.learner,
        save_video=_is_eval_mode() and FLAGS.save_video,
        classifier=True,
        enable_tactile=FLAGS.enable_tactile
    )
    env = RecordEpisodeStatistics(env)

    agent: SACAgent = make_sac_pixel_agent_hybrid_single_arm(
        seed=FLAGS.seed,
        sample_obs=env.observation_space.sample(),
        sample_action=env.action_space.sample(),
        image_keys=config.image_keys,
        encoder_type=config.encoder_type,
        discount=config.discount,
        # state_weights=config.state_weights,
        # image_weights=config.image_weights,
    )
    include_robot_arm_penalty = True
    include_grasp_penalty = True

    # agent: SACAgent = make_sac_pixel_agent(
    #     seed=FLAGS.seed,
    #     sample_obs=env.observation_space.sample(),
    #     sample_action=env.action_space.sample(),
    #     image_keys=config.image_keys,
    #     encoder_type=config.encoder_type,
    #     discount=config.discount,
    #     # state_weights=config.state_weights,
    #     # image_weights=config.image_weights,
    # )
    
    
    # replicate agent across devices
    # need the jnp.array to avoid a bug where device_put doesn't recognize primitives
    agent = jax.device_put(
        jax.tree_util.tree_map(jnp.array, agent), sharding.replicate()
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

    agent_pick = None
    if FLAGS.exp_name in MULTI_STAGE_EXP_NAMES:
        agent_pick: SACAgent = make_sac_pixel_agent_hybrid_single_arm(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
            # state_weights=config.state_weights,
        )
        # replicate agent across devices
        # need the jnp.array to avoid a bug where device_put doesn't recognize primitives
    
        agent_pick = jax.device_put(
            jax.tree_util.tree_map(jnp.array, agent_pick), sharding.replicate()
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
            include_robot_arm_penalty=include_robot_arm_penalty,
        )
        # set up wandb and logging
        wandb_project = f"hil-ablation-{FLAGS.exp_name}-{_wandb_project_date()}-0"
        wandb_logger = make_wandb_logger(
            project=wandb_project,
            description=FLAGS.exp_name,
            debug=FLAGS.debug,
        )
        print_green(f"WandB project: {wandb_project}")
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
            include_robot_arm_penalty=include_robot_arm_penalty,
        )

        assert FLAGS.demo_path is not None
        demo_files = iter_transition_files(FLAGS.demo_path)
        assert demo_files, f"No demo pkl files found from --demo_path={FLAGS.demo_path}"
        for path in demo_files:
            transitions = load_transition_stream(path)
            print_green(f"Loaded {len(transitions)} demo transitions from {path}")
            for transition in transitions:
                demo_buffer.insert(ensure_penalty_keys(transition))
        print_green(f"demo buffer size: {len(demo_buffer)}")
        print_green(f"online buffer size: {len(replay_buffer)}")

        if FLAGS.checkpoint_path is not None and os.path.exists(
            os.path.join(FLAGS.checkpoint_path, "buffer")
        ):
            for file in iter_transition_files([os.path.join(FLAGS.checkpoint_path, "buffer")]):
                transitions = load_transition_stream(file)
                for transition in transitions:
                    replay_buffer.insert(ensure_penalty_keys(transition))
            print_green(
                f"Loaded previous buffer data. Replay buffer size: {len(replay_buffer)}"
            )

        if FLAGS.checkpoint_path is not None and os.path.exists(
            os.path.join(FLAGS.checkpoint_path, "demo_buffer")
        ):
            for file in iter_transition_files([os.path.join(FLAGS.checkpoint_path, "demo_buffer")]):
                transitions = load_transition_stream(file)
                for transition in transitions:
                    demo_buffer.insert(ensure_penalty_keys(transition))
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
