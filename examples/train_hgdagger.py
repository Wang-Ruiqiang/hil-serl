#!/usr/bin/env python3

import time
import jax
import jax.numpy as jnp
from natsort import natsorted
import numpy as np
import tqdm
from absl import app, flags
from flax.training import checkpoints
import os
import copy
import glob
import json
import pickle as pkl
import re
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from pynput import keyboard

from serl_launcher.agents.continuous.bc import BCAgent
from serl_launcher.utils.timer_utils import Timer

from agentlace.trainer import TrainerServer, TrainerClient
from agentlace.data.data_store import QueuedDataStore

from serl_launcher.utils.launcher import (
    make_bc_agent,
    make_trainer_config,
    make_wandb_logger,
)
from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore

from experiments.mappings import NEW_MAPPING
from examples.utils.runtime import MULTI_STAGE_EXP_NAMES, STOP_COMMAND_EXP_NAMES

FLAGS = flags.FLAGS

flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_boolean("learner", False, "Whether this is a learner.")
flags.DEFINE_boolean("actor", False, "Whether this is an actor.")
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
flags.DEFINE_multi_string("demo_path", None, "Path to the demo data.")
flags.DEFINE_string("demo_buffer_path", None, "Path to folder of demo buffers.")
flags.DEFINE_string("checkpoint_path", None, "Path to save checkpoints.")
flags.DEFINE_string("stage1_checkpoint_path", None, "Path to the stage-1 policy checkpoint.")
flags.DEFINE_integer(
    "stage1_checkpoint_step",
    -1,
    "Checkpoint step for the stage-1 policy. Use -1 for latest.",
)
flags.DEFINE_string(
    "init_bc_checkpoint_path",
    None,
    "Optional checkpoint directory for initializing HG-DAgger from an offline BC policy.",
)
flags.DEFINE_integer(
    "init_bc_checkpoint_step",
    -1,
    "Checkpoint step to restore from --init_bc_checkpoint_path. Use -1 for latest.",
)
flags.DEFINE_integer("eval_checkpoint_step", 0, "Step to evaluate the checkpoint.")
flags.DEFINE_integer("eval_n_trajs", 0, "Number of trajectories to evaluate.")
flags.DEFINE_integer(
    "eval_checkpoint_step_interval",
    0,
    "If > 0, load the next checkpoint after each eval episode by increasing eval_checkpoint_step by this interval.",
)
flags.DEFINE_integer(
    "eval_max_episode_steps",
    0,
    "Maximum number of policy steps per eval trajectory. Use 0 to disable.",
)
flags.DEFINE_bool("save_video", False, "Save videos for eval trajectories.")
flags.DEFINE_integer(
    "enable_tactile",
    -1,
    "Whether to include tactile observations. Use -1 to follow the task config default.",
)
flags.DEFINE_integer("pretrain_steps", 10000, "Number of pretraining steps.")
flags.DEFINE_integer(
    "max_episode_steps",
    150,
    "Maximum number of policy steps per task episode during actor training. Use 0 to disable.",
)
flags.DEFINE_float(
    "hand_action_weight",
    1.0,
    "BC/HG-DAgger loss weight for the last action dimension, used for hand progress.",
)

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


def _restore_checkpoint(agent, path, step, label):
    assert path is not None, f"{label} path is required."
    restore_step = None if step < 0 else step
    ckpt = checkpoints.restore_checkpoint(
        os.path.abspath(path),
        agent.state,
        step=restore_step,
    )
    print_green(
        f"Loaded {label}{' latest' if restore_step is None else f' at step {restore_step}'}: {path}"
    )
    return agent.replace(state=ckpt)


def _maybe_restore_init_bc(agent):
    if FLAGS.init_bc_checkpoint_path is None:
        return agent
    return _restore_checkpoint(
        agent,
        FLAGS.init_bc_checkpoint_path,
        FLAGS.init_bc_checkpoint_step,
        "initial offline BC checkpoint",
    )


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


def _is_multi_stage_task():
    return FLAGS.exp_name in MULTI_STAGE_EXP_NAMES


def _stage1_label():
    return {
        "tennis_ball_place": "pick",
        "twist_bottle_cap": "lid_grip",
    }.get(FLAGS.exp_name, "stage1")


def _stage2_label():
    return {
        "tennis_ball_place": "place",
        "twist_bottle_cap": "twist",
    }.get(FLAGS.exp_name, "stage2")


def _sample_bc_action(agent, obs, key, *, argmax):
    actions = agent.sample_actions(
        observations=jax.device_put(obs),
        argmax=argmax,
        seed=key,
    )
    actions = np.asarray(jax.device_get(actions)).copy()
    actions[..., 3:6] = 0.0
    return actions


def _hold_grasp_for_place_policy(actions):
    if FLAGS.exp_name != "tennis_ball_place":
        return actions
    actions = actions.copy()
    if actions[..., 6].item() < 0.0:
        print_yellow(
            f"[S2_PLACE] blocked policy open-hand action: {actions[..., 6].item():.4f} -> 0.0"
        )
        actions[..., 6] = 0.0
    return actions


def _run_stage1_until_complete(agent_stage1, env, obs, key):
    actions = _sample_bc_action(agent_stage1, obs, key, argmax=False)
    next_obs, reward, done, truncated, info = env.step(actions)
    executed_action = info.get("intervene_action", actions)
    label = _stage1_label()
    print_green(
        f"[S1_{label.upper()}] policy=stage1_{label}_policy "
        f"ckpt={FLAGS.stage1_checkpoint_path} step={FLAGS.stage1_checkpoint_step} "
        f"policy_action={np.round(actions, 4)} "
        f"executed_action={np.round(executed_action, 4)} "
        f"intervened={'intervene_action' in info} "
        f"reward={reward} stage1_active_after={info.get('is_pick', None)}"
    )
    if not info.get("is_pick", True):
        print_green(f"stage-1 {label} task done")
        return next_obs, True
    if done or truncated:
        print_yellow(f"stage-1 {label} ended before the {_stage2_label()} stage started")
    return next_obs, False


def _iter_pickle_files(paths):
    files = []
    for path in paths or []:
        path = os.path.expanduser(path)
        if os.path.isdir(path):
            path_files = glob.glob(os.path.join(path, "*.pkl"))
            if not path_files and os.path.isdir(os.path.join(path, "demo_buffer")):
                path_files = glob.glob(os.path.join(path, "demo_buffer", "*.pkl"))
            files.extend(path_files)
        else:
            matches = glob.glob(path)
            files.extend(matches if matches else [path])
    return natsorted(set(files))


def _load_transitions_into_buffer(paths, demo_buffer, label):
    loaded_files = 0
    loaded_transitions = 0
    for file in _iter_pickle_files(paths):
        with open(file, "rb") as f:
            transitions = pkl.load(f)
        for transition in transitions:
            demo_buffer.insert(transition)
        loaded_files += 1
        loaded_transitions += len(transitions)

    if loaded_files:
        print_green(
            f"Loaded {label}: files={loaded_files}, transitions={loaded_transitions}, "
            f"buffer_size={len(demo_buffer)}"
        )
    return loaded_transitions


def _reset_env_with_confirm(env):
    if FLAGS.exp_name == "tube_insertion":
        env.open_hand(steps=20, step_time=0.05)
        time.sleep(1.5)
    input("reset env")
    return env.reset()[0]


def _reached_max_episode_steps(episode_steps):
    return FLAGS.max_episode_steps > 0 and episode_steps >= FLAGS.max_episode_steps


def _reached_eval_max_episode_steps(eval_episode_steps):
    return (
        FLAGS.eval_max_episode_steps > 0
        and eval_episode_steps >= FLAGS.eval_max_episode_steps
    )


def _save_eval_video(env, ckpt_step, episode):
    if not FLAGS.save_video:
        return None
    video_id = f"ckpt_{ckpt_step}_episode_{episode}"
    env.unwrapped.save_video_recording(video_id)
    return os.path.abspath(os.path.join("videos", video_id))


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
        "stage1_checkpoint_path": FLAGS.stage1_checkpoint_path,
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


def _actor_stats_path():
    return os.path.join(FLAGS.checkpoint_path, "hgdagger_actor_stats.json")


def _load_actor_stats():
    default_stats = {
        "exp_name": FLAGS.exp_name,
        "checkpoint_path": FLAGS.checkpoint_path,
        "episodes": 0,
        "successes": 0,
        "intervention_segments": 0,
        "intervention_steps": 0,
        "manual_successes": 0,
        "manual_failures": 0,
        "max_length_terminations": 0,
        "last_episode": None,
        "episodes_detail": [],
        "updated_at": None,
    }
    path = _actor_stats_path()
    if path and os.path.exists(path):
        with open(path, "r") as f:
            loaded = json.load(f)
        default_stats.update(loaded)
    return default_stats


def _write_actor_stats(stats):
    os.makedirs(FLAGS.checkpoint_path, exist_ok=True)
    stats["updated_at"] = time.strftime("%Y-%m-%d_%H-%M-%S")
    with open(_actor_stats_path(), "w") as f:
        json.dump(stats, f, indent=2)


def _update_actor_stats(
    stats,
    *,
    step,
    reward,
    episode_steps,
    intervention_count,
    intervention_steps,
    manual_success,
    manual_failure,
    max_length_terminated,
):
    reward_value = float(np.asarray(reward).item())
    episode_record = {
        "episode": int(stats["episodes"]),
        "step": int(step),
        "reward": reward_value,
        "success": bool(reward_value),
        "episode_steps": int(episode_steps),
        "intervention_segments": int(intervention_count),
        "intervention_steps": int(intervention_steps),
        "manual_success": bool(manual_success),
        "manual_failure": bool(manual_failure),
        "max_length_terminated": bool(max_length_terminated),
    }
    stats["episodes"] += 1
    stats["successes"] += int(bool(reward_value))
    stats["intervention_segments"] += int(intervention_count)
    stats["intervention_steps"] += int(intervention_steps)
    stats["manual_successes"] += int(bool(manual_success))
    stats["manual_failures"] += int(bool(manual_failure))
    stats["max_length_terminations"] += int(bool(max_length_terminated))
    stats["last_episode"] = episode_record
    stats["episodes_detail"].append(episode_record)
    _write_actor_stats(stats)
    success_rate = stats["successes"] / stats["episodes"] if stats["episodes"] else 0.0
    print_green(
        "[HG-DAgger actor stats] "
        f"episodes={stats['episodes']} successes={stats['successes']} "
        f"success_rate={success_rate:.3f} "
        f"intervention_segments={stats['intervention_segments']} "
        f"intervention_steps={stats['intervention_steps']} "
        f"last_steps={episode_steps} "
        f"last_segments={intervention_count} "
        f"last_intervention_steps={intervention_steps}"
    )


should_reset = False
should_stop_eval = False
should_succeed = False

def on_press(key):
    global should_reset, should_stop_eval, should_succeed
    if key == keyboard.Key.esc:
        should_reset = True
        print("ESC pressed. Ending current task...")
        return
    key_char = getattr(key, "char", None)
    if key_char == "1":
        should_succeed = True
        print("1 pressed. Marking current task as success...")
    elif key_char == "2":
        should_reset = True
        print("2 pressed. Ending current task as failure...")

# Start the keyboard listener in a non-blocking way


##############################################################################


def actor(agent: BCAgent, data_store, env, sampling_rng, agent_pick=None):
    """
    This is the actor loop, which runs when "--actor" is set to True.
    """
    global should_reset, should_stop_eval, should_succeed
    if _is_multi_stage_task():
        assert agent_pick is not None, "stage-1 policy is required for multi-stage actor/eval."
        print_green(
            "[multi-stage] "
            f"S1 {_stage1_label()} policy={FLAGS.stage1_checkpoint_path}, "
            f"step={FLAGS.stage1_checkpoint_step}; "
            f"S2 {_stage2_label()} policy={FLAGS.checkpoint_path}"
        )

    if FLAGS.eval_checkpoint_step:
        listener = keyboard.Listener(on_press=on_press)
        listener.start()
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

        total_success_counter = 0
        total_episode_counter = 0
        total_time_list = []
        eval_records = []
        interrupted = False

        try:
            for ckpt_index, ckpt_step in enumerate(eval_steps):
                if should_stop_eval:
                    break
                success_counter = 0.0
                episode_counter = 0
                time_list = []

                try:
                    agent = _restore_checkpoint(
                        agent,
                        FLAGS.checkpoint_path,
                        ckpt_step,
                        "eval policy checkpoint",
                    )
                except ValueError as exc:
                    print_yellow(f"Skipping checkpoint_{ckpt_step}: {exc}")
                    break

                print_green(f"Evaluating checkpoint step {ckpt_step}")
                n_episodes = 1 if FLAGS.eval_checkpoint_step_interval > 0 else FLAGS.eval_n_trajs
                for episode in range(n_episodes):
                    if should_stop_eval:
                        break
                    needs_confirmed_reset = episode > 0 or ckpt_index > 0
                    obs = _reset_env_with_confirm(env) if needs_confirmed_reset else env.reset()[0]
                    done = False
                    mode = "S1_INFERENCE" if _is_multi_stage_task() else "S2_TRAIN"
                    eval_episode_steps = 0
                    start_time = time.time()
                    while not done and not should_stop_eval:
                        sampling_rng, key = jax.random.split(sampling_rng)
                        if mode == "S1_INFERENCE":
                            obs, stage1_done = _run_stage1_until_complete(agent_pick, env, obs, key)
                            eval_episode_steps += 1
                            if should_succeed:
                                should_succeed = False
                                stage1_done = True
                                print_yellow(f"Manual stage-1 {_stage1_label()} success requested by 1.")
                            if should_reset:
                                should_reset = False
                                episode_counter += 1
                                total_episode_counter += 1
                                print_yellow("Manual episode failure requested.")
                                print(0)
                                print(f"checkpoint_{ckpt_step}: {success_counter}/{episode_counter}")
                                video_dir = _save_eval_video(env, ckpt_step, episode)
                                _record_eval_result(
                                    eval_records,
                                    ckpt_step=ckpt_step,
                                    episode=episode,
                                    reward=0,
                                    duration=time.time() - start_time,
                                    episode_steps=eval_episode_steps,
                                    video_dir=video_dir,
                                    termination_reason="manual",
                                )
                                break
                            if _reached_eval_max_episode_steps(eval_episode_steps):
                                episode_counter += 1
                                total_episode_counter += 1
                                print_yellow(
                                    f"Reached eval max episode length {FLAGS.eval_max_episode_steps}."
                                )
                                print(0)
                                print(f"checkpoint_{ckpt_step}: {success_counter}/{episode_counter}")
                                video_dir = _save_eval_video(env, ckpt_step, episode)
                                _record_eval_result(
                                    eval_records,
                                    ckpt_step=ckpt_step,
                                    episode=episode,
                                    reward=0,
                                    duration=time.time() - start_time,
                                    episode_steps=eval_episode_steps,
                                    video_dir=video_dir,
                                    termination_reason="max_length",
                                )
                                break
                            if stage1_done:
                                mode = "S2_TRAIN"
                            continue

                        actions = _sample_bc_action(agent, obs, key, argmax=False)
                        # actions = _hold_grasp_for_place_policy(actions)

                        next_obs, reward, done, truncated, info = env.step(actions)
                        eval_episode_steps += 1
                        obs = next_obs

                        manual_success_done = should_succeed
                        if manual_success_done:
                            should_succeed = False
                            done = True
                            reward = 1
                            print_yellow("Manual episode success requested by 1.")

                        max_length_done = _reached_eval_max_episode_steps(eval_episode_steps)
                        if max_length_done and not done:
                            done = True
                            reward = 0
                            print_yellow(
                                f"Reached eval max episode length {FLAGS.eval_max_episode_steps}."
                            )

                        if done or truncated or should_reset or should_stop_eval:
                            termination_reason = "success" if reward else "env_done"
                            if truncated:
                                termination_reason = "truncated"
                            if max_length_done:
                                termination_reason = "max_length"
                            if manual_success_done:
                                termination_reason = "manual_success"
                            if should_reset:
                                reward = 0
                                termination_reason = "manual_failure"
                                print_yellow("Manual episode failure requested.")
                            should_reset = False
                            episode_counter += 1
                            total_episode_counter += 1
                            reward_value = float(np.asarray(reward).item())
                            duration = time.time() - start_time
                            if reward_value:
                                time_list.append(duration)
                                total_time_list.append(duration)
                                print(duration)

                            success_counter += reward_value
                            total_success_counter += reward_value
                            print(reward_value)
                            print(f"checkpoint_{ckpt_step}: {success_counter}/{episode_counter}")
                            video_dir = _save_eval_video(env, ckpt_step, episode)
                            _record_eval_result(
                                eval_records,
                                ckpt_step=ckpt_step,
                                episode=episode,
                                reward=reward_value,
                                duration=duration,
                                episode_steps=eval_episode_steps,
                                video_dir=video_dir,
                                termination_reason=termination_reason,
                            )
                            break

                if episode_counter:
                    print(f"checkpoint_{ckpt_step} success rate: {success_counter / episode_counter}")
                else:
                    print(f"checkpoint_{ckpt_step} success rate: no completed eval episodes")
                print(
                    f"checkpoint_{ckpt_step} average time: "
                    f"{np.mean(time_list) if time_list else float('nan')}"
                )

            if FLAGS.eval_checkpoint_step_interval > 0:
                if total_episode_counter:
                    print(f"all checkpoints success rate: {total_success_counter / total_episode_counter}")
                else:
                    print("all checkpoints success rate: no completed eval episodes")
                print(
                    f"all checkpoints average time: "
                    f"{np.mean(total_time_list) if total_time_list else float('nan')}"
                )
        except KeyboardInterrupt:
            interrupted = True
            print_yellow("Eval interrupted by Ctrl+C.")
        finally:
            _write_eval_summary(eval_records, interrupted=interrupted)
            listener.stop()
            print_green("Resetting env before eval exit.")
            try:
                env.reset()
            except Exception as exc:
                print_yellow(f"Failed to reset env before eval exit: {exc}")
        return  # after done eval, return and exit

    existing_buffer_files = (
        natsorted(glob.glob(os.path.join(FLAGS.checkpoint_path, "buffer", "transitions_*.pkl")))
        if FLAGS.checkpoint_path
        else []
    )
    # Backward compatibility for runs created before the full buffer was saved.
    existing_step_files = existing_buffer_files or (
        natsorted(
            glob.glob(
                os.path.join(FLAGS.checkpoint_path, "demo_buffer", "transitions_*.pkl")
            )
        )
        if FLAGS.checkpoint_path
        else []
    )
    start_step = (
        int(os.path.basename(existing_step_files[-1])[12:-4]) + 1
        if existing_step_files
        else 0
    )


    client = TrainerClient(
        "actor_env",
        FLAGS.ip,
        make_trainer_config(),
        data_store,
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
    actor_stats = _load_actor_stats()
    _write_actor_stats(actor_stats)
    print_green(f"HG-DAgger actor stats will be saved to {_actor_stats_path()}")

    obs, _ = env.reset()
    done = False

    # training loop
    timer = Timer()
    running_return = 0.0
    already_intervened = False
    intervention_count = 0
    intervention_steps = 0
    pick_steps = 0
    episode_steps = 0
    mode = "S1_INFERENCE" if _is_multi_stage_task() else "S2_TRAIN"

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    pbar = tqdm.tqdm(range(start_step, config.max_steps), dynamic_ncols=True)
    try:
        for raw_step in pbar:
            step = raw_step - pick_steps if _is_multi_stage_task() else raw_step
            if step < start_step:
                step = start_step

            sampling_rng, key = jax.random.split(sampling_rng)
            if mode == "S1_INFERENCE":
                pick_steps += 1
                obs, stage1_done = _run_stage1_until_complete(agent_pick, env, obs, key)
                if should_succeed:
                    should_succeed = False
                    stage1_done = True
                    print_yellow(f"Manual stage-1 {_stage1_label()} success requested by 1.")
                if should_reset:
                    should_reset = False
                    running_return = 0.0
                    intervention_count = 0
                    intervention_steps = 0
                    already_intervened = False
                    episode_steps = 0
                    mode = "S1_INFERENCE" if _is_multi_stage_task() else "S2_TRAIN"
                    obs = _reset_env_with_confirm(env)
                elif stage1_done:
                    mode = "S2_TRAIN"
                    running_return = 0.0
                    intervention_count = 0
                    intervention_steps = 0
                    already_intervened = False
                    episode_steps = 0
                continue

            timer.tick("total")

            with timer.context("sample_actions"):
                actions = _sample_bc_action(agent, obs, key, argmax=False)
                # actions = _hold_grasp_for_place_policy(actions)

            # Step environment
            with timer.context("step_env"):

                next_obs, reward, done, truncated, info = env.step(actions)
                episode_steps += 1
                manual_success_done = should_succeed
                if manual_success_done:
                    should_succeed = False
                    done = True
                    reward = 1
                    info = dict(info)
                    info["succeed"] = True
                    print_yellow("Manual episode success requested by 1.")
                max_length_done = False
                if not done and _reached_max_episode_steps(episode_steps):
                    truncated = True
                    done = True
                    max_length_done = True
                    reward = 0
                    print_yellow(
                        f"reached max episode length {FLAGS.max_episode_steps}; ending episode."
                    )
                manual_failure_done = should_reset
                if manual_failure_done:
                    done = True
                    reward = 0
                    info = dict(info)
                    info["succeed"] = False
                    print_yellow("Manual episode failure requested.")
                if "left" in info:
                    info.pop('left')
                if "right" in info:
                    info.pop('right')

                # override the action with the intervention action
                if "intervene_action" in info:
                    actions = info.pop("intervene_action")
                    intervention_steps += 1
                    if not already_intervened:
                        intervention_count += 1
                    already_intervened = True
                else:
                    already_intervened = False

                reward = np.asarray(reward, dtype=np.float32)
                running_return += reward
                transition = dict(
                    observations=obs,
                    actions=actions,
                    next_observations=next_obs,
                    rewards=reward,
                    masks=1.0 - done,
                    dones=done,
                )
                transitions.append(copy.deepcopy(transition))
                if already_intervened and not manual_failure_done and not max_length_done:
                    data_store.insert(transition)
                    demo_transitions.append(copy.deepcopy(transition))

                obs = next_obs
                if done or truncated:
                    _update_actor_stats(
                        actor_stats,
                        step=step,
                        reward=reward,
                        episode_steps=episode_steps,
                        intervention_count=intervention_count,
                        intervention_steps=intervention_steps,
                        manual_success=manual_success_done,
                        manual_failure=manual_failure_done,
                        max_length_terminated=max_length_done,
                    )
                    episode_info = info.setdefault('episode', {})
                    episode_info['intervention_count'] = intervention_count
                    episode_info['intervention_steps'] = intervention_steps
                    episode_info['episode_steps'] = episode_steps
                    episode_info['manual_success'] = manual_success_done
                    episode_info['manual_failure'] = manual_failure_done
                    episode_info['max_length_terminated'] = max_length_done
                    stats = {"environment": info}  # send stats to the learner to log
                    client.request("send-stats", stats)
                    pbar.set_description(f"last return: {running_return}")
                    running_return = 0.0
                    intervention_count = 0
                    intervention_steps = 0
                    already_intervened = False
                    episode_steps = 0
                    should_reset = False
                    mode = "S1_INFERENCE" if _is_multi_stage_task() else "S2_TRAIN"
                    client.update()
                    obs = _reset_env_with_confirm(env)

            if step > 0 and config.buffer_period > 0 and step % config.buffer_period == 0:
                # dump to pickle file
                buffer_path = os.path.join(FLAGS.checkpoint_path, "buffer")
                demo_buffer_path = os.path.join(FLAGS.checkpoint_path, "demo_buffer")
                os.makedirs(buffer_path, exist_ok=True)
                os.makedirs(demo_buffer_path, exist_ok=True)
                with open(os.path.join(buffer_path, f"transitions_{step}.pkl"), "wb") as f:
                    pkl.dump(transitions, f)
                    transitions = []
                with open(os.path.join(demo_buffer_path, f"transitions_{step}.pkl"), "wb") as f:
                    pkl.dump(demo_transitions, f)
                    demo_transitions = []

            timer.tock("total")

            if step % config.log_period == 0:
                stats = {"timer": timer.get_average_times()}
                client.request("send-stats", stats)
    finally:
        listener.stop()


##############################################################################


def learner(rng, agent: BCAgent, demo_buffer, wandb_logger=None):
    """
    The learner loop, which runs when "--learner" is set to True.
    """
    current_step = 0

    def stats_callback(type: str, payload: dict) -> dict:
        """Callback for when server receives stats request."""
        assert type == "send-stats", f"Invalid request type: {type}"
        if wandb_logger is not None:
            wandb_logger.log(payload, step=current_step)
        return {}  # not expecting a response

    # Create server
    server = TrainerServer(make_trainer_config(), request_callback=stats_callback)
    server.register_data_store("actor_env", demo_buffer)
    server.start(threaded=True)

    update_step = _latest_checkpoint_step(FLAGS.checkpoint_path)
    if update_step > 0:
        agent = _restore_checkpoint(
            agent,
            FLAGS.checkpoint_path,
            update_step,
            "HG-DAgger resume checkpoint",
        )
        current_step = update_step
        print_green(f"Resuming HG-DAgger learner from step {update_step + 1}.")

    # send the initial network to the actor
    server.publish_network(agent.state.params)
    print_green("sent initial network to actor")

    demo_iterator = demo_buffer.get_iterator(
        sample_args={
            "batch_size": config.batch_size,
            "pack_obs_and_next_obs": True,
        },
        device=sharding.replicate(),
    )

    # Pretrain BC policy to get started, unless we explicitly initialized from
    # an offline BC checkpoint.
    if update_step > 0:
        pass
    elif FLAGS.init_bc_checkpoint_path is not None:
        if FLAGS.pretrain_steps:
            print_yellow(
                "Skipping HG-DAgger pretrain_steps because --init_bc_checkpoint_path was provided."
            )
    elif FLAGS.pretrain_steps:
        if os.path.isdir(
            os.path.join(
                FLAGS.checkpoint_path, f"checkpoint_{FLAGS.pretrain_steps}"
            )
        ):
            print_green(
                f"BC checkpoint at {FLAGS.pretrain_steps} steps found, restoring BC checkpoint"
            )
            ckpt = checkpoints.restore_checkpoint(
                os.path.abspath(FLAGS.checkpoint_path), agent.state, step=FLAGS.pretrain_steps
            )
            agent = agent.replace(state=ckpt)
            update_step = FLAGS.pretrain_steps
        else:
            update_step = 0
            print_yellow(
                f"No BC checkpoint at {FLAGS.pretrain_steps} steps found, starting from scratch"
            )
            for step in tqdm.tqdm(
                range(FLAGS.pretrain_steps),
                dynamic_ncols=True,
                desc="bc_pretraining",
            ):
                update_step += 1
                current_step = update_step
                batch = next(demo_iterator)
                agent, bc_update_info = agent.update(batch)
                if update_step % config.log_period == 0 and wandb_logger:
                    wandb_logger.log({"bc": bc_update_info}, step=update_step)
            checkpoints.save_checkpoint(
                os.path.abspath(FLAGS.checkpoint_path), agent.state, step=update_step, keep=20
            )
            print_green("bc pretraining done and saved checkpoint")

    agent = jax.block_until_ready(agent)
    server.publish_network(agent.state.params)

    # wait till the replay buffer is filled with enough data
    timer = Timer()
    for step in tqdm.tqdm(range(update_step + 1, config.max_steps), dynamic_ncols=True, desc="learner"):
        current_step = step

        with timer.context("train"):
            batch = next(demo_iterator)
            agent, update_info = agent.update(
                batch,
            )
  
        # publish the updated network
        if step > 0 and step % (config.steps_per_update) == 0:
            agent = jax.block_until_ready(agent)
            server.publish_network(agent.state.params)

        if step % config.log_period == 0 and wandb_logger:
            wandb_logger.log(update_info, step=step)
            wandb_logger.log({"timer": timer.get_average_times()}, step=step)

        if step > 0 and config.checkpoint_period and step % config.checkpoint_period == 0:
            checkpoints.save_checkpoint(
                os.path.abspath(FLAGS.checkpoint_path), agent.state, step=step, keep=100
            )



##############################################################################


def main(_):
    global config
    assert FLAGS.exp_name in NEW_MAPPING, 'Experiment folder not found.'
    config = NEW_MAPPING[FLAGS.exp_name]()

    assert config.batch_size % num_devices == 0
    # seed
    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, sampling_rng = jax.random.split(rng)

    env = config.get_environment(
        fake_env=FLAGS.learner,
        save_video=FLAGS.save_video and FLAGS.actor and bool(FLAGS.eval_checkpoint_step),
        classifier=FLAGS.actor,
        enable_tactile=None if FLAGS.enable_tactile < 0 else bool(FLAGS.enable_tactile),
    )
    env = RecordEpisodeStatistics(env)

    rng, sampling_rng = jax.random.split(rng)
    agent: BCAgent = make_bc_agent(
        seed=FLAGS.seed,
        sample_obs=env.observation_space.sample(),
        sample_action=env.action_space.sample(),
        image_keys=config.image_keys,
        encoder_type=config.encoder_type,
        hand_action_weight=FLAGS.hand_action_weight,
    )

    # replicate agent across devices
    # need the jnp.array to avoid a bug where device_put doesn't recognize primitives
    agent: BCAgent = jax.device_put(
        jax.tree_util.tree_map(jnp.array, agent), sharding.replicate()
    )
    agent = _maybe_restore_init_bc(agent)
    agent_pick = None
    if FLAGS.actor and _is_multi_stage_task():
        assert (
            FLAGS.stage1_checkpoint_path is not None
        ), "--stage1_checkpoint_path is required when actor/eval runs a multi-stage task."
        agent_pick = make_bc_agent(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            hand_action_weight=FLAGS.hand_action_weight,
        )
        agent_pick = jax.device_put(
            jax.tree_util.tree_map(jnp.array, agent_pick), sharding.replicate()
        )
        agent_pick = _restore_checkpoint(
            agent_pick,
            FLAGS.stage1_checkpoint_path,
            FLAGS.stage1_checkpoint_step,
            "stage-1 policy checkpoint",
        )

    if FLAGS.learner:
        sampling_rng = jax.device_put(sampling_rng, device=sharding.replicate())
        wandb_date = _wandb_project_date()
        wandb_project = f"hgdagger-{FLAGS.exp_name}-{wandb_date}-1"
        wandb_logger = make_wandb_logger(
            project=wandb_project,
            description=FLAGS.exp_name,
            debug=FLAGS.debug,
        )
        print_green(f"WandB project: {wandb_project}")
        demo_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=50000,
            image_keys=config.image_keys,
        )

        previous_demo_buffer_paths = []
        if FLAGS.demo_buffer_path is not None:
            previous_demo_buffer_paths.append(FLAGS.demo_buffer_path)

        checkpoint_demo_buffer_path = (
            os.path.join(FLAGS.checkpoint_path, "demo_buffer")
            if FLAGS.checkpoint_path is not None
            else None
        )
        if (
            checkpoint_demo_buffer_path is not None
            and os.path.isdir(checkpoint_demo_buffer_path)
            and checkpoint_demo_buffer_path not in previous_demo_buffer_paths
        ):
            previous_demo_buffer_paths.append(checkpoint_demo_buffer_path)

        assert FLAGS.demo_path is not None or previous_demo_buffer_paths

        _load_transitions_into_buffer(
            previous_demo_buffer_paths,
            demo_buffer,
            "previous HG-DAgger intervention demo buffer",
        )
        _load_transitions_into_buffer(
            FLAGS.demo_path,
            demo_buffer,
            "initial offline demo data",
        )
        print(f"demo buffer size: {len(demo_buffer)}")
        
        # learner loop
        print_green("starting learner loop")
        learner(
            sampling_rng,
            agent,
            demo_buffer=demo_buffer,
            wandb_logger=wandb_logger,
        )

    elif FLAGS.actor:
        sampling_rng = jax.device_put(sampling_rng, sharding.replicate())
        data_store = QueuedDataStore(50000)  # the queue size on the actor
        
        actor(
            agent, 
            data_store,
            env, 
            sampling_rng,
            agent_pick=agent_pick,
            )

    else:
        raise NotImplementedError("Must be either a learner or an actor")


if __name__ == "__main__":
    app.run(main)
