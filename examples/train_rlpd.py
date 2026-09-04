#!/usr/bin/env python3

import os, sys, threading, queue, termios, tty, select
import inspect
import glob
import time
import cv2
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from absl import app, flags
from flax.training import checkpoints
import copy
import pickle as pkl
from pathlib import Path
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from natsort import natsorted

from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.utils.timer_utils import Timer
from serl_launcher.utils.train_utils import concat_batches
from serl_launcher.utils.gaze_utils import ensure_optional_transition_fields

from agentlace.trainer import TrainerServer, TrainerClient
from agentlace.data.data_store import QueuedDataStore

from serl_launcher.utils.launcher import (
    make_gaze_sac_pixel_agent_hybrid_single_arm,
    make_trainer_config,
    make_wandb_logger,
)
from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore
from serl_launcher.utils.gaze_mask_utils import PHASE_ONEHOT_DIM
from experiments.mappings import NEW_MAPPING

FLAGS = flags.FLAGS
REPO_ROOT = Path(__file__).resolve().parents[1]

SUPPORTED_EXPS = {"tennis_ball_pick", "tennis_ball_pick_and_place"}

flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_boolean("learner", False, "Whether this is a learner.")
flags.DEFINE_boolean("actor", False, "Whether this is an actor.")
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
flags.DEFINE_multi_string("demo_path", None, "Path to the demo data.")
flags.DEFINE_string("checkpoint_path", None, "Path to save checkpoints.")
flags.DEFINE_integer("eval_checkpoint_step", 0, "Step to evaluate the checkpoint.")
flags.DEFINE_integer("eval_n_trajs", 21, "Number of trajectories to evaluate.")
flags.DEFINE_string(
    "eval_encoder_type",
    "",
    "Encoder type used to construct an eval-only checkpoint. Empty uses the task config.",
)
flags.DEFINE_string(
    "eval_encoder_checkpoint_path",
    "",
    "Pretrained encoder checkpoint used to construct a vit-grounded eval agent.",
)
flags.DEFINE_string(
    "encoder_type",
    "",
    "Override config.encoder_type (e.g. vit-grounded). Empty keeps the config "
    "value, so the resnet-pretrained baseline scripts are unaffected.",
)
flags.DEFINE_string(
    "encoder_checkpoint_path",
    "",
    "Pretrained encoder checkpoint for vit-grounded. Empty trains from scratch.",
)
flags.DEFINE_boolean("save_video", False, "Save video.")
flags.DEFINE_integer("enable_tactile", 1, "evaluate pick or place task.")
flags.DEFINE_string(
    "gaze_predictor_checkpoint_path",
    str(REPO_ROOT / "examples" / "gaze_data_process" / "gaze_heatmap_ckpt"),
    "Checkpoint directory for the frozen gaze heatmap predictor used by the actor.",
)
flags.DEFINE_string(
    "mask_predictor_checkpoint_path",
    str(
        REPO_ROOT
        / "examples"
        / "gaze_data_process"
        / "SAM_process"
        / "mask_predictor_ckpt"
        / "best.pt"
    ),
    "Checkpoint file for the frozen RGB mask predictor used by front_camera_mask.",
)
flags.DEFINE_enum(
    "mask_selection_mode",
    "pick_classifier",
    ["gaze", "pick_classifier", "pick_only", "place_only"],
    "How front_camera_mask chooses mask1/mask2.",
)
flags.DEFINE_string(
    "pick_classifier_checkpoint_path",
    "examples/reward_classifier/classifier_ckpt_ball_pick",
    "Checkpoint directory for pick classifier mask selection.",
)
flags.DEFINE_float(
    "pick_classifier_threshold",
    0.8,
    "Pick classifier probability threshold. Below uses mask1; above uses mask2.",
)
flags.DEFINE_integer(
    "gaze_target_mask_dilation",
    2,
    "Dilation radius in mask-predictor pixels when checking whether gaze hits a mask.",
)
flags.DEFINE_float(
    "discount",
    -1.0,
    "Overrides the experiment config's discount when >= 0. Per-run rather than "
    "in TrainConfig on purpose: the phase and replicate modes exist to be "
    "comparable to runs already on disk, and a shared value would silently "
    "retune them too.",
)
flags.DEFINE_boolean(
    "debug", False, "Debug mode."
)  # debug mode will disable wandb logging
flags.DEFINE_boolean(
    "actor_feature_overlay",
    False,
    "Display current critic feature/attention heatmap over front_camera in actor/eval.",
)
flags.DEFINE_integer(
    "actor_feature_overlay_period",
    1,
    "Display feature overlay every N actor env steps. Set 0 to disable.",
)
flags.DEFINE_float(
    "actor_feature_overlay_alpha",
    0.45,
    "Heatmap opacity for actor feature overlay.",
)
flags.DEFINE_boolean(
    "eval_record",
    True,
    "In eval mode, write per-episode raw/attention video plus an .npz of the "
    "per-step diagnostics. Costs two extra network forwards per control step "
    "(critic and grasp critic) to log Q values.",
)
flags.DEFINE_string(
    "eval_record_dir",
    None,
    "Where eval recordings go. Defaults to <checkpoint_path>/eval_recordings.",
)
flags.DEFINE_float(
    "actor_feature_overlay_gamma",
    0.35,
    "Display-only dynamic-range compression for the softmax overlay "
    "(vit-grounded). 1.0 renders per-cell probability faithfully, which hides "
    "any secondary object: the ball covers ~4 tokens and the basket ~40, so "
    "during pick the peak sits ~7x higher and the hand -- holding 23% of the "
    "attention mass -- renders at 5% of it, i.e. solid blue. Values below 1 "
    "lift the low end so the whole support is visible. Ignored in energy mode "
    "so the resnet-pretrained baseline renders exactly as before.",
)


devices = jax.local_devices()
num_devices = len(devices)
sharding = jax.sharding.PositionalSharding(devices)

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

    def drain(self):
        while self.get_key_nowait() is not None:
            pass

    def stop(self):
        self._stop.set()


def zero_action_rpy(action):
    """Pin the rotation slots of a 7-dim action to zero.

    A no-op once ArmActionSubspaceWrapper has narrowed the action space, where
    index 3 is the gripper rather than a rotation -- zeroing it there would
    silently destroy every grasp command.
    """
    action = np.asarray(action).copy()
    if action.shape[-1] >= 7:
        action[..., 3:6] = 0.0
    return action


def _last_rgb_frame(image):
    image = np.asarray(image)
    if image.ndim == 4:
        image = image[-1]
    if image.ndim != 3 or image.shape[-1] < 3:
        raise ValueError(f"Expected RGB image with shape HxWxC, got {image.shape}")
    return image[..., :3].astype(np.uint8)


def _latest_unstacked_observation(observation, observation_space):
    """Extract one raw observation from either raw or horizon-stacked data."""
    latest = {}
    for key, value in observation.items():
        array = np.asarray(value)
        if key not in observation_space.spaces:
            latest[key] = array[-1] if array.ndim > 0 and array.shape[0] == 1 else array
            continue
        stacked_shape = observation_space.spaces[key].shape
        raw_shape = stacked_shape[1:]
        if array.shape == raw_shape:
            latest[key] = array
        elif array.ndim == len(stacked_shape) and array.shape[1:] == raw_shape:
            latest[key] = array[-1]
        else:
            raise ValueError(
                f"Cannot adapt demo observation {key!r} with shape {array.shape} "
                f"to replay shape {stacked_shape}."
            )
    return latest


def _stack_demo_transition_history(transitions, observation_space, horizon):
    """Yield temporal observation windows without rewriting demo pickle files."""
    if horizon <= 1:
        yield from transitions
        return

    history = None
    previous_done = True
    previous_episode = None
    for transition in transitions:
        transition = dict(transition)
        info = transition.get("infos", {})
        episode = (info.get("source_root"), info.get("episode_id"))
        current = _latest_unstacked_observation(
            transition["observations"], observation_space
        )
        next_observation = _latest_unstacked_observation(
            transition["next_observations"], observation_space
        )
        starts_episode = (
            history is None
            or previous_done
            or (
                previous_episode != (None, None)
                and episode != (None, None)
                and episode != previous_episode
            )
        )
        if starts_episode:
            history = [current for _ in range(horizon)]
        else:
            history = history[1:] + [current]

        next_history = history[1:] + [next_observation]
        transition["observations"] = {
            key: np.stack([item[key] for item in history], axis=0)
            for key in current
        }
        transition["next_observations"] = {
            key: np.stack([item[key] for item in next_history], axis=0)
            for key in next_observation
        }
        previous_done = bool(transition.get("dones", False))
        previous_episode = episode
        yield transition


def _attention_heatmap_overlay(rgb_image, attention_map, alpha, mode="energy", gamma=1.0):
    """Overlay an encoder attention map on the RGB frame.

    ``mode`` must match what the encoder actually returns:

    "softmax"
        The map is raw attention *logits* (vit-grounded's grounding query).
        These have to go through the same softmax the CGL loss uses before
        they mean anything. Rescaling the logits linearly instead -- which is
        what this function used to do for every encoder -- exaggerates
        secondary regions badly: a cell one nat below the peak holds ~37% of
        the peak's probability but renders at ~90% of its brightness once the
        map is divided by its own max.

    "energy"
        The map is a non-negative feature-energy map (resnet-pretrained's
        ``mean(square(features))``). Softmax is meaningless on an unbounded
        magnitude, so keep the original robust-max rescaling.

    Either way the result is normalized to its own peak, so brightness is
    always *relative to the strongest cell in this frame*, never absolute.
    That relative scale is what makes a widely-spread secondary object
    disappear next to a tightly-peaked primary one, so ``gamma`` < 1 may be
    applied afterwards (softmax mode only) to compress the dynamic range for
    display. It changes nothing the policy sees.
    """
    attention = np.asarray(attention_map, dtype=np.float32)
    if attention.ndim > 2:
        attention = attention.reshape((-1, *attention.shape[-2:]))[0]
    attention = np.nan_to_num(attention, nan=0.0, posinf=0.0, neginf=0.0)

    peak_probability = None
    if mode == "softmax":
        flat = attention.reshape(-1)
        probabilities = np.exp(flat - flat.max())
        probabilities /= max(probabilities.sum(), 1e-8)
        attention = probabilities.reshape(attention.shape)
        peak_probability = float(attention.max())
        attention = attention / (peak_probability + 1e-12)
    else:
        attention = np.maximum(attention, 0.0)
        robust_max = float(np.percentile(attention, 99.0))
        attention = attention / (robust_max + 1e-6)
    attention = np.clip(attention, 0.0, 1.0)
    if mode == "softmax" and gamma != 1.0:
        attention = attention ** float(gamma)
    attention_uint8 = np.clip(attention * 255.0, 0, 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(
        cv2.resize(
            attention_uint8,
            (rgb_image.shape[1], rgb_image.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        ),
        cv2.COLORMAP_JET,
    )
    base_bgr = rgb_image[..., ::-1]
    blended = cv2.addWeighted(base_bgr, 1.0 - alpha, heatmap, alpha, 0)
    if peak_probability is not None:
        # Brightness is relative, so print the peak's absolute probability:
        # a tidy-looking blob at peak=0.02 is a near-uniform map.
        cv2.putText(
            blended,
            f"peak p={peak_probability:.3f}"
            + ("" if gamma == 1.0 else f"  g={gamma:g}"),
            (4, 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return blended


class ActorFeatureOverlayWorker(threading.Thread):
    """Render actor attention without blocking the robot control loop."""

    def __init__(self, display_queue, alpha, mode="energy", gamma=1.0):
        super().__init__(daemon=True)
        self.display_queue = display_queue
        self.alpha = alpha
        self.mode = mode
        self.gamma = gamma
        self.pending = queue.Queue(maxsize=1)

    def submit(self, rgb_image, attention_map):
        item = (np.asarray(rgb_image), np.asarray(attention_map, dtype=np.float32))
        try:
            self.pending.put_nowait(item)
            return
        except queue.Full:
            pass
        try:
            self.pending.get_nowait()
        except queue.Empty:
            pass
        try:
            self.pending.put_nowait(item)
        except queue.Full:
            pass

    def run(self):
        while True:
            item = self.pending.get()
            if item is None:
                return
            rgb_image, attention_map = item
            overlay = _attention_heatmap_overlay(
                rgb_image,
                attention_map,
                self.alpha,
                mode=self.mode,
                gamma=self.gamma,
            )
            self.display_queue.put({"front_camera_attention_weights": overlay})


def policy_distribution_stats(distribution):
    """Pre-tanh loc/scale, the only thing that separates saturated from confident.

    Q values are deliberately NOT computed here. Each critic forward is an
    unjitted pass through the encoder (measured: ~120 ms), and three of them
    per control step overruns a 10 Hz budget by 3.6x -- which would change the
    very behaviour the eval is meant to measure. Everything the critics need is
    in the recorded observations, so analyze_eval.py computes them afterwards
    in batch instead.
    """
    base = getattr(distribution, "distribution", None)
    if base is None:
        return {}
    return {
        "pretanh_mean": np.asarray(jax.device_get(base.loc), np.float32).reshape(-1),
        "pretanh_std": np.asarray(jax.device_get(base.scale_diag), np.float32).reshape(-1),
    }


def make_actor_feature_overlay_worker(env):
    if not FLAGS.actor_feature_overlay or FLAGS.actor_feature_overlay_period <= 0:
        return None
    unwrapped = env.unwrapped
    # The learner builds its env with fake_env=True (train_rlpd passes
    # fake_env=FLAGS.learner). FrankaEnv.__init__ sets display_image from the
    # config *before* the `if fake_env: return`, but only creates img_queue
    # afterwards -- so a fake env advertises display_image=True while having no
    # queue. That is expected on the learner and is not worth warning about.
    if getattr(unwrapped, "fake_env", False):
        return None
    missing = []
    if not getattr(unwrapped, "display_image", False):
        missing.append("config.DISPLAY_IMAGE is False")
    if not hasattr(unwrapped, "img_queue"):
        missing.append("env has no img_queue (camera display never started)")
    if missing:
        print_red(
            "[warn] actor attention overlay disabled: " + "; ".join(missing)
        )
        return None
    # vit-grounded surfaces the grounding query's logits; resnet-pretrained
    # surfaces a feature-energy map. They need different display transforms.
    encoder_type = FLAGS.encoder_type or config.encoder_type
    # Both ViT pipelines surface grounding-query logits, which are a
    # distribution over tokens; the resnet path surfaces feature energy.
    mode = "softmax" if encoder_type in ("vit-grounded", "vit-gaze") else "energy"
    gamma = FLAGS.actor_feature_overlay_gamma if mode == "softmax" else 1.0
    print_green(
        f"[actor overlay] encoder_type={encoder_type} display={mode} gamma={gamma:g}"
    )
    worker = ActorFeatureOverlayWorker(
        unwrapped.img_queue,
        FLAGS.actor_feature_overlay_alpha,
        mode=mode,
        gamma=gamma,
    )
    worker.start()
    return worker


def maybe_display_actor_feature_overlay(worker, obs, attention_map, step):
    if (
        worker is None
        or FLAGS.actor_feature_overlay_period <= 0
        or step % FLAGS.actor_feature_overlay_period != 0
    ):
        return
    if "front_camera" not in obs:
        return
    try:
        worker.submit(
            _last_rgb_frame(obs["front_camera"]),
            jax.device_get(attention_map),
        )
    except Exception as exc:
        print_red(f"[warn] failed to display actor feature overlay: {exc}")


def actor(
    agent,
    data_store,
    intvn_data_store,
    env,
    sampling_rng,
    manual_failure_penalty=0.0,
):
    """Run a single-agent actor for tennis_ball_pick / tennis_ball_pick_and_place."""
    manual_failure_penalty = float(manual_failure_penalty)
    if manual_failure_penalty > 0.0:
        raise ValueError(
            "manual_failure_penalty must be non-positive, got "
            f"{manual_failure_penalty}."
        )
    overlay_worker = make_actor_feature_overlay_worker(env)

    if FLAGS.eval_checkpoint_step:
        try:
            print("in eval mode")
            if not FLAGS.checkpoint_path:
                raise ValueError("--checkpoint_path is required in eval mode.")
            checkpoint_dir = os.path.abspath(FLAGS.checkpoint_path)
            if not os.path.isdir(checkpoint_dir):
                raise FileNotFoundError(
                    f"Eval checkpoint directory does not exist: {checkpoint_dir}"
                )
            checkpoint_step_dir = os.path.join(
                checkpoint_dir, f"checkpoint_{FLAGS.eval_checkpoint_step}"
            )
            if not os.path.isdir(checkpoint_step_dir):
                raise FileNotFoundError(
                    "Eval checkpoint step does not exist: "
                    f"{checkpoint_step_dir}"
                )
            print_green(f"Loaded previous checkpoint at step {FLAGS.eval_checkpoint_step}.")
            ckpt = checkpoints.restore_checkpoint(
                checkpoint_dir,
                agent.state,
                step=FLAGS.eval_checkpoint_step,
            )
            agent = agent.replace(state=ckpt)

            recorder = None
            if FLAGS.eval_record:
                from eval_recorder import EvalEpisodeRecorder

                recorder = EvalEpisodeRecorder(
                    FLAGS.eval_record_dir
                    or os.path.join(checkpoint_dir, "eval_recordings"),
                    gamma=FLAGS.actor_feature_overlay_gamma,
                )

            obs, _ = env.reset()
            print_green("Eval policy execution enabled.")
            key_reader = KeyReader()
            key_reader.start()
            success_counter = 0
            intervention_label = 0
            done_by_manual = False
            time_list = []
            eval_step = 0

            for episode in range(FLAGS.eval_n_trajs):
                done = False
                start_time = time.time()
                while not done:
                    sampling_rng, key = jax.random.split(sampling_rng)
                    # Recording needs the attention too, so it must not depend
                    # on --actor_feature_overlay being on.
                    if overlay_worker is not None or recorder is not None:
                        actions, attention_map, distribution = (
                            agent.sample_actions_with_attention(
                                observations=jax.device_put(obs),
                                argmax=True,
                                seed=key,
                                return_distribution=True,
                            )
                        )
                        actions, attention_map = jax.device_get(
                            (actions, attention_map)
                        )
                        step_stats = (
                            policy_distribution_stats(distribution)
                            if recorder is not None
                            else {}
                        )
                    else:
                        actions = agent.sample_actions(
                            observations=jax.device_put(obs),
                            argmax=True,
                            seed=key,
                        )
                        actions = jax.device_get(actions)
                        attention_map = None
                        step_stats = {}
                    actions = zero_action_rpy(actions)
                    maybe_display_actor_feature_overlay(
                        overlay_worker,
                        obs,
                        attention_map,
                        eval_step,
                    )
                    eval_step += 1
                    print("actions = ", actions)

                    step_obs = obs
                    next_obs, reward, done, truncated, info = env.step(actions)
                    obs = next_obs

                    if recorder is not None:
                        extras = dict(step_stats)
                        extras["attention"] = attention_map
                        recorder.step(
                            step_obs, actions, extras,
                            reward=reward, done=done, info=info,
                        )

                    key = key_reader.get_key_nowait()
                    while key is not None:
                        if key == "1":
                            done = True
                            done_by_manual = True
                            info = dict(info)
                            info["succeed"] = True
                        key = key_reader.get_key_nowait()

                    if "intervene_action" in info:
                        intervention_label = 1

                    if done:
                        # Snapshot the episode's flags before the loop clears
                        # them, so the recording reports this episode's outcome
                        # rather than the reset defaults.
                        episode_flags = {
                            "checkpoint_step": int(FLAGS.eval_checkpoint_step),
                            "succeeded": bool(reward),
                            "done_by_manual": bool(done_by_manual),
                            "intervened": bool(intervention_label),
                            "wall_seconds": round(time.time() - start_time, 2),
                        }
                        if reward:
                            dt = time.time() - start_time
                            time_list.append(dt)
                            print(dt)
                        if not intervention_label or not done_by_manual:
                            success_counter += 1
                        print(f"{success_counter}/{episode + 1}")
                        intervention_label = 0
                        done_by_manual = False

                        if FLAGS.exp_name == "tennis_ball_pick" and reward:
                            env.unwrapped.move_up()
                        if FLAGS.save_video:
                            env.unwrapped.save_video_recording(episode)
                        if recorder is not None:
                            recorder.save(episode, extra=episode_flags)
                        key_reader.drain()
                        input("reset env")
                        key_reader.drain()
                        obs, _ = env.reset()

            print(f"success rate: {success_counter / FLAGS.eval_n_trajs}")
            if time_list:
                print(f"average time: {np.mean(time_list)}")
            return
        except KeyboardInterrupt:
            return
        finally:
            if hasattr(env, "close"):
                env.close()

    start_step = 0
    if FLAGS.checkpoint_path and os.path.exists(FLAGS.checkpoint_path):
        buffer_dir = os.path.join(FLAGS.checkpoint_path, "buffer")
        buffer_pkls = []
        if os.path.exists(buffer_dir):
            buffer_pkls = natsorted(
                glob.glob(os.path.join(buffer_dir, "transitions_*.pkl"))
            )
        if buffer_pkls:
            last_pkl = os.path.basename(buffer_pkls[-1])
            start_step = int(last_pkl.replace("transitions_", "").replace(".pkl", "")) + 1

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

    def update_params(params):
        nonlocal agent
        agent = agent.replace(state=agent.state.replace(params=params))

    client.recv_network_callback(update_params)

    transitions = []
    demo_transitions = []
    obs, _ = env.reset()
    timer = Timer()
    running_return = 0.0
    already_intervened = False
    intervention_count = 0
    intervention_steps = 0
    episode_index = 0
    key_reader = KeyReader()
    key_reader.start()

    pbar = tqdm.tqdm(range(start_step, config.max_steps), dynamic_ncols=True)
    try:
        for step in pbar:
            if step > 0 and config.buffer_period > 0 and step % config.buffer_period == 0:
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

            timer.tick("total")
            with timer.context("sample_actions"):
                print_green(f"obs[state] =  {obs['state']}")
                sampling_rng, key = jax.random.split(sampling_rng)
                if overlay_worker is not None:
                    actions, attention_map = agent.sample_actions_with_attention(
                        observations=jax.device_put(obs),
                        argmax=True,
                        seed=key,
                    )
                    actions, attention_map = jax.device_get(
                        (actions, attention_map)
                    )
                else:
                    actions = agent.sample_actions(
                        observations=jax.device_put(obs),
                        argmax=True,
                        seed=key,
                    )
                    actions = jax.device_get(actions)
                    attention_map = None
                actions = zero_action_rpy(actions)
                maybe_display_actor_feature_overlay(
                    overlay_worker,
                    obs,
                    attention_map,
                    step,
                )

            with timer.context("step_env"):
                next_obs, reward, done, truncated, info = env.step(actions)

                key = key_reader.get_key_nowait()
                while key is not None:
                    if key == "1":
                        done = True
                        reward = 1.0
                        info = dict(info)
                        info["succeed"] = True
                        info["manual_success"] = True
                    elif key == "2":
                        done = True
                        reward = manual_failure_penalty
                        info = dict(info)
                        info["succeed"] = False
                        info["manual_failure"] = True
                        info["manual_failure_penalty"] = manual_failure_penalty
                        print_red(
                            "manual failure: terminating episode with "
                            f"reward={manual_failure_penalty:.3f}"
                        )
                    key = key_reader.get_key_nowait()

                if "intervene_action" in info:
                    actions = zero_action_rpy(info.pop("intervene_action"))
                    intervention_steps += 1
                    if not already_intervened:
                        intervention_count += 1
                    already_intervened = True
                else:
                    already_intervened = False
                    actions = zero_action_rpy(actions)

                print("actions = ", actions)
                running_return += reward
                transition = dict(
                    observations=obs,
                    next_observations=next_obs,
                    actions=actions,
                    rewards=reward,
                    masks=1.0 - done,
                    dones=done,
                )
                transition["grasp_penalty"] = np.float32(info.get("grasp_penalty", 0.0))
                transition["robot_arm_penalty"] = np.float32(
                    info.get("robot_arm_penalty", 0.0)
                )

                data_store.insert(transition)
                transitions.append(copy.deepcopy(transition))
                if already_intervened:
                    intvn_data_store.insert(transition)
                    demo_transitions.append(copy.deepcopy(transition))

                obs = next_obs
                if done:
                    print_green(f" task done = {done}")
                    info.setdefault("episode", {})
                    info["episode"]["intervention_count"] = intervention_count
                    info["episode"]["intervention_steps"] = intervention_steps
                    client.request("send-stats", {"environment": info})
                    pbar.set_description(f"last return: {running_return}")
                    running_return = 0.0
                    intervention_count = 0
                    intervention_steps = 0
                    already_intervened = False
                    client.update()

                    if FLAGS.save_video:
                        env.unwrapped.save_video_recording(episode_index)
                    episode_index += 1
                    if FLAGS.exp_name == "tennis_ball_pick" and reward:
                        env.unwrapped.move_up()
                    key_reader.drain()
                    input("reset env")
                    key_reader.drain()
                    obs, _ = env.reset()

            timer.tock("total")
            if step % config.log_period == 0:
                client.request("send-stats", {"timer": timer.get_average_times()})
    finally:
        key_reader.stop()

##############################################################################

def learner(rng, agent, replay_buffer, demo_buffer, wandb_logger=None):
    """
    The learner loop, which runs when "--learner" is set to True.
    """
    latest_ckpt = (
        checkpoints.latest_checkpoint(os.path.abspath(FLAGS.checkpoint_path))
        if FLAGS.checkpoint_path and os.path.exists(FLAGS.checkpoint_path)
        else None
    )
    start_step = int(os.path.basename(latest_ckpt)[11:]) + 1 if latest_ckpt else 0

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

    # Loop to wait until replay_buffer is filled. Keep this as simple prints
    # instead of tqdm because actor logs can break tqdm's carriage-return refresh.
    print_green(
        f"Filling up replay buffer: {len(replay_buffer)}/{config.training_starts}"
    )
    last_buffer_size = len(replay_buffer)
    while len(replay_buffer) < config.training_starts:
        current_buffer_size = len(replay_buffer)
        if current_buffer_size != last_buffer_size:
            print_green(
                f"Filling up replay buffer: "
                f"{current_buffer_size}/{config.training_starts}"
            )
            last_buffer_size = current_buffer_size
        time.sleep(1)
    print_green(
        f"Replay buffer ready: {len(replay_buffer)}/{config.training_starts}"
    )

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
        use_visual_aux = bool(agent.config.get("use_visual_aux", False))
        if use_visual_aux:
            train_critic_networks_to_update = train_critic_networks_to_update | frozenset(
                {"visual_aux"}
            )
            train_networks_to_update = train_networks_to_update | frozenset(
                {"visual_aux"}
            )

    for step in tqdm.tqdm(
        range(start_step, config.max_steps), dynamic_ncols=True, desc="learner"
    ):
        update_kwargs = {"train_step": step}
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
                    **update_kwargs,
                )

        with timer.context("train"):
            batch = next(replay_iterator)
            demo_batch = next(demo_iterator)
            batch = concat_batches(batch, demo_batch, axis=0)
            agent, update_info = agent.update(
                batch,
                networks_to_update=train_networks_to_update,
                **update_kwargs,
            )
        # publish the updated network
        if step > 0 and step % (config.steps_per_update) == 0:
            agent = jax.block_until_ready(agent)
            server.publish_network(agent.state.params)

        visual_aux_info = update_info.get("visual_aux", {})
        if step % 100 == 0 and "mask_grounding_loss" in visual_aux_info:
            def info_scalar(key):
                return float(np.asarray(jax.device_get(visual_aux_info[key])).mean())

            print_green(
                f"[learner step {step}] mask grounding: "
                f"td={info_scalar('visual_aux_reference_td_loss'):.4g} "
                f"total={info_scalar('mask_grounding_loss'):.4g} "
                f"cgl={info_scalar('mask_grounding_cgl_loss'):.4g} "
                f"aux={info_scalar('mask_grounding_aux_loss'):.4g} "
                f"aux/td={info_scalar('mask_grounding_to_td_ratio'):.4g} "
                f"inside={info_scalar('mask_grounding_coverage'):.3f} "
                f"outside={info_scalar('mask_grounding_outside_mass'):.3f} "
                f"entropy={info_scalar('mask_feature_entropy'):.3f} "
                f"valid={info_scalar('mask_grounding_valid_fraction'):.3f}"
            )

        if step % config.log_period == 0:
            if wandb_logger:
                wandb_logger.log(update_info, step=step)
                wandb_logger.log({"timer": timer.get_average_times()}, step=step)

        if step > 0 and config.checkpoint_period and step % config.checkpoint_period == 0:
            checkpoints.save_checkpoint(
                os.path.abspath(FLAGS.checkpoint_path), agent.state, step=step, keep=1000
            )

#############################################################################


def main(_):
    global config
    if FLAGS.exp_name not in SUPPORTED_EXPS:
        raise ValueError(
            f"train_rlpd.py now supports {sorted(SUPPORTED_EXPS)}, got {FLAGS.exp_name}."
        )
    config = NEW_MAPPING[FLAGS.exp_name]()

    assert config.batch_size % num_devices == 0
    # seed
    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, sampling_rng = jax.random.split(rng)

    env_kwargs = dict(
        fake_env=FLAGS.learner,
        save_video=FLAGS.save_video,
        classifier=True,
        enable_tactile=FLAGS.enable_tactile,
    )
    env_signature = inspect.signature(config.get_environment).parameters
    if "gaze_predictor_checkpoint_path" in env_signature:
        env_kwargs["gaze_predictor_checkpoint_path"] = FLAGS.gaze_predictor_checkpoint_path
    if "mask_predictor_checkpoint_path" in env_signature:
        env_kwargs["mask_predictor_checkpoint_path"] = FLAGS.mask_predictor_checkpoint_path
    if "mask_selection_mode" in env_signature:
        env_kwargs["mask_selection_mode"] = FLAGS.mask_selection_mode
    if "encoder_type" in env_signature:
        # The environment has to know: vit-gaze takes no mask observations and
        # no phase one-hot, so the config must not build them or load the
        # predictors that produce them.
        env_kwargs["encoder_type"] = FLAGS.encoder_type or config.encoder_type
    # Only the pick_classifier selection mode has any use for it. Passing the
    # path in the other modes left a live checkpoint reference in the run's
    # recorded configuration for a network that is never built, which reads as
    # if the classifier were still part of the gaze pipeline. The wrapper
    # already gates the load on the mode; this makes the flags agree.
    if FLAGS.mask_selection_mode == "pick_classifier":
        if "pick_classifier_checkpoint_path" in env_signature:
            env_kwargs["pick_classifier_checkpoint_path"] = (
                FLAGS.pick_classifier_checkpoint_path
            )
        if "pick_classifier_threshold" in env_signature:
            env_kwargs["pick_classifier_threshold"] = FLAGS.pick_classifier_threshold
    elif FLAGS.pick_classifier_checkpoint_path:
        print_green(
            f"[gaze] mask_selection_mode={FLAGS.mask_selection_mode}: "
            "pick classifier not loaded."
        )
    if "gaze_target_mask_dilation" in env_signature:
        env_kwargs["gaze_target_mask_dilation"] = FLAGS.gaze_target_mask_dilation
    if "condition_on_gaze_xy" in env_signature:
        # Read it off the encoder rather than taking a flag: the two state
        # columns hold either a phase one-hot or a gaze position, and only the
        # checkpoint knows which one its grounding query was trained to read.
        # Pairing them by hand is a silent failure -- the query would mix its
        # rows by whatever happened to be in those columns.
        _enc_ckpt = FLAGS.encoder_checkpoint_path or getattr(
            config, "encoder_checkpoint_path", None
        )
        _gaze_cond = False
        if _enc_ckpt:
            try:
                from serl_launcher.vision.encoder_utils import (
                    load_encoder_checkpoint_config,
                )
                _gaze_cond = bool(
                    load_encoder_checkpoint_config(_enc_ckpt).get(
                        "grounding_gaze_conditioned", False
                    )
                )
            except Exception as exc:  # noqa: BLE001 - config is advisory here
                print_red(f"[gaze] could not read encoder config: {exc}")
        env_kwargs["condition_on_gaze_xy"] = _gaze_cond
        if _gaze_cond:
            print_green(
                "[gaze] encoder is gaze-conditioned: state[..., -2:] will carry "
                "the gaze position instead of a phase one-hot"
            )
    env = config.get_environment(**env_kwargs)
    env = RecordEpisodeStatistics(env)
    sample_obs = env.observation_space.sample()
    state_dim = sample_obs["state"].shape[-1]
    action_dim = int(np.prod(env.action_space.shape))
    print_green(f"[action space] dim={action_dim} "
                f"(7 means rpy slots are still present; 4 means xyz + grip)")
    agent_factory = make_gaze_sac_pixel_agent_hybrid_single_arm

    encoder_type = FLAGS.encoder_type or config.encoder_type
    freeze_encoder = bool(getattr(config, "freeze_encoder", False))
    encoder_checkpoint_path = FLAGS.encoder_checkpoint_path or getattr(
        config, "encoder_checkpoint_path", None
    )
    if FLAGS.eval_checkpoint_step:
        if FLAGS.eval_encoder_type:
            encoder_type = FLAGS.eval_encoder_type
        if FLAGS.eval_encoder_checkpoint_path:
            encoder_checkpoint_path = FLAGS.eval_encoder_checkpoint_path

    agent_kwargs = dict(
        seed=FLAGS.seed,
        sample_obs=sample_obs,
        sample_action=env.action_space.sample(),
        image_keys=config.image_keys,
        encoder_type=encoder_type,
        discount=(FLAGS.discount if FLAGS.discount >= 0 else config.discount),
    )
    if encoder_checkpoint_path:
        agent_kwargs["encoder_checkpoint_path"] = encoder_checkpoint_path
    agent_kwargs["freeze_encoder"] = freeze_encoder
    if hasattr(config, "tactile_encoder_type"):
        agent_kwargs["tactile_encoder_type"] = config.tactile_encoder_type
    if hasattr(config, "mask_pick_place_phase_control"):
        agent_kwargs["mask_pick_place_phase_control"] = (
            config.mask_pick_place_phase_control
        )
    if hasattr(config, "mask_feature_gate_alpha"):
        agent_kwargs["mask_feature_gate_alpha"] = config.mask_feature_gate_alpha
    if hasattr(config, "mask_feature_min_gate"):
        agent_kwargs["mask_feature_min_gate"] = config.mask_feature_min_gate
    agent: SACAgent = agent_factory(**agent_kwargs)
    print(
        f"[vision_config] encoder_type={encoder_type} "
        f"tactile_encoder={getattr(config, 'tactile_encoder_type', 'cnn')} "
        f"mask_observation=True"
    )
    include_robot_arm_penalty = True
    include_grasp_penalty = True

    # replicate agent across devices
    # need the jnp.array to avoid a bug where device_put doesn't recognize primitives
    agent = jax.device_put(
        jax.tree_util.tree_map(jnp.array, agent), sharding.replicate()
    )
    
    latest_ckpt = None
    if FLAGS.checkpoint_path is not None and os.path.exists(FLAGS.checkpoint_path):
        # input("Checkpoint path already exists. Press Enter to resume training.")
        latest_ckpt = checkpoints.latest_checkpoint(FLAGS.checkpoint_path)
    
    if latest_ckpt is not None and not FLAGS.eval_checkpoint_step:
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
        wandb_logger = make_wandb_logger(
            project=getattr(
                config,
                "wandb_project",
                "tennis_ball_pick-and-place-cnn-8-17",
            ),
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
            include_robot_arm_penalty=include_robot_arm_penalty,
        )
        logged_demo_front_camera_mask = False

        def prepare_replay_transition(transition):
            nonlocal logged_demo_front_camera_mask
            transition = dict(transition)
            # Demos and buffers recorded before the action space was narrowed
            # still carry 7-dim actions; keep (x, y, z, grip) so they line up
            # with what the policy emits now.
            actions = np.asarray(transition["actions"], dtype=np.float32)
            if actions.shape[-1] > action_dim and actions.shape[-1] == 7:
                transition["actions"] = actions[..., [0, 1, 2, 6]]
            for obs_key in ("observations", "next_observations"):
                obs_dict = dict(transition[obs_key])
                state = np.asarray(obs_dict["state"], dtype=np.float32)
                if state.shape[-1] > state_dim:
                    obs_dict["state"] = state[..., :state_dim]
                missing_image_keys = [
                    image_key for image_key in config.image_keys if image_key not in obs_dict
                ]
                if missing_image_keys:
                    raise KeyError(
                        f"Demo transition {obs_key} is missing image keys "
                        f"{missing_image_keys}. Re-export demos with the same "
                        "image_keys used by training, or disable those modalities."
                    )
                if (
                    not logged_demo_front_camera_mask
                    and obs_key == "observations"
                    and "front_camera_mask1" in obs_dict
                ):
                    front_camera_mask = np.asarray(obs_dict["front_camera_mask1"])
                    state = np.asarray(obs_dict["state"], dtype=np.float32)
                    print_green(
                        "[demo front_camera_mask1 obs] "
                        f"shape={front_camera_mask.shape} "
                        f"active_pixels={int(np.count_nonzero(front_camera_mask))} "
                        f"phase={state[..., -PHASE_ONEHOT_DIM:]}"
                    )
                    logged_demo_front_camera_mask = True
                transition[obs_key] = obs_dict
            return ensure_optional_transition_fields(transition)

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
                stacked_transitions = _stack_demo_transition_history(
                    transitions,
                    env.observation_space,
                    getattr(config, "observation_horizon", 1),
                )
                for transition_index, transition in enumerate(stacked_transitions):
                    # if 'infos' in transition and 'grasp_penalty' in transition:
                    #     transition['grasp_penalty'] = transition['infos']['grasp_penalty']
                    # if 'infos' in transition and 'robot_arm_penalty' in transition['infos']:
                    #     transition['robot_arm_penalty'] = transition['infos']['robot_arm_penalty']
                    demo_buffer.insert(prepare_replay_transition(transition))
                    transitions[transition_index] = None
        print_green(f"demo buffer size: {len(demo_buffer)}")
        print_green(f"online buffer size: {len(replay_buffer)}")

        if FLAGS.checkpoint_path is not None and os.path.exists(
            os.path.join(FLAGS.checkpoint_path, "buffer")
        ):
            for file in glob.glob(os.path.join(FLAGS.checkpoint_path, "buffer/*.pkl")):
                with open(file, "rb") as f:
                    transitions = pkl.load(f)
                    stacked_transitions = _stack_demo_transition_history(
                        transitions,
                        env.observation_space,
                        getattr(config, "observation_horizon", 1),
                    )
                    for transition_index, transition in enumerate(stacked_transitions):
                        replay_buffer.insert(prepare_replay_transition(transition))
                        transitions[transition_index] = None
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
                        demo_buffer.insert(prepare_replay_transition(transition))
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
            manual_failure_penalty=config.manual_failure_penalty,
        )

    else:
        raise NotImplementedError("Must be either a learner or an actor")

if __name__ == "__main__":
    app.run(main)
