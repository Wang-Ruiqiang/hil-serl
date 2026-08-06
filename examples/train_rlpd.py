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
    action = np.asarray(action).copy()
    action[..., 3:6] = 0.0
    return action


def _last_rgb_frame(image):
    image = np.asarray(image)
    if image.ndim == 4:
        image = image[-1]
    if image.ndim != 3 or image.shape[-1] < 3:
        raise ValueError(f"Expected RGB image with shape HxWxC, got {image.shape}")
    return image[..., :3].astype(np.uint8)


def _attention_heatmap_overlay(rgb_image, attention_map, alpha):
    attention = np.asarray(attention_map, dtype=np.float32)
    if attention.ndim > 2:
        attention = attention.reshape((-1, *attention.shape[-2:]))[0]
    attention = attention - np.nanmin(attention)
    denom = float(np.nanmax(attention))
    if denom > 1e-8:
        attention = attention / denom
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
    return cv2.addWeighted(base_bgr, 1.0 - alpha, heatmap, alpha, 0)


def maybe_display_actor_feature_overlay(agent, env, obs, action, step):
    if (
        not FLAGS.actor_feature_overlay
        or FLAGS.actor_feature_overlay_period <= 0
        or step % FLAGS.actor_feature_overlay_period != 0
        or not hasattr(agent, "forward_gaze_attention")
    ):
        return
    unwrapped = env.unwrapped
    if not getattr(unwrapped, "display_image", False) or not hasattr(unwrapped, "img_queue"):
        return
    if "front_camera" not in obs:
        return
    try:
        critic_action = np.asarray(action, dtype=np.float32)
        if critic_action.shape[-1] > 6:
            critic_action = critic_action[..., :-1]
        attention_map = agent.forward_gaze_attention(
            jax.device_put(obs),
            jax.device_put(critic_action),
            rng=None,
            train=False,
        )
        overlay = _attention_heatmap_overlay(
            _last_rgb_frame(obs["front_camera"]),
            jax.device_get(attention_map),
            FLAGS.actor_feature_overlay_alpha,
        )
        unwrapped.img_queue.put({"front_camera_feature_overlay": overlay})
    except Exception as exc:
        print_red(f"[warn] failed to display actor feature overlay: {exc}")


def actor(agent, data_store, intvn_data_store, env, sampling_rng):
    """Run a single-agent actor for tennis_ball_pick / tennis_ball_pick_and_place."""

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
                    actions = agent.sample_actions(
                        observations=jax.device_put(obs),
                        argmax=True,
                        seed=key,
                    )
                    actions = zero_action_rpy(jax.device_get(actions))
                    maybe_display_actor_feature_overlay(
                        agent,
                        env,
                        obs,
                        actions,
                        eval_step,
                    )
                    eval_step += 1
                    print("actions = ", actions)

                    next_obs, reward, done, truncated, info = env.step(actions)
                    obs = next_obs

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
                actions = agent.sample_actions(
                    observations=jax.device_put(obs),
                    argmax=True,
                    seed=key,
                )
                actions = zero_action_rpy(jax.device_get(actions))
                maybe_display_actor_feature_overlay(agent, env, obs, actions, step)

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
                        reward = 0.0
                        info = dict(info)
                        info["succeed"] = False
                        info["manual_failure"] = True
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
    if "pick_classifier_checkpoint_path" in env_signature:
        env_kwargs["pick_classifier_checkpoint_path"] = FLAGS.pick_classifier_checkpoint_path
    if "pick_classifier_threshold" in env_signature:
        env_kwargs["pick_classifier_threshold"] = FLAGS.pick_classifier_threshold
    if "gaze_target_mask_dilation" in env_signature:
        env_kwargs["gaze_target_mask_dilation"] = FLAGS.gaze_target_mask_dilation
    env = config.get_environment(**env_kwargs)
    env = RecordEpisodeStatistics(env)
    sample_obs = env.observation_space.sample()
    state_dim = sample_obs["state"].shape[-1]
    agent_factory = make_gaze_sac_pixel_agent_hybrid_single_arm

    agent_kwargs = dict(
        seed=FLAGS.seed,
        sample_obs=sample_obs,
        sample_action=env.action_space.sample(),
        image_keys=config.image_keys,
        encoder_type=config.encoder_type,
        discount=config.discount,
    )
    if hasattr(config, "mask_pick_place_phase_control"):
        agent_kwargs["mask_pick_place_phase_control"] = (
            config.mask_pick_place_phase_control
        )
    if hasattr(config, "mask_feature_gate_alpha"):
        agent_kwargs["mask_feature_gate_alpha"] = config.mask_feature_gate_alpha
    if hasattr(config, "mask_feature_min_gate"):
        agent_kwargs["mask_feature_min_gate"] = config.mask_feature_min_gate
    agent: SACAgent = agent_factory(**agent_kwargs)
    print(f"[vision_config] encoder_type={config.encoder_type} mask_observation=True")
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
            project="tennis_ball_pick-and-place-vit-8-5",
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
                        f"phase={state[..., -3:]}"
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
                for transition in transitions:
                    # if 'infos' in transition and 'grasp_penalty' in transition:
                    #     transition['grasp_penalty'] = transition['infos']['grasp_penalty']
                    # if 'infos' in transition and 'robot_arm_penalty' in transition['infos']:
                    #     transition['robot_arm_penalty'] = transition['infos']['robot_arm_penalty']
                    demo_buffer.insert(prepare_replay_transition(transition))
        print_green(f"demo buffer size: {len(demo_buffer)}")
        print_green(f"online buffer size: {len(replay_buffer)}")

        if FLAGS.checkpoint_path is not None and os.path.exists(
            os.path.join(FLAGS.checkpoint_path, "buffer")
        ):
            for file in glob.glob(os.path.join(FLAGS.checkpoint_path, "buffer/*.pkl")):
                with open(file, "rb") as f:
                    transitions = pkl.load(f)
                    for transition in transitions:
                        replay_buffer.insert(prepare_replay_transition(transition))
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
        )

    else:
        raise NotImplementedError("Must be either a learner or an actor")

if __name__ == "__main__":
    app.run(main)
