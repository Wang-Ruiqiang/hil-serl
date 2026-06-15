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
    make_gaze_sac_pixel_agent_hybrid_single_arm,
    make_sac_pixel_agent,
    make_sac_pixel_agent_hybrid_single_arm,
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
flags.DEFINE_integer("eval_n_trajs", 21, "Number of trajectories to evaluate.")
flags.DEFINE_boolean("save_video", False, "Save video.")
flags.DEFINE_integer("enable_tactile", 1, "evaluate pick or place task.")
flags.DEFINE_boolean(
    "use_gaze_relevance",
    True,
    "Use the gaze-relevance critic variant. False keeps the original HIL-RL/SAC logic.",
)
flags.DEFINE_float(
    "gaze_regularization_weight",
    0.2,
    "Weight for the gaze auxiliary critic loss when use_gaze_relevance=True.",
)
flags.DEFINE_string(
    "gaze_predictor_checkpoint_path",
    "examples/gaze_data_process/gaze_heatmap_ckpt",
    "Checkpoint directory for the frozen gaze heatmap predictor used by the actor.",
)
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


def _latest_gaze_image(obs, image_key):
    if image_key not in obs:
        return None
    image = np.asarray(obs[image_key])
    if image.ndim == 4:
        image = image[-1]
    if image.ndim != 3:
        return None
    if image.shape[-1] > 3:
        image = image[..., -3:]
    return image


def _select_gaze_predictor_image_key(obs):
    for image_key in getattr(config, "image_keys", ()):
        if "tactile" in image_key:
            continue
        if _latest_gaze_image(obs, image_key) is not None:
            return image_key
    return None


def _infer_gaze_heatmap_shape(obs):
    image_key = _select_gaze_predictor_image_key(obs)
    if image_key is None:
        if FLAGS.use_gaze_relevance:
            raise ValueError(
                "use_gaze_relevance=True but no RGB camera image was found in "
                f"config.image_keys={getattr(config, 'image_keys', None)}."
            )
        return (1, 1)
    image = _latest_gaze_image(obs, image_key)
    return int(image.shape[0]), int(image.shape[1])


def _load_gaze_predictor_if_needed(obs):
    if not FLAGS.use_gaze_relevance or not FLAGS.gaze_predictor_checkpoint_path:
        return None

    image_key = _select_gaze_predictor_image_key(obs)
    if image_key is None:
        print_red(
            "Could not load gaze predictor: no RGB camera image was found in "
            f"config.image_keys={getattr(config, 'image_keys', None)}."
        )
        return None

    image = _latest_gaze_image(obs, image_key)
    from serl_launcher.networks.gaze_point_predictor import load_gaze_point_predictor_func

    sample_observations = {
        image_key: np.zeros(
            (1, image.shape[0], image.shape[1], image.shape[2]),
            dtype=np.float32,
        )
    }
    print_green(
        "Loading frozen gaze predictor "
        f"checkpoint={FLAGS.gaze_predictor_checkpoint_path} "
        f"image_key={image_key} "
        "encoder=resnetv1-10"
    )
    predictor_func = load_gaze_point_predictor_func(
        key=np.asarray([0, 0], dtype=np.uint32),
        sample_observations=sample_observations,
        image_keys=[image_key],
        checkpoint_path=os.path.abspath(FLAGS.gaze_predictor_checkpoint_path),
        encoder_variant="resnetv1-10",
    )
    return predictor_func, image_key


def _compute_gaze_transition_fields(obs, gaze_predictor, gaze_heatmap_shape):
    if not FLAGS.use_gaze_relevance:
        return {}

    gaze_conf = 0.0
    gaze_heatmap = np.zeros(gaze_heatmap_shape, dtype=np.float32)
    if gaze_predictor is not None:
        gaze_predictor_func, image_key = gaze_predictor
        image = _latest_gaze_image(obs, image_key)
        if image is not None:
            outputs = gaze_predictor_func(
                {image_key: image[None].astype(np.float32)}
            )
            gaze_conf = float(np.asarray(outputs["gaze_conf"])[0])
            gaze_heatmap = np.asarray(outputs["gaze_heat"][0], dtype=np.float32)
            if gaze_heatmap.shape != gaze_heatmap_shape:
                gaze_heatmap = np.asarray(
                    jax.image.resize(
                        jnp.asarray(gaze_heatmap)[None, ..., None],
                        (1, *gaze_heatmap_shape, 1),
                        method="bilinear",
                    )[0, ..., 0],
                    dtype=np.float32,
                )

    return {
        "gaze_conf": np.float32(gaze_conf),
        "gaze_heatmap": gaze_heatmap,
    }


def _gaze_xy_from_heatmap(gaze_heatmap):
    gaze_heatmap = np.asarray(gaze_heatmap)
    while gaze_heatmap.ndim > 2:
        gaze_heatmap = gaze_heatmap[0]
    if gaze_heatmap.ndim != 2 or gaze_heatmap.size == 0:
        return None
    if not np.isfinite(gaze_heatmap).all() or float(np.max(gaze_heatmap)) <= 0.0:
        return None

    y, x = np.unravel_index(int(np.argmax(gaze_heatmap)), gaze_heatmap.shape)
    height, width = gaze_heatmap.shape
    x_norm = 0.0 if width <= 1 else float(x) / float(width - 1)
    y_norm = 0.0 if height <= 1 else float(y) / float(height - 1)
    return x_norm, y_norm


def _update_env_gaze_prediction_overlay(env, gaze_fields, gaze_predictor):
    try:
        set_overlay = env.unwrapped.set_gaze_prediction_overlay
    except Exception:
        return

    if not FLAGS.use_gaze_relevance or gaze_predictor is None:
        set_overlay(xy_norm=None)
        return

    _, image_key = gaze_predictor
    xy_norm = _gaze_xy_from_heatmap(gaze_fields.get("gaze_heatmap"))
    if xy_norm is None:
        set_overlay(image_key=image_key, xy_norm=None)
        return
    set_overlay(
        image_key=image_key,
        xy_norm=xy_norm,
        conf=float(gaze_fields.get("gaze_conf", 0.0)),
    )


def _ensure_gaze_transition_fields(transition, gaze_heatmap_shape):
    if not FLAGS.use_gaze_relevance:
        return transition
    transition = dict(transition)
    transition.setdefault("gaze_conf", np.float32(0.0))
    transition.setdefault("gaze_heatmap", np.zeros(gaze_heatmap_shape, dtype=np.float32))
    return transition


def _ensure_optional_transition_fields(transition):
    transition = dict(transition)
    transition.setdefault("grasp_penalty", np.float32(0.0))
    transition.setdefault("robot_arm_penalty", np.float32(0.0))
    return transition


def _add_or_compute_gaze_transition_fields(transition, gaze_predictor, gaze_heatmap_shape):
    transition = _ensure_optional_transition_fields(transition)
    if not FLAGS.use_gaze_relevance:
        return transition
    if "gaze_conf" not in transition or "gaze_heatmap" not in transition:
        transition.update(
            _compute_gaze_transition_fields(
                transition["observations"],
                gaze_predictor,
                gaze_heatmap_shape,
            )
        )
    return _ensure_gaze_transition_fields(transition, gaze_heatmap_shape)


def _pause_env_display_for_gaze_eval(env):
    try:
        env.unwrapped.pause_display()
    except Exception as exc:
        print_red(f"Could not pause env display before gaze attention viewer: {exc}")


def _normalise_heatmap_np(heatmap):
    heatmap = np.asarray(heatmap, dtype=np.float32)
    while heatmap.ndim > 2:
        heatmap = heatmap[0]
    heatmap = np.nan_to_num(heatmap, nan=0.0, posinf=0.0, neginf=0.0)
    heatmap = heatmap - np.min(heatmap)
    denom = np.max(heatmap)
    if denom > 1e-8:
        heatmap = heatmap / denom
    return heatmap


def _image_for_gaze_view(obs, image_key):
    image = _latest_gaze_image(obs, image_key)
    if image is None:
        return None
    image = np.asarray(image)
    if image.dtype != np.uint8:
        image = np.clip(image, 0.0, 255.0)
        if image.max() <= 1.0:
            image = image * 255.0
        image = image.astype(np.uint8)
    return image


def _overlay_heatmap(image_rgb, heatmap, label, value_text=None):
    import cv2

    heatmap = _normalise_heatmap_np(heatmap)
    heatmap = cv2.resize(
        heatmap,
        (image_rgb.shape[1], image_rgb.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    heatmap_color = cv2.applyColorMap((255.0 * heatmap).astype(np.uint8), cv2.COLORMAP_JET)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(image_bgr, 0.65, heatmap_color, 0.35, 0.0)
    cv2.putText(
        overlay,
        label,
        (8, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    if value_text:
        cv2.putText(
            overlay,
            value_text,
            (8, image_rgb.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return overlay


def _tactile_panel_for_gaze_view(obs, target_width):
    import cv2

    if "tactile_data" not in obs:
        return None
    tactile = np.asarray(obs["tactile_data"])
    while tactile.ndim > 3:
        tactile = tactile[0]
    if tactile.ndim == 2:
        tactile = cv2.applyColorMap(
            np.clip(tactile, 0, 255).astype(np.uint8),
            cv2.COLORMAP_JET,
        )
    elif tactile.ndim == 3:
        if tactile.shape[-1] > 3:
            tactile = tactile[..., -3:]
        if tactile.dtype != np.uint8:
            tactile = np.clip(tactile, 0.0, 255.0)
            if tactile.max() <= 1.0:
                tactile = tactile * 255.0
            tactile = tactile.astype(np.uint8)
        tactile = cv2.cvtColor(tactile, cv2.COLOR_RGB2BGR)
    else:
        return None

    panel = tactile.copy()
    cv2.putText(
        panel,
        "tactile heatmap",
        (8, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if panel.shape[1] != target_width:
        scale = target_width / float(panel.shape[1])
        panel = cv2.resize(
            panel,
            (target_width, max(1, int(panel.shape[0] * scale))),
            interpolation=cv2.INTER_LINEAR,
        )
    return panel


def _show_eval_gaze_attention(agent, obs, actions, gaze_predictor):
    if not FLAGS.use_gaze_relevance or gaze_predictor is None:
        return

    import cv2

    gaze_predictor_func, image_key = gaze_predictor
    image_rgb = _image_for_gaze_view(obs, image_key)
    if image_rgb is None:
        return

    outputs = gaze_predictor_func({image_key: image_rgb[None].astype(np.float32)})
    gaze_heatmap = np.asarray(jax.device_get(outputs["gaze_heat"]))[0]
    gaze_conf = float(np.asarray(jax.device_get(outputs["gaze_conf"]))[0])

    critic_actions = np.asarray(actions)
    if critic_actions.ndim == 1:
        critic_actions = critic_actions[None]
    critic_actions = critic_actions[..., :-1]
    gaze_relevance, attention_map = agent.forward_gaze_relevance_and_attention(
        jax.device_put(obs),
        jax.device_put(critic_actions),
        rng=None,
        train=False,
    )
    gaze_relevance = float(np.asarray(jax.device_get(gaze_relevance)).reshape(-1)[0])
    attention_map = np.asarray(jax.device_get(attention_map))
    attention_map = attention_map.reshape((-1, *attention_map.shape[-2:]))[0]

    gaze_panel = _overlay_heatmap(
        image_rgb,
        gaze_heatmap,
        "gaze predictor",
        f"gaze_conf={gaze_conf:.3f}",
    )
    attention_panel = _overlay_heatmap(
        image_rgb,
        attention_map,
        "critic attention",
        f"gaze_relevance={gaze_relevance:.3f}",
    )
    top_row = np.concatenate([gaze_panel, attention_panel], axis=1)
    tactile_panel = _tactile_panel_for_gaze_view(obs, top_row.shape[1])
    if tactile_panel is not None:
        canvas = np.concatenate([top_row, tactile_panel], axis=0)
    else:
        canvas = top_row
    cv2.imshow("eval gaze attention", canvas)
    cv2.waitKey(1)


def _close_eval_gaze_attention_window():
    try:
        import cv2

        cv2.destroyWindow("eval gaze attention")
    except Exception:
        pass


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

            if FLAGS.exp_name == "tennis_ball_place" or FLAGS.exp_name == "twist_bottle_cap":
                print_green("Loaded previous checkpoint at step 48000.")
                ckpt_pick = checkpoints.restore_checkpoint(
                    os.path.abspath(FLAGS.checkpoint_path_pick),
                    agent.state,
                    step=48000,
                )
                agent_pick = agent.replace(state=ckpt_pick)
            
            obs, _ = env.reset()
            eval_gaze_predictor = _load_gaze_predictor_if_needed(obs)
            show_eval_gaze_attention = (
                FLAGS.use_gaze_relevance
                and eval_gaze_predictor is not None
                and hasattr(agent, "forward_gaze_relevance_and_attention")
            )
            if show_eval_gaze_attention:
                _pause_env_display_for_gaze_eval(env)
            key_reader = KeyReader()
            key_reader.start()
            ckpt_step = FLAGS.eval_checkpoint_step
            done_by_manual = False
            for episode in range(FLAGS.eval_n_trajs):
                done = False
                start_time = time.time()
                
                # print_green(f"Loaded previous checkpoint at step {ckpt_step}.")
                # ckpt = checkpoints.restore_checkpoint(
                #     os.path.abspath(FLAGS.checkpoint_path),
                #     agent.state,
                #     step=ckpt_step,
                # )
                # agent = agent.replace(state=ckpt)
                
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
                        if show_eval_gaze_attention:
                            _show_eval_gaze_attention(
                                agent_pick,
                                obs,
                                actions,
                                eval_gaze_predictor,
                            )

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

                    # print_green(f"obs[state] =  {obs['state']}")
                    actions = agent.sample_actions(
                        observations=jax.device_put(obs),
                        argmax=False,
                        seed=key
                    )
                    
                    # actions = np.asarray(jax.device_get(actions))
                    actions = np.asarray(jax.device_get(actions)).copy()
                    actions[..., 3:6] = 0.0
                    if show_eval_gaze_attention:
                        _show_eval_gaze_attention(agent, obs, actions, eval_gaze_predictor)

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
                        ckpt_step += 2000
                        done_by_manual = False

                        # if FLAGS.exp_name == "tennis_ball_pick" or FLAGS.exp_name == "tennis_ball_place" or FLAGS.exp_name == "lid_grip":
                        #     env.unwrapped.stop_cur_command()
                        if FLAGS.exp_name == "tube_insertion":
                            env.open_hand(steps=20, step_time=0.05)
                            time.sleep(1.5)
                        # elif FLAGS.exp_name == "tennis_ball_pick":
                        #     env.move_up()
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
            _close_eval_gaze_attention_window()
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
    gaze_heatmap_shape = _infer_gaze_heatmap_shape(obs)
    gaze_predictor = _load_gaze_predictor_if_needed(obs)
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

    pbar = tqdm.tqdm(range(start_step, config.max_steps), dynamic_ncols=True)
    for step in pbar:
        if FLAGS.exp_name == "tennis_ball_place" or FLAGS.exp_name == "twist_bottle_cap":
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
        if (FLAGS.exp_name == "tennis_ball_place" or FLAGS.exp_name == "twist_bottle_cap") and mode == "S1_INFERENCE":
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
                observations=jax.device_put(obs),
                argmax=True,
                seed=key,
            )
            actions = np.asarray(jax.device_get(actions_sample)).copy()
            actions[..., 3:6] = 0.0
            print("actions sampled= ", actions)
            # if actions[..., 6] < 0.0:
            #     random_action = np.random.uniform(0.0, 0.30)
            #     actions[..., 6] = random_action
            gaze_fields = _compute_gaze_transition_fields(
                obs,
                gaze_predictor,
                gaze_heatmap_shape,
            )
            _update_env_gaze_prediction_overlay(
                env,
                gaze_fields,
                gaze_predictor,
            )
            
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
            if FLAGS.exp_name == "twist_bottle_cap" or FLAGS.exp_name == "lid_grip":
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
            transition.update(gaze_fields)
            transition["grasp_penalty"] = np.float32(info.get("grasp_penalty", 0.0))
            transition["robot_arm_penalty"] = np.float32(info.get("robot_arm_penalty", 0.0))
            
            # print("info['robot_arm_penalty'] = ", info.get('robot_arm_penalty', 0))
            # print("info['grasp_penalty'] = ", info.get('grasp_penalty', 0))
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
                # if FLAGS.exp_name == "tennis_ball_pick" or FLAGS.exp_name == "tennis_ball_place" or FLAGS.exp_name == "lid_grip":
                #     env.unwrapped.stop_cur_command()
                if FLAGS.save_video:
                    env.unwrapped.save_video_recording(demo_count)
                demo_count += 1
                # print("demo count: ", demo_count)
                if FLAGS.exp_name == "tube_insertion":
                    env.open_hand(steps=20, step_time=0.05)
                    time.sleep(1.5)
                # elif FLAGS.exp_name == "tennis_ball_pick":
                #     env.move_up()
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
    sample_obs = env.observation_space.sample()
    gaze_heatmap_shape = _infer_gaze_heatmap_shape(sample_obs)

    agent_factory = (
        make_gaze_sac_pixel_agent_hybrid_single_arm
        if FLAGS.use_gaze_relevance
        else make_sac_pixel_agent_hybrid_single_arm
    )
    gaze_agent_kwargs = {}
    if FLAGS.use_gaze_relevance:
        gaze_agent_kwargs = {
            "gaze_regularization_weight": FLAGS.gaze_regularization_weight,
            "gaze_heatmap_size": gaze_heatmap_shape,
        }
        print_green(
            "Using gaze relevance agent "
            f"(weight={FLAGS.gaze_regularization_weight}, "
            f"heatmap_size={gaze_heatmap_shape})."
        )
    else:
        print_green("Using original HIL-RL/SAC agent without gaze relevance.")

    agent: SACAgent = agent_factory(
        seed=FLAGS.seed,
        sample_obs=sample_obs,
        sample_action=env.action_space.sample(),
        image_keys=config.image_keys,
        encoder_type=config.encoder_type,
        discount=config.discount,
        # state_weights=config.state_weights,
        **gaze_agent_kwargs,
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
    if FLAGS.exp_name == "tennis_ball_place" or FLAGS.exp_name == "twist_bottle_cap":
        agent_pick: SACAgent = agent_factory(
            seed=FLAGS.seed,
            sample_obs=sample_obs,
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
            # state_weights=config.state_weights,
            **gaze_agent_kwargs,
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
            include_gaze_aux=FLAGS.use_gaze_relevance,
            gaze_heatmap_shape=gaze_heatmap_shape,
        )
        # set up wandb and logging
        wandb_logger = make_wandb_logger(
            project="tennis_ball_pick-6-15",
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
            include_gaze_aux=FLAGS.use_gaze_relevance,
            gaze_heatmap_shape=gaze_heatmap_shape,
        )
        learner_gaze_predictor = _load_gaze_predictor_if_needed(
            sample_obs
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
                    # if 'infos' in transition and 'grasp_penalty' in transition:
                    #     transition['grasp_penalty'] = transition['infos']['grasp_penalty']
                    # if 'infos' in transition and 'robot_arm_penalty' in transition['infos']:
                    #     transition['robot_arm_penalty'] = transition['infos']['robot_arm_penalty']
                    demo_buffer.insert(
                        _add_or_compute_gaze_transition_fields(
                            transition,
                            learner_gaze_predictor,
                            gaze_heatmap_shape,
                        )
                    )
        print_green(f"demo buffer size: {len(demo_buffer)}")
        print_green(f"online buffer size: {len(replay_buffer)}")

        if FLAGS.checkpoint_path is not None and os.path.exists(
            os.path.join(FLAGS.checkpoint_path, "buffer")
        ):
            for file in glob.glob(os.path.join(FLAGS.checkpoint_path, "buffer/*.pkl")):
                with open(file, "rb") as f:
                    transitions = pkl.load(f)
                    for transition in transitions:
                        replay_buffer.insert(
                            _add_or_compute_gaze_transition_fields(
                                transition,
                                learner_gaze_predictor,
                                gaze_heatmap_shape,
                            )
                        )
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
                        demo_buffer.insert(
                            _add_or_compute_gaze_transition_fields(
                                transition,
                                learner_gaze_predictor,
                                gaze_heatmap_shape,
                            )
                        )
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
