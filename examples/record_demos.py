import os
import sys
from tqdm import tqdm
import numpy as np
import copy
import pickle as pkl
import datetime
from absl import app, flags
import time
import sys, threading, queue, termios, tty, select
import json
import cv2
from pathlib import Path

# 提前输入export PYTHONPATH=$(pwd)/../serl_robot_infra:$PYTHONPATH
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../serl_robot_infra'))
# sys.path.insert(0, project_root)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../serl_launcher'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.mappings import NEW_MAPPING

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", "tennis_ball_pick", "Name of experiment corresponding to folder.")
flags.DEFINE_integer("successes_needed", 20, "Number of successful demos to collect.")
flags.DEFINE_integer("enable_tactile", 1, "evaluate pick or place task.")
flags.DEFINE_boolean("record_data", True, "Save robot/camera/tactile frame data while recording demos.")
flags.DEFINE_boolean("sam2_enable", True, "Enable inline SAM2 v2 batch labeling during recording.")
flags.DEFINE_integer("sam2_batch_episodes", 1, "Number of episodes per SAM2 labeling batch.")
flags.DEFINE_integer("sam2_y_prompts_et", 20, "ET keyframes to annotate per episode.")
flags.DEFINE_integer("sam2_y_prompts_rs", 10, "RS keyframes to annotate per episode.")


# def _stdin_key_pressed(target_char="1"):
#     """若用户按下 target_char（默认 '1'）则返回 True。否则 False。"""
#     # 检查是否有可读的输入（不阻塞）
#     if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
#         ch = sys.stdin.read(1)
#         return ch == '1'
#     return False

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


def _get_unwrapped_env(env):
    return getattr(env, "unwrapped", env)


def _int_like(x):
    try:
        if isinstance(x, (list, tuple)):
            return int(_int_like(x[0]) if len(x) else 0)
        if isinstance(x, np.ndarray):
            return int(x.reshape(-1)[0])
        return int(x)
    except Exception:
        return int(x)


def _mask_shape_from_env(env):
    spaces = getattr(getattr(env, "observation_space", None), "spaces", {})

    def _shape_of_mask(space):
        shp = tuple(space.shape)
        if len(shp) >= 3:
            return int(shp[-3]), int(shp[-2])
        if len(shp) == 2:
            return int(shp[0]), int(shp[1])
        return 128, 128

    if "images" in spaces and hasattr(spaces["images"], "spaces"):
        image_spaces = spaces["images"].spaces
        if "gaze_mask" in image_spaces:
            return _shape_of_mask(image_spaces["gaze_mask"])
    if "gaze_mask" in spaces:
        return _shape_of_mask(spaces["gaze_mask"])
    return 128, 128


def _patch_mask_into_obs(obs_obj, mask_bgr):
    if mask_bgr is None:
        return obs_obj
    mask_bgr = np.ascontiguousarray(mask_bgr.astype(np.uint8, copy=False))
    if not isinstance(obs_obj, dict):
        return {"images": {"gaze_mask": mask_bgr}}
    images = obs_obj.get("images", None)
    if isinstance(images, dict):
        images["gaze_mask"] = mask_bgr
        obs_obj["images"] = images
    else:
        obs_obj["gaze_mask"] = mask_bgr
    return obs_obj


def _resolve_frame_idx(info_dict, frame_root: Path):
    idx = info_dict.get("frame_idx", None)
    if idx is None:
        return None
    idx = _int_like(idx)
    for cand in (idx - 1, idx):
        if (frame_root / f"frame_{cand}").exists():
            return cand
    return idx - 1


def _load_mask_bgr_for_frame(fid: int, frame_root: Path, out_shape_hw=(128, 128)):
    frame_dir = frame_root / f"frame_{fid}"
    gaze_json = frame_dir / "gaze_contact.json"
    if not gaze_json.exists():
        return None
    try:
        gaze_contact = json.loads(gaze_json.read_text())
        class_id = gaze_contact.get("class_id", None)
    except Exception:
        class_id = None
    if class_id is None or int(class_id) not in (0, 1):
        return None

    mask_path = frame_dir / f"rs_mask_obj{int(class_id)}.png"
    if not mask_path.exists():
        return None
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    out_h, out_w = out_shape_hw
    return cv2.resize(mask_bgr, (out_w, out_h), interpolation=cv2.INTER_NEAREST)


def _ranges_to_fid_set(frame_ranges):
    out = set()
    for start, end in frame_ranges:
        start, end = _int_like(start), _int_like(end)
        if end < start:
            start, end = end, start
        out.update(range(start, end + 1))
    return out


def _mark_episode_start(env, reason):
    root_env = _get_unwrapped_env(env)
    if hasattr(root_env, "mark_episode_start_for_recording"):
        start = root_env.mark_episode_start_for_recording()
        if start is not None:
            print(f"[record_demos][mark] start={start} @{reason}")
            return
    if hasattr(root_env, "global_frame_id"):
        root_env._cur_ep_start = int(root_env.global_frame_id)
        print(f"[record_demos][mark] start={root_env._cur_ep_start} @{reason}")


def _collect_episode_range(env):
    root_env = _get_unwrapped_env(env)
    if not hasattr(root_env, "end_episode_and_collect"):
        return None
    rng = root_env.end_episode_and_collect()
    if rng is None:
        return None
    if isinstance(rng, np.ndarray):
        if rng.size < 2:
            return None
        flat = rng.reshape(-1)
        return int(flat[0]), int(flat[1])
    if isinstance(rng, (list, tuple)) and len(rng) >= 2:
        return int(rng[0]), int(rng[1])
    return None


def _run_sam2_v2_and_patch(env, transitions_list, frame_ranges):
    if not frame_ranges:
        print("[record_demos][SAM2] empty ranges; skip")
        return

    root_env = _get_unwrapped_env(env)
    if not hasattr(root_env, "run_inline_sam2_and_label_v2"):
        print("[record_demos][SAM2] env has no run_inline_sam2_and_label_v2; skip")
        return

    print(f"[record_demos][SAM2] run v2 on ranges={frame_ranges}")
    try:
        if hasattr(root_env, "pause_display"):
            root_env.pause_display()
        root_env.run_inline_sam2_and_label_v2(
            frame_ranges=frame_ranges,
            y_prompts_et=FLAGS.sam2_y_prompts_et,
            y_prompts_rs=FLAGS.sam2_y_prompts_rs,
            rs_select_uses_same_keyset=False,
            random_seed=42,
        )
        print("[record_demos][SAM2] v2 labeling finished")
    finally:
        if hasattr(root_env, "resume_display"):
            root_env.resume_display()

    frame_root = Path(getattr(root_env, "frame_root", getattr(root_env, "frame_save_path", "./")))
    fid_set = _ranges_to_fid_set(frame_ranges)
    out_shape_hw = _mask_shape_from_env(env)
    patched = 0

    for i, transition in enumerate(transitions_list):
        try:
            info = transition.get("infos", {})
            fid = _resolve_frame_idx(info, frame_root)
            if fid is None or fid not in fid_set:
                continue

            mask_cur = _load_mask_bgr_for_frame(fid, frame_root, out_shape_hw)
            if mask_cur is None:
                continue
            mask_next = _load_mask_bgr_for_frame(fid + 1, frame_root, out_shape_hw)
            if mask_next is None:
                mask_next = mask_cur

            transition["observations"] = _patch_mask_into_obs(transition.get("observations", {}), mask_cur)
            transition["next_observations"] = _patch_mask_into_obs(
                transition.get("next_observations", {}), mask_next
            )
            patched += 1
        except Exception as exc:
            print(f"[record_demos][SAM2][PATCH-ERROR] i={i}, info={transition.get('infos', {})}: {exc}")

    print(f"[record_demos][SAM2] patched transitions in batch: {patched}")


def main(_):
    assert FLAGS.exp_name in NEW_MAPPING, 'Experiment folder not found.'
    config = NEW_MAPPING[FLAGS.exp_name]()
    env = config.get_environment(
        fake_env=False,
        save_video=False,
        classifier=True,
        enable_tactile=FLAGS.enable_tactile,
        record_data=FLAGS.record_data or FLAGS.sam2_enable,
        record_gaze=FLAGS.sam2_enable,
    )
    
    fd = sys.stdin.fileno()
    old_term_settings = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    
    obs, info = env.reset()
    if FLAGS.record_data or FLAGS.sam2_enable:
        _mark_episode_start(env, "first reset")
    transitions = []
    success_count = 0
    success_needed = FLAGS.successes_needed
    pbar = tqdm(total=success_needed)
    trajectory = []
    returns = 0
    batch_frame_ranges = []
    episodes_in_batch = 0
    
    key_reader = KeyReader()
    key_reader.start()
    try:
        while success_count < success_needed:
            actions = np.zeros(env.action_space.sample().shape)
            # actions[1] = -0.1
            next_obs, rew, done, truncated, info = env.step(actions)
            # print("reward = ", rew)
            # print(f"obs[state] =  {obs['state']}")
            
            key = key_reader.get_key_nowait()
            while key is not None:
                if key == '1':
                    done = True
                    info = dict(info)
                    rew = 1.0
                    info['succeed'] = True
                key = key_reader.get_key_nowait()
            returns += rew
            if "intervene_action" in info:
                actions = info["intervene_action"]
            
            # print("actions taken: ", actions)
            # force_end = _stdin_key_pressed("1")
            # if force_end:
            #     done = True
            #     info = dict(info)  # 防止底层是只读映射
            #     info["succeed"] = True
            
            transition = copy.deepcopy(
                dict(
                    observations=obs,
                    actions=actions,
                    next_observations=next_obs,
                    rewards=rew,
                    masks=1.0 - done,
                    dones=done,
                    infos=copy.deepcopy(info),
                )
            )
            if 'grasp_penalty' in info:
                transition['grasp_penalty']= info['grasp_penalty']
            if 'robot_arm_penalty' in info:
                transition['robot_arm_penalty']= info['robot_arm_penalty']
                
            print("info['robot_arm_penalty'] = ", info.get('robot_arm_penalty', None))
            print("info['grasp_penalty'] = ", info.get('grasp_penalty', None))
            trajectory.append(transition)
            # if "is_pick" in info:
            #     is_pick = info["is_pick"]
            # else:
            #     is_pick = True
            
            pbar.set_description(f"Return: {returns}")

            obs = next_obs
            if done:
                if info["succeed"]:
                    # time.sleep(0.5)
                    # actions = np.zeros(env.action_space.sample().shape)
                    # # actions[6] = 1.0
                    # stable_obs, _, _, _, _ = env.step(actions)
                    # trajectory[-1]["next_observations"] = stable_obs
                    for transition in trajectory:
                        transitions.append(copy.deepcopy(transition))
                    success_count += 1
                    pbar.update(1)

                if FLAGS.sam2_enable:
                    try:
                        rng = _collect_episode_range(env)
                        if rng is not None:
                            batch_frame_ranges.append(tuple(rng))
                            episodes_in_batch += 1
                            print(
                                f"[record_demos][SAM2] collected episode range: {rng} "
                                f"({episodes_in_batch}/{FLAGS.sam2_batch_episodes})"
                            )
                            if episodes_in_batch >= FLAGS.sam2_batch_episodes:
                                print("[record_demos][SAM2] buffered data mode: postpone labeling until exit")
                                episodes_in_batch = 0
                        else:
                            print("[record_demos][SAM2] rng is None (episode had no frames?)")
                    except Exception as exc:
                        print(f"[record_demos][WARN] end_episode_and_collect failed: {exc}")

                trajectory = []
                returns = 0
                if FLAGS.exp_name == "tube_insertion":
                    env.unwrapped.open_hand(steps=20, step_time=0.05)
                    time.sleep(1.5)
                # elif FLAGS.exp_name == "tennis_ball_pick":
                #     env.move_up()
                input("reset env")
                obs, info = env.reset()
                if FLAGS.record_data or FLAGS.sam2_enable:
                    _mark_episode_start(env, "reset")
    finally:
        root_env = _get_unwrapped_env(env)
        save_owner = env if hasattr(env, "save_all_data_on_exit") else root_env
        if (FLAGS.record_data or FLAGS.sam2_enable) and hasattr(save_owner, "save_all_data_on_exit"):
            try:
                save_owner.save_all_data_on_exit()
            except Exception as exc:
                print(f"[record_demos][WARN] save_all_data_on_exit failed: {exc}")
        if FLAGS.sam2_enable and batch_frame_ranges:
            batch_size = max(1, int(FLAGS.sam2_batch_episodes))
            for start in range(0, len(batch_frame_ranges), batch_size):
                chunk = batch_frame_ranges[start : start + batch_size]
                print(
                    f"[record_demos][SAM2] labeling saved batch "
                    f"{start // batch_size + 1}: episodes={len(chunk)} ranges={chunk}"
                )
                _run_sam2_v2_and_patch(env, transitions, chunk)
            batch_frame_ranges.clear()
            episodes_in_batch = 0
        key_reader.stop()
        if hasattr(env, "keyboard_process") and env.keyboard_process.is_alive():
            print("Shutting down keyboard process...")
            env.keyboard_process.terminate()
            env.keyboard_process.join()
        if hasattr(env, "close"):
            env.close()
            
    if not os.path.exists("./demo_data"):
        os.makedirs("./demo_data")
    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"./demo_data/{FLAGS.exp_name}_{success_needed}_demos_{uuid}.pkl"
    with open(file_name, "wb") as f:
        pkl.dump(transitions, f)
        print(f"saved {success_needed} demos to {file_name}")

if __name__ == "__main__":
    app.run(main)
