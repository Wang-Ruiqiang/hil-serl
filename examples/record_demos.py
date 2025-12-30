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


def main(_):
    assert FLAGS.exp_name in NEW_MAPPING, 'Experiment folder not found.'
    config = NEW_MAPPING[FLAGS.exp_name]()
    env = config.get_environment(fake_env=False, save_video=False, classifier=True, enable_tactile=FLAGS.enable_tactile)
    
    fd = sys.stdin.fileno()
    old_term_settings = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    
    obs, info = env.reset()
    transitions = []
    success_count = 0
    success_needed = FLAGS.successes_needed
    pbar = tqdm(total=success_needed)
    trajectory = []
    returns = 0
    
    key_reader = KeyReader()
    key_reader.start()
    try:
        while success_count < success_needed:
            actions = np.zeros(env.action_space.sample().shape)
            # actions[1] = -0.1
            next_obs, rew, done, truncated, info = env.step(actions)
            # print("reward = ", rew)
            print(f"obs[state] =  {obs['state']}")
            
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
            
            print("actions taken: ", actions)
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
                    infos=info,
                )
            )
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
                trajectory = []
                returns = 0
                if FLAGS.exp_name == "tube_insertion":
                    env.unwrapped.open_hand(steps=20, step_time=0.05)
                    time.sleep(1.5)
                input("reset env")
                obs, info = env.reset()
    finally:
        env.save_all_data_on_exit()
        if hasattr(env, "keyboard_process") and env.keyboard_process.is_alive():
            print("Shutting down keyboard process...")
            env.keyboard_process.terminate()
            env.keyboard_process.join()
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