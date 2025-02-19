import copy
import os
from tqdm import tqdm
import numpy as np
import pickle as pkl
import datetime
from absl import app, flags
from pynput import keyboard
import gymnasium as gym
import pinocchio as pin

from examples.utils import read_utils

from experiments.mappings import NEW_MAPPING

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", "tennis_ball_pick", "Name of experiment corresponding to folder.")
flags.DEFINE_integer("successes_needed", 200, "Number of successful transistions to collect.")

robot_urdf_path = "/home/ruiqiang/workspace/HK_TACTEXO_DATA/denso_robot_with_ati_4.urdf"

is_first_run = True

def save_batch_to_pickle(batch_data, file_path):
    """
    将批次数据追加保存到 pickle 文件。
    :param batch_data: 要保存的批次数据
    :param file_path: 保存路径
    """
    with open(file_path, "ab") as f: 
        pkl.dump(batch_data, f)
        print(f"Saved batch of {len(batch_data)} transitions to {file_path}")


def main(_):
    # Action/Observation Space

    assert FLAGS.exp_name in NEW_MAPPING, 'Experiment folder not found.'
    config = NEW_MAPPING[FLAGS.exp_name]()
    env = config.get_environment(fake_env=False, save_video=False, classifier=False)

    successes = []
    failures = []

    if not os.path.exists("./classifier_data"):
        os.makedirs("./classifier_data")
    file_dir_name = "./classifier_data"
    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    success_file = os.path.join(file_dir_name, f"success_images_{uuid}.pkl")
    failure_file = os.path.join(file_dir_name, f"failure_images_{uuid}.pkl")
    batch_size = 500
    
    actions = np.zeros(env.action_space.sample().shape) 
    data_dir = "/home/ruiqiang/workspace/HK_TACTEXO_DATA/wrq_project_data"
    for collect_data_dir in sorted(os.listdir(data_dir)):
        collect_data_path = os.path.join(data_dir, collect_data_dir)
        if not os.path.isdir(collect_data_path):
            continue

        frame_dirs = sorted(os.listdir(collect_data_path))

        for i in range(len(frame_dirs) - 1):
        # for i in list(range(start_frame, end_frame+1)):
            current_frame_path = os.path.join(collect_data_path, frame_dirs[i])
            next_frame_path = os.path.join(collect_data_path, frame_dirs[i + 1])
            if not os.path.isdir(current_frame_path) or not os.path.isdir(next_frame_path):
                continue

            obs, is_record_success= read_utils.get_frame_data(current_frame_path, robot_urdf_path)
            next_obs, _ = read_utils.get_frame_data(next_frame_path, robot_urdf_path)

            transition = copy.deepcopy(
                dict(
                    observations=obs,
                    next_observations=next_obs,
                    actions=actions,
                    rewards=0,
                    masks=1.0,
                    dones=0,
                )
            )

            if is_record_success:
                successes.append(transition)
            else:
                failures.append(transition)

            if len(successes) >= batch_size:
                save_batch_to_pickle(successes, success_file)
                successes = []  # 清空内存中的成功数据

            if len(failures) >= batch_size:
                save_batch_to_pickle(failures, failure_file)
                failures = []  # 清空内存中的失败数据

    # 保存剩余数据
    if successes:
        save_batch_to_pickle(successes, success_file)
    if failures:
        save_batch_to_pickle(failures, failure_file)
        
if __name__ == "__main__":
    app.run(main)
