import copy
import os
from tqdm import tqdm
import numpy as np
import pickle as pkl
import datetime
from absl import app, flags
import gymnasium as gym
import pinocchio as pin
import re
import json

from examples.utils import read_utils

from experiments.mappings import NEW_MAPPING

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", "tennis_ball_pick", "Name of experiment corresponding to folder.")
# flags.DEFINE_integer("successes_needed", 200, "Number of successful transistions to collect.")
flags.DEFINE_string("data_dir", "/home/qiangqiang/workspaces/data/classifier_data", "classifier data dir")
# flags.DEFINE_string("data_dir", "/home/qiangqiang/workspaces/data/test_data", "classifier data dir")
flags.DEFINE_string("robot_urdf_path", "/home/qiangqiang/workspaces/HK_TACTEXO_DATA/denso_robot_with_ati_4.urdf", "robot urdf dir")

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
    data_dir = FLAGS.data_dir

    for collect_data_dir in sorted(os.listdir(data_dir)):
        collect_data_path = os.path.join(data_dir, collect_data_dir)
        if not os.path.isdir(collect_data_path):
            continue

        frame_dirs = sorted(
            [os.path.join(collect_data_path, d) for d in os.listdir(collect_data_path) if os.path.isdir(os.path.join(collect_data_path, d))],
            key=lambda folder: int(re.search(r'frame_(\d+)', os.path.basename(folder)).group(1)) if re.search(r'frame_(\d+)', os.path.basename(folder)) else float('inf')
        )

        clip_marks_json = os.path.join(collect_data_path, 'clip_marks.json')
        with open(clip_marks_json, 'r') as f:
            clip_marks = json.load(f)

        for clip in clip_marks:
            start_frame = int(clip['start'].split('_')[-1])
            end_frame = int(clip['end'].split('_')[-1])
        
            print("start_frame = ", start_frame)
            print("end_frame = ", end_frame)
            for i in list(range(start_frame, end_frame+1)):
            # for i in list(range(start_frame, end_frame+1)):
                current_frame_path = os.path.join(collect_data_path, frame_dirs[i])
                next_frame_path = os.path.join(collect_data_path, frame_dirs[i + 1])
                if not os.path.isdir(current_frame_path) or not os.path.isdir(next_frame_path):
                    continue

                obs, is_record_success= read_utils.get_frame_data(current_frame_path, FLAGS.robot_urdf_path)
                next_obs, _ = read_utils.get_frame_data(next_frame_path, FLAGS.robot_urdf_path)

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
