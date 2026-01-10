from absl import app, flags
import os
import sys
import jax
import cv2
import numpy as np
from jax import numpy as jnp
import gymnasium as gym


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../serl_launcher'))
sys.path.insert(0, project_root)
from serl_launcher.networks.reward_classifier import load_classifier_func
from examples.utils import read_utils

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../serl_robot_infra'))
sys.path.insert(0, project_root)

from experiments.mappings import NEW_MAPPING



FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", "tube_insertion", "Name of experiment corresponding to folder.")
flags.DEFINE_integer("num_epochs", 150, "Number of training epochs.")
flags.DEFINE_integer("batch_size", 256, "Batch size.")

classifier_keys = ["front_camera", "wrist_camera","tactile_data"]
classifier_key_weights = {"front_camera": 1.0, "wrist_camera": 1.0, "tactile_data": 1.0}
# classifier_keys = ["front_camera", "side_camera"]
robot_urdf_path = "/home/wrq/workspaces/HK_TACEXO_WANG/hm_denso_wrq_ws/src/hm_denso/hm_denso_description/urdf/denso_robot_with_ati_4.urdf"

# observation_space = gym.spaces.Dict({
#     "front_camera": gym.spaces.Box(low=0, high=255, shape=(240, 320, 3), dtype=np.uint8),
#     "side_camera": gym.spaces.Box(low=0, high=255, shape=(240, 320, 3), dtype=np.uint8),
#     "state": gym.spaces.Box(-np.inf, np.inf, shape=(23,), dtype=np.float32)
# })



def main(_):
    data, _ = read_utils.read_data(robot_urdf_path, enable_tactile=True)
    success_count = 0
    record_success_count = 0
    success_as_fail = 0
    fail_as_success = 0

    assert FLAGS.exp_name in NEW_MAPPING, 'Experiment folder not found.'
    config = NEW_MAPPING[FLAGS.exp_name]()
    env = config.get_environment(fake_env=True, save_video=False, classifier=False, enable_tactile=True)
    terminate = False
    
    classifier = load_classifier_func(
        key=jax.random.PRNGKey(0),
        sample=env.observation_space.sample(),
        image_keys=classifier_keys,
        image_key_weights=classifier_key_weights,
        checkpoint_path=os.path.abspath("classifier_ckpt_tube_insertion/"),
    )

    def reward_func(obs):
        # print("classifier obs = ", classifier(obs))
        sigmoid = lambda x: 1 / (1 + jnp.exp(-x))
        # print("sigmoid(classifier(obs) = ", sigmoid(classifier(obs)))
        # added check for z position to further robustify classifier, but should work without as well
        return int(sigmoid(classifier(obs)).item() > 0.95)
    
    
    history_obs = read_utils.ObsHistoryBuffer(obs_horizon=3)
    is_first_time = True
    failure_as_success_dir = "/home/wrq/workspaces/HK_TACEXO_WANG/recorded_data/fail_as_success"
    success_as_failure_dir = "/home/wrq/workspaces/HK_TACEXO_WANG/recorded_data/success_as_failure"
    try:
        for data_count in range(len(data)):
            obs = data[data_count]["observations"]
            is_record_success = data[data_count]["is_record_success"]

            reward = reward_func(obs)
            if is_record_success:
                print("is_record_success = ", is_record_success)
                record_success_count+=1
            print("reward = ", reward)
            print("----------------------------------------")

            if is_record_success and reward:
                success_count+=1
                print("success-------------------------------------------")

            if is_record_success==0 and reward:
                fail_as_success+=1
                print("seem failure sample as success-------------------------------------------")

                # 保存 front_camera 图像
                if "front_camera" in obs:
                    img = obs["front_camera"]
                    save_path = os.path.join(failure_as_success_dir, f"front_camera_{data_count}.jpg")
                    cv2.imwrite(save_path, img)
                    print(f"Saved front_camera image to {save_path}")

                # 如果你还想保存 tactile_data
                if "tactile_data" in obs:
                    tactile_img = obs["tactile_data"]
                    save_path = os.path.join(failure_as_success_dir, f"tactile_data_{data_count}.jpg")
                    cv2.imwrite(save_path, tactile_img)
                    print(f"Saved tactile_data image to {save_path}")


            if is_record_success and reward == 0:
                success_as_fail+=1
                print("seem success sample as failure-------------------------------------------")
                if "front_camera" in obs:
                    img = obs["front_camera"]
                    save_path = os.path.join(success_as_failure_dir, f"front_camera_{data_count}.jpg")
                    cv2.imwrite(save_path, img)
                    print(f"Saved front_camera image to {save_path}")

                # 如果你还想保存 tactile_data
                if "tactile_data" in obs:
                    tactile_img = obs["tactile_data"]
                    save_path = os.path.join(success_as_failure_dir, f"tactile_data_{data_count}.jpg")
                    cv2.imwrite(save_path, tactile_img)
                    print(f"Saved tactile_data image to {save_path}") 

    except KeyboardInterrupt:
        # 捕获 Ctrl+C 异常
        print("\nUser interrupted the program. Printing final logs...")
        
    finally:
        print("success_count:", success_count)
        print("data_count:", data_count)
        print("fail_as_success:", fail_as_success)
        print("success_as_fail:", success_as_fail)
        print("record_success_count:", record_success_count)
        print("success rate of classifier:", {success_count / record_success_count})
        print("success_as_fail rate of classifier:", {success_as_fail / record_success_count})
        print("fail_as_success rate of data_count:", {fail_as_success / data_count})
        # print(f"success rate: {success_counter / FLAGS.eval_n_trajs}")
        # print(f"average time: {np.mean(time_list)}")
        # env.close()
        return  # after done eval, return and exit




if __name__ == "__main__":
    app.run(main)