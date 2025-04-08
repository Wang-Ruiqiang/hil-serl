from absl import app, flags
import os
import jax
import numpy as np
from jax import numpy as jnp
import gymnasium as gym
from serl_launcher.networks.reward_classifier import load_classifier_func
from examples.utils import read_utils


from experiments.mappings import NEW_MAPPING



FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", "tennis_ball_pick", "Name of experiment corresponding to folder.")
flags.DEFINE_integer("num_epochs", 150, "Number of training epochs.")
flags.DEFINE_integer("batch_size", 256, "Batch size.")

classifier_keys = ["front_camera", "side_camera"]
robot_urdf_path = "/home/qiangqiang/workspaces/HK_TACTEXO_DATA/denso_robot_with_ati_4.urdf"

# observation_space = gym.spaces.Dict({
#     "front_camera": gym.spaces.Box(low=0, high=255, shape=(240, 320, 3), dtype=np.uint8),
#     "side_camera": gym.spaces.Box(low=0, high=255, shape=(240, 320, 3), dtype=np.uint8),
#     "state": gym.spaces.Box(-np.inf, np.inf, shape=(23,), dtype=np.float32)
# })

log_file = "classifier_log.txt"


def main(_):
    data = read_utils.read_data(robot_urdf_path, True)
    success_count = 0
    record_success_count = 0
    success_as_fail = 0
    fail_as_success = 0

    assert FLAGS.exp_name in NEW_MAPPING, 'Experiment folder not found.'
    config = NEW_MAPPING[FLAGS.exp_name]()
    env = config.get_environment(fake_env=False, save_video=False, classifier=True)
    terminate = False
    
    classifier = load_classifier_func(
        key=jax.random.PRNGKey(0),
        sample=env.observation_space.sample(),
        image_keys=classifier_keys,
        checkpoint_path=os.path.abspath("classifier_ckpt/"),
    )

    def reward_func(obs):
        # print("classifier obs = ", classifier(obs))
        sigmoid = lambda x: 1 / (1 + jnp.exp(-x))
        # print("sigmoid(classifier(obs) = ", sigmoid(classifier(obs)))
        # added check for z position to further robustify classifier, but should work without as well
        return int(sigmoid(classifier(obs)).item() > 0.45)
    
    
    history_obs = read_utils.ObsHistoryBuffer(obs_horizon=3)
    is_first_time = True
    try:
        with open(log_file, "w") as f:
            for data_count in range(len(data)):
                obs = data[data_count]["observations"]
                is_record_success = data[data_count]["is_record_success"]

                # log_msg = f"obs = {obs}\n"
                # print(log_msg, end="")
                # f.write(log_msg)
                # if is_first_time:
                #     history_obs.reset(obs)
                #     is_first_time = False
                # else:
                #     history_obs.append(obs)
                # stacked_obs = history_obs.get_stacked_obs()

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

                if is_record_success and reward == 0:
                    success_as_fail+=1
                    print("seem success sample as failure-------------------------------------------")
                # f.flush()  

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
        print("日志文件路径:", os.path.abspath("classifier_log.txt"))
        # print(f"success rate: {success_counter / FLAGS.eval_n_trajs}")
        # print(f"average time: {np.mean(time_list)}")
        return  # after done eval, return and exit




if __name__ == "__main__":
    app.run(main)