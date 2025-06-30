import pickle as pkl
import numpy as np

# 加载 pkl 文件
with open("../../demo_data/tennis_ball_pick_100_demos_2025-06-04_19-33-00.pkl", "rb") as f:
    data = pkl.load(f)

# 保存为 txt
with open("demo_obs_act_nextobs.txt", "w") as f:
    for i, t in enumerate(data):
        obs_state = t["observations"]["state"]
        next_obs_state = t["next_observations"]["state"]
        action = t["actions"]

        f.write(f"Step {i}:\n")
        f.write(f"obs_state: {np.array2string(obs_state, precision=4)}\n")
        f.write(f"action: {np.array2string(action, precision=4)}\n")
        f.write(f"next_obs_state: {np.array2string(next_obs_state, precision=4)}\n")
        f.write("-" * 50 + "\n")
