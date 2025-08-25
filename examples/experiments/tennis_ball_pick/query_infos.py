import pickle as pkl
import pickle
import tempfile
import os


def main():
    file_path = "../../demo_data/tennis_ball_pick_20_demos_2025-08-19_20-42-07.pkl"
    # file_path = "../../demo_data/tennis_ball_pick_19_demos_2025-08-20_17-49-18.pkl"
    with open(file_path, "rb") as f:
        transitions = []
        while True:
            try:
                transitions.extend(pkl.load(f))  # 读取并扩展列表
            except EOFError:
                break  # 读取结束
    print(f"一共 {len(transitions)} 条 transition")


    # # 随便看前几条的 infos
    # for i in range(10):
    #     print(f"第 {i} 条 infos keys:", transitions[i]["infos"].keys())
    #     print(f"第 {i} 条 infos 内容示例:", {k: type(v) for k, v in transitions[i]["infos"].items()})


if __name__ == "__main__":
    main()

