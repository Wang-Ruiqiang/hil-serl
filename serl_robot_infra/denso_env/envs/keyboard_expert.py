import numpy as np
import multiprocessing
from pynput import keyboard

class KeyboardExpert:
    """
    This class reads keyboard input in a separate process and provides
    a "get_action" method that returns the latest action vector.
    Keys:
        - W/S: X+
        - A/D: Y+
        - Q/E: Z+
        - Z/X: Gripper open/close (if 7-DoF control is used)
    """

    def __init__(self, exp_name: str = "tennins_ball_pick"):
        # 创建共享字典用于跨进程通信
        self.manager = multiprocessing.Manager()
        self.latest_data = self.manager.dict()
        self.latest_data["action"] = [0.0] * 7
        if exp_name == "tennis_ball_pick":
            self.LARGE_STEP = 1
            self.SMALL_STEP = 0.2
        elif exp_name == "twist_bottle_cap":
            self.LARGE_STEP = 0.5
            self.SMALL_STEP = 0.2
        elif exp_name == "tube_insertion":
            self.LARGE_STEP = 1
            self.SMALL_STEP = 0.5

        # self.process = multiprocessing.Process(target=self._read_keyboard)
        self._stop_event = multiprocessing.Event()
        self.process = multiprocessing.Process(
            target=self._read_keyboard, args=(self._stop_event,)
        )
        self.process.daemon = True
        self.process.start()

    def _read_keyboard(self, stop_event: multiprocessing.Event):
        # 记录按键状态
        current_keys = set()

        def on_press(key):
            try:
                current_keys.add(key.char)
            except AttributeError:
                pass  # ignore special keys

        def on_release(key):
            try:
                if key.char in current_keys:
                    current_keys.remove(key.char)
            except AttributeError:
                pass

        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.start()

        while not stop_event.is_set():
            action = [0.0] * 7

            # 控制 xyz 方向移动
            if 'w' in current_keys:
                # print("z+")
                action[2] += self.LARGE_STEP
            if 's' in current_keys:
                # print("z-")
                action[2] -= self.LARGE_STEP
            if 'a' in current_keys:
                # print("x-")
                action[0] -= self.LARGE_STEP
            if 'd' in current_keys:
                # print("x+")
                action[0] += self.LARGE_STEP
            if 'q' in current_keys:
                # print("y+")
                action[1] += self.LARGE_STEP
            if 'e' in current_keys:
                # print("y-")
                action[1] -= self.LARGE_STEP

            if 't' in current_keys:
                # print("z+")
                action[2] += self.SMALL_STEP
            if 'g' in current_keys:
                # print("z-")
                action[2] -= self.SMALL_STEP
            if 'f' in current_keys:
                # print("x-")
                action[0] -= self.SMALL_STEP
            if 'h' in current_keys:
                # print("x+")
                action[0] += self.SMALL_STEP
            if 'r' in current_keys:
                # print("y+")
                action[1] += self.SMALL_STEP
            if 'y' in current_keys:
                # print("y-")
                action[1] -= self.SMALL_STEP

            # rotate toward z-axis
            # if 'k' in current_keys:
            #     action[6] = 0.1
            # if 'l' in current_keys:
            #     action[6] = 0.5
                
            # if 'j' in current_keys:
            #     action[6] = 1

            # 可选：gripper 控制（i = close, o = open）
            if 'i' in current_keys:
                action[6] = 1
            if 'o' in current_keys:
                action[6] = -1

            # 缩放动作大小
            # action = [a * 1 for a in action]

            # 更新共享状态
            try:
                self.latest_data["action"] = action
            except (BrokenPipeError, EOFError, ConnectionResetError):
                break

        listener.stop()

    def get_action(self) -> tuple[np.ndarray, list]:
        action = self.latest_data["action"]
        if np.linalg.norm(action) > 0.001:
            self.latest_data["action"] = [0.0] * 7  # 只在非零动作时清空
        # self.latest_data["action"] = [0.0] * 6
        return np.array(action)

    def close(self):
        self._stop_event.set()
        self.process.join(timeout=1)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join()
        self.manager.shutdown()


