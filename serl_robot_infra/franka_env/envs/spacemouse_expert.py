import multiprocessing
import time
from typing import Tuple

import numpy as np
import pyspacemouse


class SpaceMouseExpert:
    """Continuously read SpaceMouse state and expose the latest action."""

    def __init__(self):
        self.action = multiprocessing.Array("d", [0.0] * 6, lock=True)
        self.buttons = multiprocessing.Array("i", [0, 0], lock=True)
        self.connected = multiprocessing.Value("b", False, lock=True)

        self._stop_event = multiprocessing.Event()
        self.process = multiprocessing.Process(
            target=self._read_spacemouse,
            args=(self._stop_event, self.action, self.buttons, self.connected),
        )
        self.process.daemon = True
        self.process.start()

    def _read_spacemouse(
        self,
        stop_event: multiprocessing.Event,
        action_buffer,
        button_buffer,
        connected_flag,
    ):
        try:
            device = pyspacemouse.open()
        except Exception as exc:
            print(f"[SpaceMouse] failed to open device: {exc}")
            return
        if device is None:
            print("[SpaceMouse] failed to open device")
            return

        with connected_flag.get_lock():
            connected_flag.value = True
        try:
            while not stop_event.is_set():
                state = device.read()
                if state is None:
                    time.sleep(0.01)
                    continue

                action = [
                    -state.y,
                    state.x,
                    state.z,
                    -state.roll,
                    -state.pitch,
                    -state.yaw,
                ]
                buttons = list(getattr(state, "buttons", [0, 0]))
                if len(buttons) < 2:
                    buttons = buttons + [0] * (2 - len(buttons))

                with action_buffer.get_lock():
                    for idx, value in enumerate(action):
                        action_buffer[idx] = float(value)
                with button_buffer.get_lock():
                    button_buffer[0] = int(buttons[0])
                    button_buffer[1] = int(buttons[1])
                time.sleep(0.005)
        finally:
            with connected_flag.get_lock():
                connected_flag.value = False
            close_fn = getattr(device, "close", None)
            if callable(close_fn):
                close_fn()

    def get_action(self) -> Tuple[np.ndarray, list]:
        with self.action.get_lock():
            action = np.asarray(list(self.action), dtype=np.float32)
        with self.buttons.get_lock():
            buttons = list(self.buttons)
        return action, buttons

    def close(self):
        self._stop_event.set()
        self.process.join(timeout=1.0)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join()
