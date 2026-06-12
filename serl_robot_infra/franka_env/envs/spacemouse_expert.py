import multiprocessing
import time
from typing import Tuple

import numpy as np
import pyspacemouse


class SpaceMouseExpert:
    """Continuously read SpaceMouse state and expose the latest action."""

    def __init__(self):
        self.manager = multiprocessing.Manager()
        self.latest_data = self.manager.dict()
        self.latest_data["action"] = [0.0] * 6
        self.latest_data["buttons"] = [0, 0]
        self.latest_data["connected"] = False

        self._stop_event = multiprocessing.Event()
        self.process = multiprocessing.Process(
            target=self._read_spacemouse,
            args=(self._stop_event,),
        )
        self.process.daemon = True
        self.process.start()

    def _read_spacemouse(self, stop_event: multiprocessing.Event):
        device = pyspacemouse.open()
        if device is None:
            print("[SpaceMouse] failed to open device")
            return

        self.latest_data["connected"] = True
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

            self.latest_data["action"] = action
            self.latest_data["buttons"] = buttons[:2]
            time.sleep(0.005)

        close_fn = getattr(device, "close", None)
        if callable(close_fn):
            close_fn()

    def get_action(self) -> Tuple[np.ndarray, list]:
        action = np.asarray(self.latest_data["action"], dtype=np.float32)
        buttons = list(self.latest_data["buttons"])
        return action, buttons

    def close(self):
        self._stop_event.set()
        self.process.join(timeout=1.0)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join()
        self.manager.shutdown()
