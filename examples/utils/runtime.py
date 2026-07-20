import queue
import select
import sys
import termios
import threading
import tty


MULTI_STAGE_EXP_NAMES = {"tennis_ball_place", "twist_bottle_cap"}
STOP_COMMAND_EXP_NAMES = {"tennis_ball_pick", "tennis_ball_place", "lid_grip"}


def print_green(message):
    print(f"\033[92m {message}\033[00m")


def print_red(message):
    print(f"\033[91m {message}\033[00m")


def print_yellow(message):
    print(f"\033[93m {message}\033[00m")


class KeyReader(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.q = queue.Queue()
        self._stop_event = threading.Event()
        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)

    def run(self):
        try:
            while not self._stop_event.is_set():
                if sys.stdin in select.select([sys.stdin], [], [], 0.01)[0]:
                    self.q.put(sys.stdin.read(1))
        finally:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)

    def get_key_nowait(self):
        try:
            return self.q.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        self._stop_event.set()
