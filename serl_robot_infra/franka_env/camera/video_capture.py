import queue
import threading
import time
import numpy as np

class VideoCapture:
    def __init__(self, cap, name=None):
        if name is None:
            name = cap.name
        self.name = name
        self.q = queue.Queue()
        self.cap = cap
        self.latest_frame = None
        self.latest_lock = threading.Lock()
        self.t = threading.Thread(target=self._reader)
        self.t.daemon = False
        self.enable = True
        self.t.start()

    def _reader(self):
        while self.enable:
            ret, frame = self.cap.read()
            if not ret:
                break
            with self.latest_lock:
                self.latest_frame = frame
            if not self.q.empty():
                try:
                    self.q.get_nowait()  # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        # print(self.name, self.q.qsize())
        return self.q.get(timeout=5)

    def read_latest(self, timeout=5):
        with self.latest_lock:
            if self.latest_frame is not None:
                return self.latest_frame
        frame = self.q.get(timeout=timeout)
        with self.latest_lock:
            self.latest_frame = frame
        return frame

    def close(self):
        self.enable = False
        self.t.join()
        self.cap.close()
