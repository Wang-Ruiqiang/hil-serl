import cv2
import numpy as np


# Ordered as top-left, top-right, bottom-right, bottom-left.
# The bottom row is intentionally above the image bottom because the eye tracker
# often loses the lower part of the monitor during manipulation.
MARKER_POINTS_NORM = np.asarray(
    [
        [0.125, 0.145],
        [0.875, 0.145],
        [0.875, 0.750],
        [0.125, 0.750],
    ],
    dtype=np.float32,
)

MARKER_NAMES = ("tl", "tr", "br", "bl")
MARKER_IDS = (0, 1, 2, 3)
ARUCO_DICT_ID = cv2.aruco.DICT_4X4_50


def _aruco_dictionary():
    return cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)


def _generate_aruco_marker(marker_id: int, size: int) -> np.ndarray:
    marker = cv2.aruco.generateImageMarker(_aruco_dictionary(), int(marker_id), int(size))
    return cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)


def marker_points_for_size(width: int, height: int) -> np.ndarray:
    size = np.asarray([max(1, width - 1), max(1, height - 1)], dtype=np.float32)
    return MARKER_POINTS_NORM * size


def draw_gaze_display_markers(image_bgr: np.ndarray) -> np.ndarray:
    vis = image_bgr.copy()
    h, w = vis.shape[:2]
    points = marker_points_for_size(w, h)
    marker_size = max(54, int(round(min(w, h) * 0.105)))
    pad = max(8, marker_size // 8)

    for name, marker_id, (x, y) in zip(MARKER_NAMES, MARKER_IDS, points):
        center = (int(round(x)), int(round(y)))
        marker = _generate_aruco_marker(marker_id, marker_size)
        tile = np.full((marker_size + pad * 2, marker_size + pad * 2, 3), 255, dtype=np.uint8)
        tile[pad : pad + marker_size, pad : pad + marker_size] = marker

        x0 = center[0] - tile.shape[1] // 2
        y0 = center[1] - tile.shape[0] // 2
        x1 = x0 + tile.shape[1]
        y1 = y0 + tile.shape[0]
        src_x0 = max(0, -x0)
        src_y0 = max(0, -y0)
        dst_x0 = max(0, x0)
        dst_y0 = max(0, y0)
        dst_x1 = min(w, x1)
        dst_y1 = min(h, y1)
        src_x1 = src_x0 + max(0, dst_x1 - dst_x0)
        src_y1 = src_y0 + max(0, dst_y1 - dst_y0)
        if dst_x1 > dst_x0 and dst_y1 > dst_y0:
            vis[dst_y0:dst_y1, dst_x0:dst_x1] = tile[src_y0:src_y1, src_x0:src_x1]

        cv2.putText(
            vis,
            f"{name}:{marker_id}",
            (max(0, x0), max(18, y0 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.45, marker_size / 95.0),
            (255, 255, 255),
            max(1, marker_size // 30),
            cv2.LINE_AA,
        )
    return vis


def detect_gaze_display_markers(image_bgr: np.ndarray):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    dictionary = _aruco_dictionary()
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(ids) == 0:
        return None

    centers_by_id = {}
    for marker_corners, marker_id in zip(corners, ids.reshape(-1)):
        if int(marker_id) not in MARKER_IDS:
            continue
        pts = marker_corners.reshape(4, 2).astype(np.float32)
        centers_by_id[int(marker_id)] = np.mean(pts, axis=0)

    if any(marker_id not in centers_by_id for marker_id in MARKER_IDS):
        return None
    return np.asarray([centers_by_id[marker_id] for marker_id in MARKER_IDS], dtype=np.float32)
