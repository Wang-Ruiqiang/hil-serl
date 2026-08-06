#!/usr/bin/env python3

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml


DEFAULT_TACT_BASE_PATH = "/home/wrq/workspaces/HK_TACEXO_WANG/9DTact/shape_reconstruction"
DEFAULT_CONFIG_NAME = "shape_config.yaml"


def _load_sensor(tact_base_path, config_name):
    base_path = Path(tact_base_path).expanduser().resolve()
    sys.path.insert(0, str(base_path.parent))

    from shape_reconstruction import Sensor

    cfg_path = Path(config_name).expanduser()
    if not cfg_path.is_absolute():
        cfg_path = base_path / cfg_path
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return Sensor(cfg), cfg_path


def process_tactile_data(sensor, image_size):
    raw_img = sensor.get_rectify_crop_image()
    if raw_img is None or raw_img.size == 0:
        raise RuntimeError("raw image is empty; check camera index, USB connection, and config yaml")

    gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
    height_map = sensor.raw_image_2_height_map(gray)
    height_map = sensor.expand_image(height_map)

    heat_map_input = cv2.normalize(height_map, None, 0, 255, cv2.NORM_MINMAX)
    heat_map_input = np.uint8(heat_map_input)
    heat_map = cv2.applyColorMap(heat_map_input, cv2.COLORMAP_JET)

    raw_img = cv2.resize(raw_img, image_size, interpolation=cv2.INTER_LINEAR)
    heat_map = cv2.resize(heat_map, image_size, interpolation=cv2.INTER_LINEAR)
    return raw_img, heat_map


def main():
    parser = argparse.ArgumentParser(description="Show one tactile sensor raw image and heatmap.")
    parser.add_argument("--tact_base_path", default=DEFAULT_TACT_BASE_PATH)
    parser.add_argument("--config", default=DEFAULT_CONFIG_NAME)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--hz", type=float, default=30.0)
    args = parser.parse_args()

    sensor, cfg_path = _load_sensor(args.tact_base_path, args.config)
    image_size = (args.width, args.height)
    period = 1.0 / max(args.hz, 1e-6)
    print(f"[test_tactile_sensor] config={cfg_path}")
    print("[test_tactile_sensor] press q or Esc to quit")

    try:
        while True:
            start = time.time()
            raw_img, heat_map = process_tactile_data(sensor, image_size)
            canvas = cv2.hconcat([raw_img, heat_map])
            cv2.putText(
                canvas,
                "raw | heatmap",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("tactile raw + heatmap", canvas)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            time.sleep(max(0.0, period - (time.time() - start)))
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
