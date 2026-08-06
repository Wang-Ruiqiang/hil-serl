#!/usr/bin/env python3

import argparse
import time

import cv2


def _device_arg(value):
    try:
        return int(value)
    except ValueError:
        return value


def _fourcc(format_name):
    format_name = format_name.upper()
    if format_name in {"MJPG", "MJPEG"}:
        return cv2.VideoWriter_fourcc(*"MJPG")
    if format_name in {"YUYV", "YUY2"}:
        return cv2.VideoWriter_fourcc(*"YUYV")
    if format_name in {"AUTO", "NONE"}:
        return None
    raise ValueError(f"Unsupported format: {format_name}")


def main():
    parser = argparse.ArgumentParser(description="Open a camera with OpenCV and show frames.")
    parser.add_argument("--device", default="/dev/video13", help="Camera index or /dev/videoX path.")
    parser.add_argument("--format", default="MJPG", help="MJPG, YUYV, or AUTO.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--window", default="opencv_camera")
    args = parser.parse_args()

    cap = cv2.VideoCapture(_device_arg(args.device), cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera: {args.device}")

    fourcc = _fourcc(args.format)
    if fourcc is not None:
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    actual_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    actual_format = "".join(chr((actual_fourcc >> 8 * i) & 0xFF) for i in range(4))
    print(
        "[camera] "
        f"device={args.device} requested={args.format} "
        f"actual_format={actual_format!r} "
        f"size={cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f} "
        f"fps={cap.get(cv2.CAP_PROP_FPS):.1f}"
    )
    print("Press q or ESC in the image window to exit.")

    frame_count = 0
    last_time = time.time()
    shown_fps = 0.0
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[warn] frame not received")
                time.sleep(0.05)
                continue

            frame_count += 1
            now = time.time()
            if now - last_time >= 1.0:
                shown_fps = frame_count / (now - last_time)
                frame_count = 0
                last_time = now

            display = frame.copy()
            cv2.putText(
                display,
                f"{args.device} {actual_format.strip()} {shown_fps:.1f} fps",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(args.window, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
