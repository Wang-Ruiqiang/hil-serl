#!/usr/bin/env python3
import pyrealsense2 as rs
import numpy as np
import cv2

# -------------------------
# 裁剪区域（你提供的参数）
# -------------------------
CROP_CONFIG = {
    # "front_camera": [242, 370, 232, 360],   # y1, y2, x1, x2
    # "wrist_camera": [0, 480, 120, 600],  # y1, y2, x1, x2
    # "front_camera": [240, 360, 210, 330],   # y1, y2, x1, x2
    "front_camera": [0, 460, 60, 520],   # y1, y2, x1, x2
    # "wrist_camera": [50, 280, 270, 500],  # y1, y2, x1, x2
}

# -------------------------
# 相机配置（你提供的参数）
# -------------------------
CAMERA_CONFIG = {
    # "front_camera": {
    #     "serial_number": "318122301393",
    #     "dim": (640, 480),
    # },
    "front_camera": {
        "serial_number": "242422303461",
        "dim": (640, 480),
    },
    "wrist_camera": {
        "serial_number": "218622271185",
        "dim": (640, 480),
    },
}


def start_camera(serial_number, width=640, height=480):
    """启动指定序列号的 RealSense 相机"""
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial_number)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)

    pipeline.start(config)
    return pipeline


def get_rgb_frame(pipeline):
    """获取一帧 RGB 图像"""
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        return None
    return np.asanyarray(color_frame.get_data())


def crop_image(img, crop_range):
    """裁剪图片：crop_range = [y1, y2, x1, x2]"""
    y1, y2, x1, x2 = crop_range
    return img[y1:y2, x1:x2]


def main():
    print("启动相机中...")

    front_pipe = start_camera(
        CAMERA_CONFIG["front_camera"]["serial_number"],
        *CAMERA_CONFIG["front_camera"]["dim"]
    )
    wrist_pipe = start_camera(
        CAMERA_CONFIG["wrist_camera"]["serial_number"],
        *CAMERA_CONFIG["wrist_camera"]["dim"]
    )

    print("相机启动成功，正在获取图像...")

    while True:
        # -------- 前相机 --------
        front_img = get_rgb_frame(front_pipe)
        if front_img is None:
            continue

        front_crop = crop_image(front_img, CROP_CONFIG["front_camera"])
        
        front_crop_128 = cv2.resize(front_crop, (128, 128))

        cv2.imshow("Front Camera - Original", front_img)
        cv2.imshow("Front Camera - Cropped", front_crop)
        cv2.imshow("Front Camera - Cropped 128x128", front_crop_128)

        # -------- 腕相机 --------
        # wrist_img = get_rgb_frame(wrist_pipe)
        # if wrist_img is None:
        #     continue

        # wrist_crop = crop_image(wrist_img, CROP_CONFIG["wrist_camera"])

        # wrist_crop_128 = cv2.resize(wrist_crop, (128, 128))

        # cv2.imshow("Wrist Camera - Original", wrist_img)
        # cv2.imshow("Wrist Camera - Cropped", wrist_crop)
        # cv2.imshow("Wrist Camera - Cropped 128x128", wrist_crop_128)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    front_pipe.stop()
    # wrist_pipe.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
