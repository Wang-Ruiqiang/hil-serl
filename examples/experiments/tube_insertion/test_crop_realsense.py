#!/usr/bin/env python3
import pyrealsense2 as rs
import numpy as np
import cv2

# -------------------------
# 裁剪区域（你提供的参数）
# -------------------------
CROP_CONFIG = {
    # "front_camera": [40, 390, 190, 390],   # y1, y2, x1, x2
    # "front_camera": [90, 380, 200, 375],   # y1, y2, x1, x2
    # "front_camera": [90, 340, 150, 400],   # y1, y2, x1, x2
    "front_camera": [240, 340, 310, 410],   # y1, y2, x1, x2
    "wrist_camera": [50, 280, 270, 500],  # y1, y2, x1, x2
    # "front_camera": [60, 340, 140, 420],   # y1, y2, x1, x2
    # "wrist_camera": [0, 480, 120, 600],  # y1, y2, x1, x2
}

# -------------------------
# 相机配置（你提供的参数）
# -------------------------
CAMERA_CONFIG = {
    "front_camera": {
        "serial_number": "218622273562",
        "exposure": 13000,
        "dim": (640, 480),
    },
    "wrist_camera": {
        "serial_number": "218622271185",
        "exposure": 10500,
        "dim": (640, 480),
    },
}


import pyrealsense2 as rs
import numpy as np

def start_camera(serial_number, width=640, height=480,
                 exposure=None, white_balance=None,
                 disable_auto=True, verbose=True):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial_number)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)

    profile = pipeline.start(config)

    dev = profile.get_device()
    sensors = dev.query_sensors()

    if verbose:
        print(f"\n[Device] {dev.get_info(rs.camera_info.name)}  S/N={dev.get_info(rs.camera_info.serial_number)}")
        for i, s in enumerate(sensors):
            try:
                print(f"  Sensor #{i}: {s.get_info(rs.camera_info.name)}")
            except Exception:
                print(f"  Sensor #{i}: (unknown name)")

    # 找一个“能调曝光/自动曝光”的 sensor（通常就是你需要的那个）
    cam_sensor = None
    for s in sensors:
        if s.supports(rs.option.exposure) or s.supports(rs.option.enable_auto_exposure):
            cam_sensor = s
            break

    if cam_sensor is None:
        print("[WARN] No sensor supports exposure/auto exposure; skip setting options.")
        return pipeline

    # 关自动
    if disable_auto:
        if cam_sensor.supports(rs.option.enable_auto_exposure):
            cam_sensor.set_option(rs.option.enable_auto_exposure, 0)
        if cam_sensor.supports(rs.option.enable_auto_white_balance):
            cam_sensor.set_option(rs.option.enable_auto_white_balance, 0)

    # 手动曝光 / WB
    if exposure is not None and cam_sensor.supports(rs.option.exposure):
        cam_sensor.set_option(rs.option.exposure, float(exposure))
    if white_balance is not None and cam_sensor.supports(rs.option.white_balance):
        cam_sensor.set_option(rs.option.white_balance, float(white_balance))

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
        *CAMERA_CONFIG["front_camera"]["dim"],
        exposure=30000, white_balance=4600,
        disable_auto=True
    )
    wrist_pipe = start_camera(
        CAMERA_CONFIG["wrist_camera"]["serial_number"],
        *CAMERA_CONFIG["wrist_camera"]["dim"],
        exposure=40000, white_balance=4600,
        disable_auto=True
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
        wrist_img = get_rgb_frame(wrist_pipe)
        if wrist_img is None:
            continue

        wrist_crop = crop_image(wrist_img, CROP_CONFIG["wrist_camera"])

        wrist_crop_128 = cv2.resize(wrist_crop, (128, 128))

        cv2.imshow("Wrist Camera - Original", wrist_img)
        cv2.imshow("Wrist Camera - Cropped", wrist_crop)
        cv2.imshow("Wrist Camera - Cropped 128x128", wrist_crop_128)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    front_pipe.stop()
    # wrist_pipe.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
