#!/usr/bin/env python3
import pyrealsense2 as rs
import numpy as np
import cv2

# -------------------------
# 相机配置（你的参数）
# -------------------------
CAMERA_CONFIG = {
    "front_camera": {
        "serial_number": "242422303461",
        "dim": (640, 480),
    },
    "wrist_camera": {
        "serial_number": "218622273562",
        "dim": (640, 480),
    },
}


def start_camera(serial_number, width=640, height=480):
    """启动指定序列号的 RealSense 相机，只开彩色流。"""
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial_number)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
    pipeline.start(config)
    return pipeline


def get_one_frame(pipeline):
    """获取一帧 RGB 图像。"""
    for _ in range(50):  # 多试几次，避免刚启动还没帧
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if color_frame:
            img = np.asanyarray(color_frame.get_data())
            return img
    return None


def select_roi_and_print(img, cam_name):
    """
    用鼠标在 img 上选 ROI，打印出 [y1, y2, x1, x2]。
    使用方法：
    - 用鼠标拖动选框
    - 按 Enter / Space 确认
    - 按 c 重新选择
    - 按 q 放弃并退出这个相机
    """
    while True:
        clone = img.copy()
        # OpenCV 自带的交互式 ROI 选择工具
        roi = cv2.selectROI(
            f"{cam_name} - Drag to select, Enter/Space to confirm",
            clone,
            fromCenter=False,
            showCrosshair=True,
        )
        x, y, w, h = roi  # (x, y) 左上角，w 宽，h 高

        if w == 0 or h == 0:
            print(f"[{cam_name}] 没有选中有效区域，按 q 退出，按 c 重选。")
        else:
            y1, y2 = int(y), int(y + h)
            x1, x2 = int(x), int(x + w)

            print(f"\n[{cam_name}] 选择结果：")
            print(f"  像素坐标：x={x1}~{x2}, y={y1}~{y2}")
            print(f"  裁剪参数形式（[y1, y2, x1, x2]）：")
            print(f"  [{y1}, {y2}, {x1}, {x2}]")

        print("\n按键说明：")
        print("  c：重新框选这个相机的 ROI")
        print("  q：结束这个相机，继续下一个 / 退出")

        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()

        if key == ord('c'):
            # 重新选择
            continue
        elif key == ord('q'):
            break
        else:
            # 默认结束当前相机
            break


def main():
    for cam_name, cfg in CAMERA_CONFIG.items():
        print(f"\n===== 处理 {cam_name} ({cfg['serial_number']}) =====")
        pipeline = start_camera(
            cfg["serial_number"],
            cfg["dim"][0],
            cfg["dim"][1],
        )

        print("正在获取一帧图像...")
        img = get_one_frame(pipeline)
        pipeline.stop()

        if img is None:
            print(f"[{cam_name}] 获取图像失败，跳过。")
            continue

        # 显示原图（用于对比）
        cv2.imshow(f"{cam_name} - Original", img)
        cv2.waitKey(1)  # 先展示一下原图

        # 让你在图像上用鼠标框选 ROI
        select_roi_and_print(img, cam_name)

        cv2.destroyAllWindows()

    print("\n全部相机处理完成。")


if __name__ == "__main__":
    main()
