import sys
import time
import pyrealsense2 as rs
import numpy as np
import cv2

TARGET_SERIAL = "218622271185"   # ← 改成你的相机序列号

def find_device_by_serial(ctx, serial):
    for dev in ctx.query_devices():
        if dev.get_info(rs.camera_info.serial_number) == serial:
            return dev
    return None

def device_has_color(dev):
    # 检查设备的传感器是否支持 color stream
    for sensor in dev.query_sensors():
        try:
            profiles = sensor.get_stream_profiles()
        except Exception:
            continue
        for p in profiles:
            try:
                if p.stream_type() == rs.stream.color:
                    return True
            except Exception:
                pass
    return False

def pick_first_color_profile(dev):
    # 选一个可用的 color 配置（避免某些分辨率不被该机型支持的问题）
    for sensor in dev.query_sensors():
        for p in sensor.get_stream_profiles():
            if p.stream_type() == rs.stream.color and p.format() in (rs.format.bgr8, rs.format.rgb8):
                v = p.as_video_stream_profile()
                return v.width(), v.height(), p.format(), v.fps()
    # 实在没找到就回退到常见参数（若不支持会被后面的 try/except 捕获）
    return 640, 480, rs.format.bgr8, 30

def main():
    ctx = rs.context()

    dev = find_device_by_serial(ctx, TARGET_SERIAL)
    if dev is None:
        print(f"[错误] 未找到序列号为 {TARGET_SERIAL} 的 RealSense 设备。")
        # 提示当前可用设备
        exist = [d.get_info(rs.camera_info.serial_number) for d in ctx.query_devices()]
        print("当前已连接设备序列号：", exist if exist else "（无）")
        sys.exit(1)

    name = dev.get_info(rs.camera_info.name)
    print(f"已找到设备：{name} (S/N: {TARGET_SERIAL})")

    pipeline = rs.pipeline(ctx)
    cfg = rs.config()
    cfg.enable_device(TARGET_SERIAL)

    use_color = device_has_color(dev)

    if use_color:
        w, h, fmt, fps = pick_first_color_profile(dev)
        print(f"设备支持彩色流，尝试启动 color {w}x{h} {fmt} @{fps}fps")
        try:
            cfg.enable_stream(rs.stream.color, w, h, fmt, fps)
            profile = pipeline.start(cfg)
            stream_type = "color"
        except Exception as e:
            print(f"[警告] 启动彩色失败：{e}\n改为尝试 IR 流显示。")
            pipeline.stop()
            cfg = rs.config()
            cfg.enable_device(TARGET_SERIAL)
            cfg.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
            profile = pipeline.start(cfg)
            stream_type = "ir"
    else:
        print("该设备不含 RGB 传感器（例如 D405）。改为显示红外 IR 流。")
        cfg.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
        profile = pipeline.start(cfg)
        stream_type = "ir"

    print("按 'q' 退出窗口。")
    try:
        # 抛弃前几帧，让自动曝光/增益稳定
        for _ in range(5):
            pipeline.wait_for_frames()

        while True:
            frames = pipeline.wait_for_frames()

            if stream_type == "color":
                c = frames.get_color_frame()
                if not c:
                    continue
                img = np.asanyarray(c.get_data())  # BGR8
                win = f"RGB (S/N: {TARGET_SERIAL})"
            else:
                ir = frames.get_infrared_frame(1)  # IR 左/红外1
                if not ir:
                    continue
                img = np.asanyarray(ir.get_data())  # Y8 单通道
                win = f"IR (S/N: {TARGET_SERIAL})"

            cv2.imshow(win, img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
