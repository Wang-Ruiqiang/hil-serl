import pyrealsense2 as rs
import numpy as np
import cv2

# 获取序列号后，修改为你的相机序列号
side_camera = "234222300515"
front_camera = "242422303461"

# 配置第一个相机
pipeline_1 = rs.pipeline()
config_1 = rs.config()
config_1.enable_device(front_camera)
config_1.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline_1.start(config_1)

# 配置第二个相机
pipeline_2 = rs.pipeline()
config_2 = rs.config()
config_2.enable_device(side_camera)
config_2.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline_2.start(config_2)

try:
    while True:
        # 获取相机1图像
        frames_1 = pipeline_1.wait_for_frames()
        color_frame_1 = frames_1.get_color_frame()
        img_1 = np.asanyarray(color_frame_1.get_data())

        # 获取相机2图像
        frames_2 = pipeline_2.wait_for_frames()
        color_frame_2 = frames_2.get_color_frame()
        img_2 = np.asanyarray(color_frame_2.get_data())

        # 显示图像
        cv2.imshow(f'Camera {front_camera}', img_1)
        cv2.imshow(f'Camera {side_camera}', img_2)

        if cv2.waitKey(1) == 27:  # 按ESC退出
            break
finally:
    pipeline_1.stop()
    pipeline_2.stop()
    cv2.destroyAllWindows()
