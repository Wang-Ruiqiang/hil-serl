import cv2
import time
import os
import yaml
from shape_reconstruction import Sensor
import numpy as np

def process_tactile_data(sensor, img_size=(320,240)):
    raw_img = sensor.get_rectify_crop_image()
    if raw_img is None or raw_img.size == 0:
        raise RuntimeError("raw_img is empty (camera not opened / frame not received)")
    img_GRAY = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
    height_map = sensor.raw_image_2_height_map(img_GRAY)
    height_map = sensor.expand_image(height_map)

    heat_map_input = cv2.normalize(height_map, None, 0, 255, cv2.NORM_MINMAX)
    heat_map_input = np.uint8(heat_map_input)
    heat_map = cv2.applyColorMap(heat_map_input, cv2.COLORMAP_JET)
    heat_map = cv2.resize(heat_map, img_size, interpolation=cv2.INTER_LINEAR)

    return heat_map

def main():
    # 配置路径（确保路径正确）
    base_path = '/home/wrq/workspaces/HK_TACEXO_WANG/9DTact/shape_reconstruction/'
    thumb_cfg_path = os.path.join(base_path, "shape_config_thumb.yaml")
    index_cfg_path = os.path.join(base_path, "shape_config_index.yaml")
    middle_cfg_path = os.path.join(base_path, "shape_config_middle.yaml")

    # 加载配置
    with open(thumb_cfg_path, 'r', encoding='utf-8') as f:
        thumb_cfg = yaml.load(f, Loader=yaml.FullLoader)
    with open(index_cfg_path, 'r', encoding='utf-8') as f:
        index_cfg = yaml.load(f, Loader=yaml.FullLoader)
    with open(middle_cfg_path, 'r', encoding='utf-8') as f:
        middle_cfg = yaml.load(f, Loader=yaml.FullLoader)

    # 初始化传感器（Sensor会打开相应的摄像头）
    thumb_sensor = Sensor(thumb_cfg)
    index_sensor = Sensor(index_cfg)
    middle_sensor = Sensor(middle_cfg)

    while True:
        # 获取每个指头的热力图
        thumb_heat = process_tactile_data(thumb_sensor)
        index_heat = process_tactile_data(index_sensor)
        middle_heat = process_tactile_data(middle_sensor)

        # 拼接展示
        # canvas = cv2.hconcat([thumb_heat, index_heat, middle_heat])
        cv2.imshow("thumb Heatmaps", thumb_heat)
        cv2.imshow("index Heatmaps", index_heat)
        cv2.imshow("middle Heatmaps", middle_heat)
    

        # 按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 清理
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
