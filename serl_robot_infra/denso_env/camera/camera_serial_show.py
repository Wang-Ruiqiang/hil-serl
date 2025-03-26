import pyrealsense2 as rs

# 创建Context对象
ctx = rs.context()
devices = ctx.query_devices()

print(f"已连接相机数: {len(devices)}")

serial_numbers = []

for dev in devices:
    serial = dev.get_info(rs.camera_info.serial_number)
    serial_numbers.append(serial)
    name = dev.get_info(rs.camera_info.name)
    print(f"设备名称: {name}, 串口序列号(Serial Number): {serial}")
