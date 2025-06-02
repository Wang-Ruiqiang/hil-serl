from scipy.spatial.transform import Rotation as R
import numpy as np

# xyz = [0.06009524, -0.03424226, -0.1692224]
# rpy = [3.14, 0.0, -3.26589793e-07]

xyz = [0.03424224387, 0.06009524 , -0.1692224]
rpy = [3.14, 0.0, 6.28]

# 构造 4x4 齐次矩阵
T = np.eye(4)
T[:3, :3] = R.from_euler('xyz', rpy).as_matrix()
T[:3, 3] = xyz

# 取逆，得到 palm_lower → end_link
T_palm_lower_to_end_link = np.linalg.inv(T)
# T_palm_lower_to_end_link = T
print("T_palm_lower_to_end_link = ", T_palm_lower_to_end_link)


# import numpy as np
# from scipy.spatial.transform import Rotation as R

# # 输入的 xyz 和 rpy 数据
# xyz_list = [
#     [-0.03424224387, -0.060095249652862544332, -0.01527759642],
#     [0, 0, -0.006],
#     [0, 0, -0.034],
#     [0, 0, 0],
#     [0, 0, 0.0414],
#     [0, 0, 0.0249],
#     [0, 0, 0.1582],
# ]

# rpy_list = [
#     [-3.14, 0, -1.570796327],
#     [0, 0, 0],
#     [0, 0, -1.570796],
#     [0, 0, -1.570796],
#     [0, 0, -1.570796],
#     [0, 0, 0],
#     [0, 0, 0],
# ]

# def build_transform(xyz, rpy):
#     T = np.eye(4)
#     T[:3, :3] = R.from_euler('xyz', rpy).as_matrix()
#     T[:3, 3] = xyz
#     return T

# # 依次构造并相乘
# T_total = np.eye(4)
# for xyz, rpy in zip(xyz_list, rpy_list):
#     T = build_transform(xyz, rpy)
#     T_total = T_total @ T  # 注意：从 palm_lower → end_link，顺序乘

# # 最终的旋转矩阵
# print("T_total =\n", T_total)
# # R_total = T_total[:3, :3]