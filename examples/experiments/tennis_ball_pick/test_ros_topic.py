import os
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
import threading
import time
from leap_hand.srv import LeapPosition, LeapPosVelEff
from message_filters import Subscriber, ApproximateTimeSynchronizer


class ROSNodeInterface(Node):
    def __init__(self):
        super().__init__('denso_env_node')

        # Publishers（发送）
        self.arm_pub = self.create_publisher(
            PoseStamped,
            '/tactexo/robot_control',
            1
        )

        self.robot_ee_sub = self.create_subscription(
            PoseStamped,
            '/cartesian_compliance_controller/current_pose',
            self.robot_ee_callback,
            1
        )

        # self.joint_sub = self.create_subscription(
        #     JointState,
        #     '/joint_states',  # 接收机械臂关节角
        #     self.joint_callback,
        #     1
        # )

        # # Subscribers（接收）
        # self.robot_ee_sub = Subscriber(
        #     self,
        #     PoseStamped,
        #     '/cartesian_compliance_controller/current_pose',
        # )

        # self.joint_sub = Subscriber(
        #     self,
        #     JointState,
        #     '/joint_states',  # 接收机械臂关节角
        # )
        

        # self.ts = ApproximateTimeSynchronizer(
        #     [self.robot_ee_sub, self.joint_sub],
        #     queue_size=1,
        #     slop=0.05,  # Adjust slop to match sample period
        #     allow_headerless=True,
        # )
        # self.ts.registerCallback(self.sync_callback)
        
        # self.leap_position_client = self.create_client(LeapPosition, '/leap_position')

        # # Wait for the service to be available
        # while not self.leap_position_client.wait_for_service(timeout_sec=5.0):
        #     self.get_logger().info('Waiting for /leap_position service...')

        # # 同步事件
        # self.robot_ee_event = threading.Event()
        # self.joint_event = threading.Event()
        # self.hand_joint_event = threading.Event()

        # 数据存储
        self.current_joints = None
        self.current_hand_joints = None

        self.cur_position = np.zeros(3, dtype=np.float32)
        self.cur_oritation = np.zeros(4, dtype=np.float32)
        self.joint_position = np.zeros(6, dtype=np.float32)
        self.robot_ee = np.zeros(7, dtype=np.float32)  # 6D position + 4D orientation

    def sync_callback(self, robot_ee_msg, joint_msg):
        # print("Synchronized data received")
        self.new_msg_received_flag  = True
        self.robot_ee_callback(robot_ee_msg)
        self.joint_callback(joint_msg)


    def robot_ee_callback(self, msg):
        #用ros2 INFO打印接收到的数据
        position = msg.pose.position
        # self.get_logger().info(f"robot_ee_received:{position}")
        self.cur_position = np.array([position.x, position.y, position.z])
        # self.robot_ee = np.array(msg.position[:7])
        # 从 msg 中提取四元数方向数据（xyzw）
        orientation = msg.pose.orientation
        self.cur_oritation = np.array([
            orientation.w, orientation.x, orientation.y, orientation.z
        ])
        self.robot_ee = np.concatenate((self.cur_position, self.cur_oritation))
        # 设置事件为已收到数据
        # self.robot_ee_event.set()


    def joint_callback(self, msg):
        # print("msg = ", msg)

        # 将 joint name 和对应的位置打包为字典
        joint_dict = {name: pos for name, pos in zip(msg.name, msg.position)}

        # 按照你需要的顺序提取关节角：joint1 ~ joint6
        ordered_joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        ordered_joint_positions = [joint_dict.get(joint, 0.0) for joint in ordered_joint_names]

        # 保存为 numpy array
        self.joint_position = np.array(ordered_joint_positions, dtype=np.float32)

        # 设置事件为“数据已接收”
        # self.joint_event.set()


    def publish_arm_action(self, pose):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        x, y, z = map(float, pose[:3])
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = z
        ori_w, ori_x, ori_y, ori_z = map(float, pose[3:7])
        # msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = pose[:3]
        msg.pose.orientation.w = ori_w
        msg.pose.orientation.x = ori_x
        msg.pose.orientation.y = ori_y
        msg.pose.orientation.z = ori_z
        # print("msg.pose = ", msg.pose)
        self.arm_pub.publish(msg)
    

    def get_current_robot_ee(self, timeout=5.0):
        # success = self.joint_event.wait(timeout=timeout)
        # if not success:
        #     raise TimeoutError("等待机械臂数据超时，请检查ROS话题是否正常发布。")
        # self.robot_ee_event.wait()
        # self.robot_ee_event.clear()
        # self.get_logger().info(f"robot_ee_received:{self.cur_position}")
        # print("get_current_robot_ee: cur_position = ", self.cur_position)
        # print("get_current_robot_ee: cur_oritation = ", self.cur_oritation)
        return self.cur_position, self.cur_oritation
    

    def get_current_joint(self, timeout=5.0):
        # success = self.joint_event.wait(timeout=timeout)
        # if not success:
        #     raise TimeoutError("等待机械臂数据超时，请检查ROS话题是否正常发布。")
        self.joint_event.wait()
        self.joint_event.clear()
        return self.joint_position
    
    def get_robot_ee(self):
        return self.robot_ee
    
    def get_joint_position(self):
        return self.joint_position
    
    
def print_green(x):
    return print("\033[92m {}\033[00m".format(x))


def wait_until_reach(node, target_pose, timeout=5.0, threshold=0.001):
    """
    等待机械臂末端运动到目标位置（只比较 xyz), timeout 秒超时。
    """
    start = time.time()
    while True:
        cur_pos = node.get_robot_ee()  # 获取当前末端执行器位置
        # print("cur_pos = ", cur_pos)
        # input("debug")
        dist = np.linalg.norm(np.array(target_pose[:3]) - np.array(cur_pos[:3]))
        # print("dist = ", dist)
        if dist < threshold:
            print(f"[INFO] 已到达目标，距离 {dist:.4f}")
            break
        if time.time() - start > timeout:
            print(f"[WARN] 等待机械臂到位超时, 当前距离 {dist:.4f}")
            break
        time.sleep(0.02)  # 20ms刷新一次


def wait_until_reach_joint(node, target_joint, timeout=5.0, threshold=0.01):
    start = time.time()
    while True:
        cur_joint = node.get_current_joint()
        dist = np.linalg.norm(np.array(target_joint) - np.array(cur_joint))
        if dist < threshold:
            print(f"[INFO] 已到达目标关节角，距离 {dist:.4f}")
            break
        if time.time() - start > timeout:
            print(f"[WARN] 等待机械臂关节到位超时, 当前距离 {dist:.4f}")
            break
        time.sleep(0.02)


def main():
    rclpy.init()
    ros_node = ROSNodeInterface()

    # 多线程 executor
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(ros_node)
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    # 预热一下, 确保有当前位姿
    time.sleep(1.0)

    # 例子：发布几个目标点
    # 格式 [x, y, z, w, x, y, z]（注意这里四元数顺序要和你的环境对应）
    targets = [
        [0.5542395, 0.04097882, 0.18197498, -0.03245958, 0.9903968, 0.12395258, -0.0519263],
        [0.54546005, 0.03200128, 0.17311737, -0.03423627, 0.9903769, 0.12514848, -0.04815954],
        # [0.5365999, 0.02303021, 0.19960387, -0.03485114, 0.99041396, 0.12312961, -0.05200597]
    ]

    joint_targets = [
        [0.043, 0.564, 1.784, 0.071, 0.905, -0.253],
        [0.026, 0.558, 1.82, 0.077, 0.867, -0.278],
    ]

    for i, (tgt, tgt_joint) in enumerate(zip(targets, joint_targets)):
        print_green(f"targets[{i}] =  {targets[i]}")
        
        start_time = time.time()
        # 发布目标位姿
        ros_node.publish_arm_action(tgt)
        print("Action published. Waiting for robot to reach...")

        # 你可以用 sleep 或事件机制等方式等待实际运动完成（简单起见 sleep 2s）
        dt = time.time() - start_time
        time.sleep(max(0, (1.0 / 20) - dt))

        # t_end = time.time()
        # print(f"[publish End] {t_end:.6f}, Step总耗时（含sleep）: {t_end - start_time:.4f}s, 实际频率: {1.0/(t_end - start_time):.2f}Hz")
        # time.sleep(2.0)  # 等待2秒，确保机械臂有足够时间到达目标位置

        start_time = time.time()
        wait_until_reach(ros_node, tgt, timeout=5.0, threshold=0.007)
        # wait_until_reach_joint(ros_node, tgt_joint, timeout=5.0, threshold=0.01)

        t_end = time.time()
        print(f"[publish End] {t_end:.6f}, Step总耗时（含sleep）: {t_end - start_time:.4f}s, 实际频率: {1.0/(t_end - start_time):.2f}Hz")
        cur_pos = ros_node.get_robot_ee()
        # joint_pos = ros_node.get_current_joint()
        t_end = time.time()
        print(f"[get joint End] {t_end:.6f}, Step总耗时（含sleep）: {t_end - start_time:.4f}s, 实际频率: {1.0/(t_end - start_time):.2f}Hz")
        print("Actual position:", cur_pos)
        # print("Actual orientation:", np.round(cur_ori, 4))
        # print("Actual joint_pos:", joint_pos)

    ros_node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()