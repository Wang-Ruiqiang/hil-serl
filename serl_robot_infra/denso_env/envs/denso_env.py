"""Gym Interface for Franka"""
import os
import numpy as np
import gymnasium as gym
import cv2
import copy

from scipy.spatial.transform import Rotation
import time
import requests
import queue
import threading
import yaml

from datetime import datetime
from collections import OrderedDict
from typing import Dict

from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
import threading
from denso_env.camera.video_capture import VideoCapture
from denso_env.camera.rs_capture import RSCapture
from leap_hand.srv import LeapPosition, LeapPosVelEff
from shape_reconstruction import Sensor

from examples.utils import read_utils


class ImageDisplayer(threading.Thread):
    def __init__(self, queue, name):
        threading.Thread.__init__(self)
        self.queue = queue
        self.daemon = True  # make this a daemon thread
        self.name = name

    def run(self):
        while True:
            img_array = self.queue.get()  # retrieve an image from the queue
            if img_array is None:  # None is our signal to exit
                break

            # frame = np.concatenate(
            #     [cv2.resize(v, (128, 128)) for k, v in img_array.items() if "full" not in k], axis=1
            # )
            # cv2.imshow(self.name, frame)
            # cv2.waitKey(1)
            for k, v in img_array.items():
                img = cv2.resize(v, (320, 240))  # 每个窗口显示 320×240
                cv2.imshow(f"{self.name} - {k}", img)

            cv2.waitKey(1)


##############################################################################

class DefaultEnvConfig:
    """Default configuration for FrankaEnv. Fill in the values below."""

    SERVER_URL: str = "http://127.0.0.1:5000/"
    REALSENSE_CAMERAS: Dict = {
        "front_camera": "242422303461",
        # "side_camera": "234222300515",
        "wrist_camera": "218622271185",
    }
    EXTRA_REALSENSE_CAMERAS: Dict = {
        # "front_camera": "242422303461",
        "side_camera": "234222300515",
    }
    IMAGE_CROP: dict[str, callable] = {}
    TARGET_POSE: np.ndarray = np.zeros((7,))
    # GRASP_POSE: np.ndarray = np.zeros((6,))
    REWARD_THRESHOLD: np.ndarray = np.zeros((6,))
    ACTION_SCALE = np.zeros((3,))
    # RESET_POSE = np.zeros((6,))
    # RANDOM_RESET = False
    # RANDOM_XY_RANGE = (0.0,)
    # RANDOM_RZ_RANGE = (0.0,)
    # ABS_POSE_LIMIT_HIGH = np.zeros((6,))
    # ABS_POSE_LIMIT_LOW = np.zeros((6,))
    # COMPLIANCE_PARAM: Dict[str, float] = {}
    # RESET_PARAM: Dict[str, float] = {}
    # PRECISION_PARAM: Dict[str, float] = {}
    # LOAD_PARAM: Dict[str, float] = {
    #     "mass": 0.0,
    #     "F_x_center_load": [0.0, 0.0, 0.0],
    #     "load_inertia": [0, 0, 0, 0, 0, 0, 0, 0, 0]
    # }
    DISPLAY_IMAGE: bool = True
    # GRIPPER_SLEEP: float = 0.6
    MAX_EPISODE_LENGTH: int = 200
    # JOINT_RESET_PERIOD: int = 0


##############################################################################

class ROSNodeInterface(Node):
    def __init__(self):
        super().__init__('denso_env_node')

        # Publishers（发送）
        self.arm_pub = self.create_publisher(
            PoseStamped,
            '/tactexo/robot_control',
            10
        )

        self.publisher_hand = self.create_publisher(
            JointState, 
            '/cmd_leap', 
            10
        )

        # Subscribers（接收）
        self.robot_ee_sub = self.create_subscription(
            PoseStamped,
            '/cartesian_compliance_controller/current_pose',
            self.robot_ee_callback,
            10
        )

        self.joint_sub = self.create_subscription(
            JointState,
            '/joint_states',  # 接收机械臂关节角
            self.joint_callback,
            10
        )
        
        self.leap_position_client = self.create_client(LeapPosition, '/leap_position')

        # Wait for the service to be available
        while not self.leap_position_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().info('Waiting for /leap_position service...')

        # 同步事件
        self.robot_ee_event = threading.Event()
        self.joint_event = threading.Event()
        self.hand_joint_event = threading.Event()

        # 数据存储
        self.current_joints = None
        self.current_hand_joints = None

        self.cur_position = np.zeros(3, dtype=np.float32)
        self.cur_oritation = np.zeros(4, dtype=np.float32)


    def robot_ee_callback(self, msg):
        #用ros2 INFO打印接收到的数据 
        position = msg.pose.position
        # self.get_logger().info(f"robot_ee_received:{position}")
        self.cur_position = np.array([position.x, position.y, position.z])

        # 从 msg 中提取四元数方向数据（xyzw）
        orientation = msg.pose.orientation
        self.cur_oritation = np.array([
            orientation.w, orientation.x, orientation.y, orientation.z
        ])
        # 设置事件为已收到数据
        # self.robot_ee_event.set()


    def joint_callback(self, msg):

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


    def publish_hand_action(self, hand_joints):
        # self.get_logger().info('publish_hand_action')
        stater = JointState()
        stater.name = [f"joint_{i}" for i in range(len(hand_joints))]
        stater.position = hand_joints
        self.publisher_hand.publish(stater)
        

    # def get_current_robot_ee(self, timeout=5.0):
    #     # success = self.joint_event.wait(timeout=timeout)
    #     # if not success:
    #     #     raise TimeoutError("等待机械臂数据超时，请检查ROS话题是否正常发布。")
    #     self.robot_ee_event.wait()
    #     self.robot_ee_event.clear()
    #     return self.cur_position, self.cur_oritation
    

    def get_current_robot_ee(self, timeout=5.0):
        # success = self.robot_ee_event.wait(timeout=timeout)
        # if not success:
        #     raise TimeoutError("等待get_current_robot_ee超时，请检查ROS话题是否正常发布。")
        # self.robot_ee_event.wait()
        # self.robot_ee_event.clear()
        # self.get_logger().info(f"robot_ee_received:{self.cur_position}")
        # print("get_current_robot_ee: cur_position = ", self.cur_position)
        # print("get_current_robot_ee: cur_oritation = ", self.cur_oritation)
        return self.cur_position, self.cur_oritation
    

    def get_current_joint(self, timeout=5.0):
        # success = self.joint_event.wait(timeout=timeout)
        # if not success:
        #     raise TimeoutError("等待get_current_joint超时，请检查ROS话题是否正常发布。")
        # self.joint_event.wait()
        # self.joint_event.clear()
        return self.joint_position

    def get_current_leap_position(self):
        # Create a request for the LeapPosition service
        req = LeapPosition.Request()
        future = self.leap_position_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            return list(future.result().position)
        else:
            self.get_logger().info("Failed to get current position, using zeros")
            return [0.0] * 16
        

    def reset_cur_pose(self):
        self.cur_position = np.array([0.55513753, 0.04267503, 0.18153528])
        self.cur_oritation = np.array([-0.03244228, 0.99039508, 0.12396424, -0.05194187])

##############################################################################

class DensoEnv(gym.Env):
    def __init__(
        self,
        hz=10,
        fake_env=False,
        save_video=False,
        config: DefaultEnvConfig = None,
        set_load=False,
    ):
        self.action_scale = config.ACTION_SCALE
        self._TARGET_POSE = config.TARGET_POSE
        # self._RESET_POSE = config.RESET_POSE
        self._REWARD_THRESHOLD = config.REWARD_THRESHOLD
        self.url = config.SERVER_URL
        self.config = config
        self.max_episode_length = config.MAX_EPISODE_LENGTH
        self.display_image = config.DISPLAY_IMAGE
        # self.gripper_sleep = config.GRIPPER_SLEEP
        self.tact_base_path = config.TACT_BASE_PATH
        self.enable_tactile = config.ENABLE_TACTILE
        self.fake_env = fake_env
        self.exp_name = config.EXP_NAME



        # convert last 3 elements from euler to quat, from size (6,) to (7,)
        # self.resetpos = np.concatenate(
        #     [config.RESET_POSE[:3], euler_2_quat(config.RESET_POSE[3:])]
        # )
        self.last_gripper_act = time.time()
        self.lastsent = time.time()
        self.hz = hz

        self.save_video = save_video
        if self.save_video:
            print("Saving videos!")
            self.recording_frames = []

        # Action/Observation Space
        low  = np.array([-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -1], dtype=np.float32)
        high = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0], dtype=np.float32)
        self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        
        if not self.enable_tactile:
            print("init arm without tactaile data")

            self.observation_space = gym.spaces.Dict(
                {
                    "state": gym.spaces.Dict(
                        {
                            "tcp_pos": gym.spaces.Box(
                                -np.inf, np.inf, shape=(3,)
                            ),
                            "tcp_ori": gym.spaces.Box(
                                -np.inf, np.inf, shape=(4,)
                            ),
                            "gripper_pose": gym.spaces.Box(
                                -np.inf, np.inf, shape=(1,), dtype=np.float32
                            )
                        }
                    ),
                    "images": gym.spaces.Dict(
                        {key: gym.spaces.Box(0, 255, shape=(128, 128, 3), dtype=np.uint8) 
                                    for key in config.REALSENSE_CAMERAS}
                    ),
                }
            )
        else:
            print("init arm with tactile data")
            # self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(7,))
            # low = np.array([-1] * 6 + [0], dtype=np.float32)
            # high = np.array([1] * 6 + [4], dtype=np.float32)

            self.observation_space = gym.spaces.Dict(
                {
                    "state": gym.spaces.Dict(
                        {
                            "tcp_pos": gym.spaces.Box(
                                -np.inf, np.inf, shape=(3,)
                            ),
                            "tcp_ori": gym.spaces.Box(
                                -np.inf, np.inf, shape=(4,)
                            ),
                            "gripper_pose": gym.spaces.Box(
                                -np.inf, np.inf, shape=(1,), dtype=np.float32
                            )
                        }
                    ),
                    "images": gym.spaces.Dict(
                        {
                            **{key: gym.spaces.Box(0, 255, shape=(128, 128, 3), dtype=np.uint8) 
                                    for key in config.REALSENSE_CAMERAS},
                            "tactile_data": gym.spaces.Box(0, 255, shape=(128, 256, 3), dtype=np.uint8),
                        }
                    ),
                }
            )


        self.front_color_buffer = []            #  D435i color 
        self.front_depth_buffer = []            #  D435i depth 
        self.side_color_buffer = []           #  D435i 2 color
        self.side_depth_buffer = []           #  D435i 2 depth
        self.wrist_color_buffer = []           #  D405 color
        self.wrist_depth_buffer = []           #  D405 depth
        self.joint_buffer = []
        self.hand_state_buffer = []
        
        if fake_env:
            return
        
        if self.enable_tactile:
            # tactile configuration loading and init
            thumb_cfg_path = os.path.join(self.tact_base_path, "shape_config_thumb.yaml")
            # assert the path exists
            if not os.path.exists(thumb_cfg_path):
                raise FileNotFoundError(f"Configuration file not found: {thumb_cfg_path}")
            thumb_f = open(thumb_cfg_path, 'r+', encoding='utf-8')
            thumb_cfg = yaml.load(thumb_f, Loader=yaml.FullLoader)
            self.thumb_tactile_sensor = Sensor(thumb_cfg)
            # self.thumb_tactile_vis  = Visualizer(self.thumb_tactile_sensor.points)

            index_cfg_path = os.path.join(self.tact_base_path, "shape_config_index.yaml")
            index_f = open(index_cfg_path, 'r+', encoding='utf-8')
            index_cfg = yaml.load(index_f, Loader=yaml.FullLoader)
            self.index_tactile_sensor = Sensor(index_cfg)
            # self.index_tactile_vis  = Visualizer(self.index_tactile_sensor.points)

            middle_cfg_path = os.path.join(self.tact_base_path, "shape_config_middle.yaml")
            middle_f = open(middle_cfg_path, 'r+', encoding='utf-8')
            middle_cfg = yaml.load(middle_f, Loader=yaml.FullLoader)
            self.middle_tactile_sensor = Sensor(middle_cfg)
            # self.middle_tactile_vis  = Visualizer(self.middle_tactile_sensor.points)

            self.thumb_raw_img = []
            self.index_raw_img = []
            self.middle_raw_img = []

            self.thumb_points = []
            self.index_points = []
            self.middle_points = []

            self.thumb_heat_map = []
            self.index_heat_map = []
            self.middle_heat_map = []

            self.rthumb_raw_buffer = []  # Right thumb raw tactile image
            self.rindex_raw_buffer = []  # Right index raw tactile image
            self.rmiddle_raw_buffer = []

            self.rthumb_heatmap_buffer = []  # Right thumb heatmap tactile image
            self.rindex_heatmap_buffer = []  # Right index heatmap tactile image
            self.rmiddle_heatmap_buffer = []

            self.tac_thumb_lock = threading.Lock()
            self.tac_index_lock = threading.Lock()
            self.tac_middle_lock = threading.Lock()
            self.tac_main_lock = threading.Lock()

            # Start threads for each tactile sensor
            self.start_tac_processing()
        
        self.cap = None
        self.init_cameras(config.REALSENSE_CAMERAS, config.EXTRA_REALSENSE_CAMERAS)
        if self.display_image:
            self.img_queue = queue.Queue()
            self.displayer = ImageDisplayer(self.img_queue, self.url)
            self.displayer.start()


        self.position_data = []
        self.oritation_data = []
        self.next_hand_pos = np.zeros(16)

        rclpy.init(args=None)

        self.ros_interface = ROSNodeInterface()

        # executor = rclpy.executors.MultiThreadedExecutor()
        # executor.add_node(self.ros_interface)
        # executor_thread = threading.Thread(target=executor.spin, daemon=True)
        # executor_thread.start()

        # Spin ROS callbacks in a background thread and keep references so we can
        # shut everything down cleanly when the environment closes.
        self.executor = rclpy.executors.MultiThreadedExecutor()
        self.executor.add_node(self.ros_interface)
        self.executor_thread = threading.Thread(
            target=self.executor.spin, daemon=True
        )
        self.executor_thread.start()

        self.interpolation_thread = None
        self.thread_lock = threading.Lock()

        self.print_action = True
        self._last_step_time = None

        self.frame_save_path = "/home/ruiqiang/workspaces/HK_TACEXO_WANG/recorded_data/recorded_data_training-11-22-0"  # 可自行修改
        os.makedirs(self.frame_save_path, exist_ok=True)
        self.frame_count = 0

        self.cur_position = np.zeros(3, dtype=np.float32)
        self.cur_oritation = np.zeros(4, dtype=np.float32)

        if self.exp_name == "tennis_ball_pick":
        # grip with index
            self.gripper_close_joint = config.GRIPPER_CLOSE_JOINT
            self.gripper_open_joint = config.GRIPPER_OPEN_JOINT

        elif self.exp_name == "twist_bottle_cap":
            self.gripper_close_joint = config.GRIPPER_CLOSE_JOINT
            self.gripper_twist_joint = config.GRIPPER_TWIST_JOINT
            self.gripper_open_joint = config.GRIPPER_OPEN_JOINT

        self.curr_leap_hand_pos = list(self.gripper_open_joint)

        self.hand_state = 0.0
        self._grip_inited = False

        print("Initialized Denso")


    def start_tac_processing(self):
        # Start threads for each tactile sensor
        self.thumb_thread = threading.Thread(target=self.process_thumb_tactile, daemon=True)
        self.thumb_thread.start()
        self.index_thread = threading.Thread(target=self.process_index_tactile, daemon=True)
        self.index_thread.start()
        self.middle_thread = threading.Thread(target=self.process_middle_tactile, daemon=True)
        self.middle_thread.start()


    def close_tac_processing(self):
        # Close threads for each tactile sensor
        self.thumb_thread.join()
        self.index_thread.join()
        self.middle_thread.join()


    def step(self, action: np.ndarray) -> tuple:
        """standard gym step function."""
        start_time = time.time()
        
        action = np.clip(action, self.action_space.low, self.action_space.high)

        self.nextpos = np.concatenate((self.cur_position, self.cur_oritation), axis=0)
        cond_x = (0.5 <= self.nextpos[0] <= 0.8)
        cond_y = (-0.18 <= self.nextpos[1] <= -0.08)
        cond_z = (0.14 <= self.nextpos[2] <= 0.24)
        
        if cond_x and cond_y and cond_z:
            # 限制三个方向的 action 范围
            xyz_delta = np.clip(action[:3], -0.2, 0.2)
        else:
            # 原始范围（-1,1）
            xyz_delta = action[:3]

        self.nextpos[:3] = self.nextpos[:3] + xyz_delta * self.action_scale[0]

        # GET ORIENTATION FROM ACTION
        rpy_delta = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        # rpy_delta[0] = action[3]
        self.nextpos[3:6] = self.nextpos[3:6] + rpy_delta * self.action_scale[1]
        # self.nextpos[3:] = (
        #     Rotation.from_euler("xyz", action[3:6] * self.action_scale[1])
        #     * Rotation.from_quat(self.cur_oritation)
        # ).as_quat()
        self.nextpos[3:] = (
            Rotation.from_euler("xyz", rpy_delta * self.action_scale[1])
            * Rotation.from_quat(self.cur_oritation)
        ).as_quat()

        current_hand_pos = np.asarray(self.curr_leap_hand_pos, dtype=np.float32)
        grip_action = float(np.clip(action[6], -1.0, 1.0))

        if self.exp_name == "twist_bottle_cap":
            target_hand_pos = self.calculate_hand_pos_segmented(grip_action, current_hand_pos)
        elif self.exp_name == "tennis_ball_pick":
            target_hand_pos = self._cal_hand_pos_ball_pick(grip_action, current_hand_pos)

        # print("target_hand_pos = ", target_hand_pos)

        self.ros_interface.publish_arm_action(self.nextpos)

        if -0.3 > grip_action or grip_action > 0.3:
            self._send_leap_hand_command(target_hand_pos.copy())

        # time.sleep(1.5)
        dt = time.time() - start_time
        time.sleep(max(0, (1.0 / self.hz) - dt))
        t_end = time.time()
        # print(f"[publish End] {t_end:.6f}, Step总耗时(含sleep): {t_end - start_time:.4f}s, 实际频率: {1.0/(t_end - start_time):.2f}Hz")

        self.curr_path_length += 1
        self._update_cur_position(self.nextpos)
        self.hand_state = grip_action

        # t_end = time.time()
        # print(f"[update_position End] {t_end:.6f}, Step总耗时（含sleep）: {t_end - start_time:.4f}s, 实际频率: {1.0/(t_end - start_time):.2f}Hz")
        # print("after publish arm action cur_position = ", self.cur_position)
        self.frame_count += 1
        # self.save_training_frame()

        ob = self._get_obs()
        reward = self.compute_reward(ob)
        # print(f"reward in denso_env = {reward}")
        # done = self.curr_path_length >= self.max_episode_length or reward or self.terminate
        done = reward or self.terminate
        # t_end = time.time()
        # print(f"[Step End] {t_end:.6f}, Step总耗时（含sleep）: {t_end - start_time:.4f}s, 实际频率: {1.0/(t_end - start_time):.2f}Hz")
        # print("curr hand pos = ", self.curr_leap_hand_pos)
        # input("debug for hand pos")
        return ob, int(reward), done, False, {"succeed": reward}
    
    def _ball_pick_cal_init(self, current_hand_pos: np.ndarray):
        self._binary_lower = np.minimum(self.gripper_open_joint, self.gripper_close_joint)
        self._binary_upper = np.maximum(self.gripper_open_joint, self.gripper_close_joint)

        self._binary_cmd_pos = np.asarray(current_hand_pos, dtype=np.float32)

        diff = self.gripper_close_joint - self.gripper_open_joint
        self.max_joint_delta = float(np.max(np.abs(diff))) / 10.0

        self.is_ball_pick_cal_init = True
        

    def _cal_hand_pos_ball_pick(self, grip_action: float, current_hand_pos: np.ndarray) -> np.ndarray:
        if not hasattr(self, "is_ball_pick_cal_init") or not self.is_ball_pick_cal_init:
            self._ball_pick_cal_init(current_hand_pos)

        if abs(grip_action) < 1e-6:
            return np.asarray(self._cmd_pos, dtype=np.float32)

        if grip_action > 0:
            target = self.gripper_close_joint   # open -> close
        else:
            target = self.gripper_open_joint    # close -> open

        pos  = np.asarray(self._cmd_pos, dtype=np.float32)
        diff = target - pos
        dist = float(np.linalg.norm(diff))

        if dist < 1e-9:
            return target.copy()

        # 一步最多移动多少（和 |a| 成比例）
        max_step = max(1e-9, abs(grip_action) * self.max_joint_delta)

        if dist <= max_step:
            new_pos = target.copy()
        else:
            direction = diff / dist
            new_pos = pos + direction * max_step

        new_pos = np.clip(new_pos, self._binary_lower, self._binary_upper)
        self._cmd_pos = new_pos.copy()
        return new_pos
    

    def _segmented_init(self, current_hand_pos: np.ndarray):
        # 仅 3 个点，形成闭环（0→1→2→0）
        self._waypoints = np.stack([self.gripper_open_joint, self.gripper_close_joint, self.gripper_twist_joint], axis=0)  # (3, D)
        self._num_wp = 3

        # 上下界
        self._lower = np.minimum.reduce(self._waypoints)
        self._upper = np.maximum.reduce(self._waypoints)

        # 选择最近的路标作为当前段起点
        diffs_wp = self._waypoints - np.asarray(current_hand_pos, dtype=np.float32)
        self._seg_start_idx = int(np.argmin(np.linalg.norm(diffs_wp, axis=1)))

        # 方向与步幅
        self._seg_dir = +1
        seg_vecs = self._waypoints[(np.arange(self._num_wp) + 1) % self._num_wp] - self._waypoints
        seg_lens = np.linalg.norm(seg_vecs, axis=1)
        mean_len = float(np.mean(seg_lens))
        self._per_step_path_len = max(1e-6, mean_len / 10.0)
        self._max_joint_delta = float(np.max(np.abs(seg_vecs))) / 10.0
        self._snap_eps = 1e-6
        self.is_segmented_init = True

        self._cmd_pos = np.asarray(current_hand_pos, dtype=np.float32)
        self._stop_on_waypoint = False
        
        
    def _project_to_current_segment(self, p: np.ndarray, start: np.ndarray, seg: np.ndarray) -> tuple:
        # p = np.asarray(p, dtype=np.float32)
        # end_idx = (start_idx + seg_dir) % self._num_wp
        # start = self._waypoints[start_idx]
        # end   = self._waypoints[end_idx]
        # seg   = end - start
        seg_len2 = float(np.dot(seg, seg))
        if seg_len2 < 1e-12:
            return 0.0, start  # （正常不会发生，因为 3 个路标互不相同）
        t = float(np.dot(p - start, seg) / seg_len2)
        t = max(0.0, min(1.0, t))
        proj = start + t * seg
        return t, proj


    def calculate_hand_pos_segmented(self, grip_action: float, current_hand_pos: np.ndarray) -> np.ndarray:
        if not hasattr(self, "is_segmented_init") or not self.is_segmented_init:
            self._segmented_init(current_hand_pos)
        a = float(np.clip(grip_action, -1.0, 1.0))
        if abs(a) < 1e-6:
            # 不推进，直接维持上次指令；也可改为返回当前反馈
            return np.asarray(self._cmd_pos, dtype=np.float32)

        # 动作决定方向
        self._seg_dir = (+1 if a > 0 else -1)

        # 用“指令位置”推进（不受真实到位与否的影响）
        pos = np.asarray(self._cmd_pos, dtype=np.float32)
        remaining_path_len = abs(a) * self._per_step_path_len

        while remaining_path_len > 0:
            start_idx = self._seg_start_idx
            end_idx   = (start_idx + self._seg_dir) % self._num_wp
            start_wp = self._waypoints[start_idx]
            end_wp   = self._waypoints[end_idx]
            seg_vec  = end_wp - start_wp
            seg_len  = float(np.linalg.norm(seg_vec))
            if seg_len < 1e-9:
                # 极端退化：直接切到下一段并继续消耗
                self._seg_start_idx = end_idx
                if self._stop_on_waypoint:
                    break
                else:
                    continue

            # 注意：这里把“投影”的参考改成 pos（指令位置），而不是 current_hand_pos（真实反馈）
            segment_progress, proj = self._project_to_current_segment(pos, start_wp, seg_vec)

            # 吸附端点，避免在端点附近数值抖动
            if segment_progress <= self._snap_eps and np.linalg.norm(pos - start_wp) <= self._snap_eps:
                pos = start_wp.copy(); segment_progress = 0.0
            if (1.0 - segment_progress) <= self._snap_eps and np.linalg.norm(pos - end_wp) <= self._snap_eps:
                pos = end_wp.copy();   segment_progress = 1.0
     
            unit = seg_vec / seg_len

            if self._seg_dir > 0:
                # 正向：从 segment_progress → 1.0（start_wp → end_wp）
                path_left = (1.0 - segment_progress) * seg_len
                if path_left < self._snap_eps:
                    # 这一段已经走完，切下一段
                    self._seg_start_idx = end_idx
                    continue

                step_here = min(remaining_path_len, path_left)
                pos = pos + unit * step_here
                remaining_path_len -= step_here

                if abs(step_here - path_left) <= self._snap_eps:
                    pos = end_wp.copy()
                    self._seg_start_idx = end_idx
                    if self._stop_on_waypoint:
                        break
                    else:
                        continue

            else:
                # 反向：从 segment_progress → 0.0（往段起点退）
                path_left = segment_progress * seg_len  # 距离 start_wp 的弧长
                if path_left < self._snap_eps:
                    # 这一段往回已经没有路了，直接切到“上一段”
                    self._seg_start_idx = (start_idx - 1) % self._num_wp
                    continue

                step_here = min(remaining_path_len, path_left)
                pos = pos - unit * step_here  # 注意方向：减 unit 是往 start_wp 退
                remaining_path_len -= step_here

                if abs(step_here - path_left) <= self._snap_eps:
                    # 正好退回到 start_wp：下一步应走上一段
                    pos = start_wp.copy()
                    self._seg_start_idx = (start_idx - 1) % self._num_wp
                    if self._stop_on_waypoint:
                        break
                    else:
                        continue

            # （可选）如果仍需要单步限幅，就把上面这段放到 pos 更新之后再裁剪
            delta = pos - self._cmd_pos
            norm = float(np.linalg.norm(delta))
            max_move = max(1e-9, abs(a) * self._max_joint_delta)
            if norm > max_move:
                pos = self._cmd_pos + delta * (max_move / norm)
                break

        # 更新“指令侧”的位置状态，并裁剪到合法范围
        pos = np.clip(pos, self._lower, self._upper)
        self._cmd_pos = pos.copy()

        # 返回“指令目标”，不阻塞等待真实到位
        return pos
    
    def move_up(self):
        print("move up to avoid collision")
        pos = self.cur_position.copy()
        pos[2] += 0.1
        ori = self.cur_oritation.copy()
        nextpos = np.concatenate((pos, ori), axis=0)
        self.ros_interface.publish_arm_action(nextpos)
        time.sleep(2.0)
    

    def compute_reward(self, obs) -> bool:
        current_pose = obs["state"]
        # convert from quat to euler first
        current_rot = Rotation.from_quat(current_pose[3:7]).as_matrix()
        target_rot = Rotation.from_quat(self._TARGET_POSE[3:7]).as_matrix()
        # target_rot = Rotation.from_euler("xyz", self._TARGET_POSE[3:]).as_matrix()
        diff_rot = current_rot.T  @ target_rot
        diff_euler = Rotation.from_matrix(diff_rot).as_euler("xyz")
        delta = np.abs(np.hstack([current_pose[:3] - self._TARGET_POSE[:3], [0, 0, 0]]))
        if np.all(delta < self._REWARD_THRESHOLD):
            return True
        else:
            # print(f'Goal not reached, the difference is {delta}, the desired threshold is {self._REWARD_THRESHOLD}')
            return False
    

    def get_im(self) -> Dict[str, np.ndarray]:
        """Get images from the realsense cameras."""
        images = {}
        display_images = {}
        full_res_images = {}  # New dictionary to store full resolution cropped images
        for key, cap in self.cap.items():
            if key == "side_camera":
                continue
            try:
                frame = cap.read()
                if frame.ndim == 3 and frame.shape[2] == 4:
                    rgb = frame[..., :3]   # BGR 彩色
                else:
                    rgb = frame
                rgb = rgb.astype(np.uint8)
                # cropped_rgb = self.config.IMAGE_CROP[key](rgb) if key in self.config.IMAGE_CROP else rgb
                cropped_rgb = rgb #当前不需要裁剪
                resized = cv2.resize(
                    cropped_rgb, self.observation_space["images"][key].shape[:2][::-1]
                )
                images[key] = resized[..., ::-1]
                display_images[key] = resized
                display_images[key + "_full"] = cropped_rgb
                full_res_images[key] = copy.deepcopy(cropped_rgb)  # Store the full resolution cropped image
            except queue.Empty:
                input(
                    f"{key} camera frozen. Check connect, then press enter to relaunch..."
                )
                cap.close()
                self.init_cameras(self.config.REALSENSE_CAMERAS, self.config.EXTRA_REALSENSE_CAMERAS)
                return self.get_im()

        if not self.enable_tactile:
            if self.display_image:
                display_image = {
                    "front_camera": display_images["front_camera_full"],
                    "wrist_camera": display_images["wrist_camera_full"],
                }
        else:
            heat_map = cv2.hconcat([self.thumb_heat_map, self.index_heat_map])
            full_res_images["tactile_data"] = heat_map
            if self.display_image:
                with self.tac_index_lock:
                    display_image = {
                        "heat_map": heat_map,
                        "front_camera": display_images["front_camera_full"],
                        "wrist_camera": display_images["wrist_camera_full"],
                    }
        
        self.img_queue.put(display_image)
        if self.save_video:
            self.recording_frames.append(full_res_images)
        return images


    def get_rgb_and_dpth_im(self) -> Dict[str, np.ndarray]:
        """Get images from the realsense cameras."""
        images = {}
        depth_images = {}
        display_images = {}
        full_res_images = {}  # New dictionary to store full resolution cropped images
        for key, cap in self.cap.items():
            try:
                frame = cap.read()
                if frame.ndim == 3 and frame.shape[2] == 4:
                    rgb = frame[..., :3]   # BGR 彩色
                    depth = frame[..., 3]    # 深度
                else:
                    rgb = frame
                    depth = None
                cropped_rgb = rgb #当前不需要裁剪
                if key in self.observation_space["images"]:
                    resized = cv2.resize(
                        cropped_rgb, self.observation_space["images"][key].shape[:2][::-1]
                    )
                else:
                    resized = cv2.resize(
                        cropped_rgb, (640, 480)
                    )
                    
                images[key] = resized[..., ::-1]
                depth_images[key] = depth
                display_images[key] = resized
                display_images[key + "_full"] = cropped_rgb
                full_res_images[key] = copy.deepcopy(cropped_rgb)  # Store the full resolution cropped image
            except queue.Empty:
                input(
                    f"{key} camera frozen. Check connect, then press enter to relaunch..."
                )
                cap.close()
                self.init_cameras(self.config.REALSENSE_CAMERAS, self.config.EXTRA_REALSENSE_CAMERAS)
                return self.get_rgb_and_dpth_im()

        return images, depth_images


    def process_tactile_data(self, sensor, img_size):
        heat_map = []
        raw_img = []
        points = []
       
        raw_img = sensor.get_rectify_crop_image()
        img_GRAY = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
        height_map = sensor.raw_image_2_height_map(img_GRAY)
        height_map = sensor.expand_image(height_map)
        heat_map_input = cv2.normalize(height_map, None, 0, 255, cv2.NORM_MINMAX)
        heat_map_input = np.uint8(heat_map_input)
        heat_map = cv2.applyColorMap(heat_map_input, cv2.COLORMAP_JET)
        target_size = img_size
        heat_map = cv2.resize(heat_map, target_size, interpolation=cv2.INTER_LINEAR)
        points, gradients = sensor.height_map_2_point_cloud_gradients(height_map)

        return raw_img, points, heat_map
    

    def process_thumb_tactile(self):
        while True:
            # thumb_raw_img, thumb_points, thumb_heat_map = self.process_tactile_data(self.thumb_tactile_sensor, (320, 240))
            thumb_raw_img, thumb_points, thumb_heat_map = self.process_tactile_data(self.thumb_tactile_sensor, (128, 128))

            with self.tac_thumb_lock:
                self.thumb_raw_img = thumb_raw_img
                self.thumb_points = thumb_points
                self.thumb_heat_map = thumb_heat_map
            time.sleep(0.01)


    def process_index_tactile(self):
        # Process index tactile data
        while True:
            index_raw_img, index_points, index_heat_map = self.process_tactile_data(self.index_tactile_sensor, (128, 128))

            with self.tac_index_lock:
                self.index_raw_img = index_raw_img
                self.index_points = index_points
                self.index_heat_map = index_heat_map
            time.sleep(0.01)
    

    def process_middle_tactile(self):
        # Process middle tactile data
        while True:
            middle_raw_img, middle_points, middle_heat_map = self.process_tactile_data(self.middle_tactile_sensor, (128, 128))

            with self.tac_middle_lock:
                self.middle_raw_img = middle_raw_img
                self.middle_points = middle_points
                self.middle_heat_map = middle_heat_map
            time.sleep(0.01)
    

    def reset(self, joint_reset=False, **kwargs):
        print("densoenv reset")
        self.data_count = 0
        self.last_gripper_act = time.time()
        # requests.post(self.url + "update_param", json=self.config.COMPLIANCE_PARAM)
        if self.save_video:
            self.save_video_recording()

        # self._recover()
        # self.go_to_reset(joint_reset=joint_reset)
        # self._recover()

        self.curr_path_length = 0

        self._update_cur_position()
        obs = self._get_obs()
        self.terminate = False
        return obs, {"succeed": False}

    def save_video_recording(self):
        try:
            if len(self.recording_frames):
                if not os.path.exists('./videos'):
                    os.makedirs('./videos')
                
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                
                for camera_key in self.recording_frames[0].keys():
                    if self.url == "http://127.0.0.1:5000/":
                        video_path = f'./videos/left_{camera_key}_{timestamp}.mp4'
                    else:
                        video_path = f'./videos/right_{camera_key}_{timestamp}.mp4'
                    
                    # Get the shape of the first frame for this camera
                    first_frame = self.recording_frames[0][camera_key]
                    height, width = first_frame.shape[:2]
                    
                    video_writer = cv2.VideoWriter(
                        video_path,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        10,
                        (width, height),
                    )
                    
                    for frame_dict in self.recording_frames:
                        video_writer.write(frame_dict[camera_key])
                    
                    video_writer.release()
                    print(f"Saved video for camera {camera_key} at {video_path}")
                
            self.recording_frames.clear()
        except Exception as e:
            print(f"Failed to save video: {e}")

    def init_cameras(self, name_serial_dict=None, extra_cameras_dict=None):
        """Init both wrist cameras."""
        if self.cap is not None:  # close cameras if they are already open
            self.close_cameras()

        self.cap = OrderedDict()
        for cam_name, kwargs in name_serial_dict.items():
            cap = VideoCapture(
                RSCapture(name=cam_name, **kwargs)
            )
            self.cap[cam_name] = cap

        for cam_name, kwargs in extra_cameras_dict.items():
            cap = VideoCapture(
                RSCapture(name=cam_name, **kwargs)
            )
            self.cap[cam_name] = cap

    def close_cameras(self):
        """Close both wrist cameras."""
        try:
            for cap in self.cap.values():
                cap.close()
        except Exception as e:
            print(f"Failed to close cameras: {e}")

    def _recover(self):
        """Internal function to recover the robot from error state."""
        requests.post(self.url + "clearerr")


    def _send_leap_hand_command(self, leap_hand_action: np.ndarray, steps=10, step_time=0.01):
        """Internal function to send leap hand command to the robot."""
        hand_action = leap_hand_action
        # step_time = 0.01  # Example step time
        # steps = 10    # Example number of steps

        if self.interpolation_thread and self.interpolation_thread.is_alive():
            return
        
        current_pos = self.ros_interface.get_current_leap_position()
        self.curr_leap_hand_pos = np.asarray(current_pos, dtype=np.float32).copy()
        # print("in send_leap_hand_command curr_leap_hand_pos = ", self.curr_leap_hand_pos)

        with self.thread_lock:
            self.interpolation_thread = threading.Thread(
                target=self.leap_interpolate_and_publish,
                args=(self.curr_leap_hand_pos, hand_action, step_time, steps),
                daemon=True
            )
            self.interpolation_thread.start()


    def leap_interpolate_and_publish(self, start_position, end_position, step_time, steps):
        for i in range(steps + 1):
            # Interpolate between start and end positions
            interpolated_position = [
                start + (end - start) * (i / steps)
                for start, end in zip(start_position, end_position)
            ]
            # Publish the interpolated position
            self.curr_leap_hand_pos = np.asarray(interpolated_position, dtype=np.float32).copy()
            self.ros_interface.publish_hand_action(interpolated_position)
            # print("in leap_interpolate_and_publish curr_leap_hand_pos = ", self.curr_leap_hand_pos)
            time.sleep(step_time)
            

    def _update_cur_position(self, arm_action, timeout=10.0, wait_threshold=0.05):
        """
        Internal function to get the latest state of the robot and its gripper.
        """
        start = time.time()
        self.cur_position, self.cur_oritation = self.ros_interface.get_current_robot_ee()
        joint_position = self.ros_interface.get_current_joint()
        self.joint_position = np.asarray(joint_position, dtype=np.float32).copy()

        hand_joint_msg = self.ros_interface.get_current_leap_position()
        self.curr_leap_hand_pos = np.asarray(hand_joint_msg, dtype=np.float32).copy()
        # print("in _update_cur_position curr_leap_hand_pos = ", self.curr_leap_hand_pos)

        diff = np.asarray(arm_action[:3], dtype=np.float32) - np.asarray(self.cur_position, dtype=np.float32)
        while np.max(np.abs(diff)) > wait_threshold:
            if time.time() - start > timeout:
                print("[WARN] 等待机械臂到位超时")
                break
            time.sleep(0.02)
            self.cur_position, self.cur_oritation = self.ros_interface.get_current_robot_ee()
            joint_position = self.ros_interface.get_current_joint()
            self.joint_position = np.asarray(joint_position, dtype=np.float32).copy()
            hand_joint_msg = self.ros_interface.get_current_leap_position()
            self.curr_leap_hand_pos = np.asarray(hand_joint_msg, dtype=np.float32).copy()
            diff = np.linalg.norm(arm_action[:3] - self.cur_position)


    def _get_obs(self) -> dict:
        images = self.get_im()
        front_camera_image = images["front_camera"]
        wrist_camera_image = images["wrist_camera"]
        # side_camera_image = images["side_camera"]
        state_flattened = np.concatenate([
            np.array(self.cur_position, dtype=np.float32).flatten(),
            np.array(self.cur_oritation, dtype=np.float32).flatten(),
            np.array(self.hand_state, dtype=np.float32).flatten(), 
        ])
        
        if not self.enable_tactile:
            obs = copy.deepcopy({
                "front_camera": front_camera_image,
                "wrist_camera": wrist_camera_image,
                "state": state_flattened
            })
        else:
            heatmap_canvas = cv2.hconcat([self.thumb_heat_map, self.index_heat_map])
            obs = copy.deepcopy({
                "front_camera": front_camera_image,
                "wrist_camera": wrist_camera_image,
                "tactile_data":heatmap_canvas,
                "state": state_flattened
            })
                
        return copy.deepcopy(obs)


    def close(self):
        if hasattr(self, 'listener'):
            self.listener.stop()
        self.close_cameras()
        if self.display_image:
            self.img_queue.put(None)
            cv2.destroyAllWindows()
            self.displayer.join()

        # Ensure ROS executor and node are cleaned up to avoid threading errors
        if hasattr(self, "executor"):
            if hasattr(self, "ros_interface"):
                # Remove the node from the executor and destroy it before shutting down ROS
                self.executor.remove_node(self.ros_interface)
                self.ros_interface.destroy_node()
            try:
                if getattr(rclpy, "is_initialized", lambda: True)():
                    # Signal the spinning thread to exit
                    rclpy.shutdown()
            except Exception:
                pass
            if hasattr(self, "executor_thread"):
                # Wait for the executor thread to finish before closing it
                self.executor_thread.join()
            self.executor.shutdown()


    def pose_callback(self, msg):
        self.arm_position = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        self.arm_orientation = np.array([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
        # self.arm_pose = np.concatenate((position, orientation), axis=0)


    def save_training_frame(self):
        try:
            joint_pose = np.concatenate([
                    self.joint_position, self.curr_leap_hand_pos], dtype=np.float32)
            
            self.joint_buffer.append(
                            copy.deepcopy(joint_pose))
            self.hand_state_buffer.append(copy.deepcopy(self.hand_state))
            # 保存图像
            images, depth_img = self.get_rgb_and_dpth_im()
            for cam_name, img in images.items():
                depth = depth_img[cam_name]
                if cam_name == "front_camera":
                    self.front_color_buffer.append(copy.deepcopy(img[..., ::-1]))
                    self.front_depth_buffer.append(copy.deepcopy(depth))
                elif cam_name == "side_camera":
                    self.side_color_buffer.append(copy.deepcopy(img[..., ::-1]))
                    self.side_depth_buffer.append(copy.deepcopy(depth))
                elif cam_name == "wrist_camera":
                    self.wrist_color_buffer.append(copy.deepcopy(img[..., ::-1]))
                    self.wrist_depth_buffer.append(copy.deepcopy(depth))

            if self.enable_tactile:
                self.rthumb_raw_buffer.append(copy.deepcopy(self.thumb_raw_img))
                self.rindex_raw_buffer.append(copy.deepcopy(self.index_raw_img))
                self.rmiddle_raw_buffer.append(copy.deepcopy(self.middle_raw_img))

                self.rthumb_heatmap_buffer.append(copy.deepcopy(self.thumb_heat_map))
                self.rindex_heatmap_buffer.append(copy.deepcopy(self.index_heat_map))
                self.rmiddle_heatmap_buffer.append(copy.deepcopy(self.middle_heat_map))

        except Exception as e:
            print("An error occurred while processing frames")
            print(e)


    def save_all_data_on_exit(self):
        for frame_id in range(self.frame_count):
            print("save frame ", frame_id)
            frame_dir = os.path.join(self.frame_save_path, f"frame_{frame_id}")
            os.makedirs(frame_dir, exist_ok=True)

            cv2.imwrite(os.path.join(frame_dir, "color_image.jpg"), self.front_color_buffer[frame_id])
            cv2.imwrite(os.path.join(frame_dir, "depth_image.png"), self.front_depth_buffer[frame_id])
            cv2.imwrite(os.path.join(frame_dir, "color_image2.jpg"), self.side_color_buffer[frame_id])
            cv2.imwrite(os.path.join(frame_dir, "depth_image2.png"), self.side_depth_buffer[frame_id])
            cv2.imwrite(os.path.join(frame_dir, "color_image3.jpg"), self.wrist_color_buffer[frame_id])
            cv2.imwrite(os.path.join(frame_dir, "depth_image3.png"), self.wrist_depth_buffer[frame_id])

            if self.enable_tactile:
                cv2.imwrite(os.path.join(frame_dir, "thumb_raw_image.jpg"), self.rthumb_raw_buffer[frame_id])
                cv2.imwrite(os.path.join(frame_dir, "thumb_heat_map.jpg"), self.rthumb_heatmap_buffer[frame_id])
            
                cv2.imwrite(os.path.join(frame_dir, "index_raw_image.jpg"), self.rindex_raw_buffer[frame_id])
                cv2.imwrite(os.path.join(frame_dir, "index_heat_map.jpg"), self.rindex_heatmap_buffer[frame_id])
            
                cv2.imwrite(os.path.join(frame_dir, "middle_raw_image.jpg"), self.rmiddle_raw_buffer[frame_id])
                cv2.imwrite(os.path.join(frame_dir, "middle_heat_map.jpg"), self.rmiddle_heatmap_buffer[frame_id])
            # 保存 state（TCP + orientation + hand joints）
            np.savetxt(os.path.join(frame_dir, "right_arm_joint.txt"), self.joint_buffer[frame_id])
            np.savetxt(os.path.join(frame_dir, "hand_state.txt"), np.atleast_1d(self.hand_state_buffer[frame_id]), fmt="%.6f")