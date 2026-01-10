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
                # img = cv2.resize(v, (320, 240))  # 每个窗口显示 320×240
                cv2.imshow(f"{self.name} - {k}", v)

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
    
    def get_current_robot_ee(self, timeout=5.0):
        return self.cur_position, self.cur_oritation
    

    def get_current_joint(self, timeout=5.0):
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

        self.last_gripper_act = time.time()
        self.lastsent = time.time()
        self.hz = hz
        self.grip_loop = self.config.LOOP_CONTROL if hasattr(self.config, 'LOOP_CONTROL') else False
        self.current_action = np.zeros(7, dtype=np.float32)

        self.save_video = save_video
        if self.save_video:
            print("Saving videos!")
            self.recording_frames = []

        # Action/Observation Space
        low  = np.array([-1, -1, -1, -1, -1, -1, -1], dtype=np.float32)
        high = np.array([1, 1, 1, 1, 1, 1, 1], dtype=np.float32)
        self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        
        state_dict = {
            "tcp_pos": gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
            "tcp_ori": gym.spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32),
            "gripper_pose": gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
        }

        # 只有非 twist 任务才带 hand_state
        # if not self.grip_loop:
        #     state_dict["gripper_pose"] = gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32)
        
        if not self.enable_tactile:
            print("init arm without tactaile data")

            self.observation_space = gym.spaces.Dict(
                {
                    "state": gym.spaces.Dict(state_dict),
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
                    "state": gym.spaces.Dict(state_dict),
                    "images": gym.spaces.Dict(
                        {
                            **{key: gym.spaces.Box(0, 255, shape=(128, 128, 3), dtype=np.uint8) 
                                    for key in config.REALSENSE_CAMERAS},
                            "tactile_data": gym.spaces.Box(0, 255, shape=(64, 128, 3), dtype=np.uint8),
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
        self.action_buffer = []
        self.grip_action_buffer = []
        
        
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

            self.tactile_size = (64, 64)
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

        # Spin ROS callbacks in a background thread and keep references so we can
        # shut everything down cleanly when the environment closes.
        rclpy.init(args=None)

        self.ros_interface = ROSNodeInterface()

        executor = rclpy.executors.MultiThreadedExecutor()
        executor.add_node(self.ros_interface)
        executor_thread = threading.Thread(target=executor.spin, daemon=True)
        executor_thread.start()

        self.interpolation_thread = None
        self.thread_lock = threading.Lock()

        self.frame_save_path = "/home/wrq/workspaces/HK_TACEXO_WANG/recorded_data/recorded_data_training-1-7-2"  # 可自行修改
        os.makedirs(self.frame_save_path, exist_ok=True)
        self.frame_count = 0
        self.video_count = 0

        self.cur_position = np.zeros(3, dtype=np.float32)
        self.cur_oritation = np.zeros(4, dtype=np.float32)

        if self.exp_name == "tennis_ball_pick" or self.exp_name == "tennis_ball_place":
        # grip with index
            self.gripper_close_joint = config.GRIPPER_CLOSE_JOINT
            self.gripper_open_joint = config.GRIPPER_OPEN_JOINT

        elif self.exp_name == "twist_bottle_cap" or self.exp_name == "tube_insertion":
            self.gripper_close_joint = config.GRIPPER_CLOSE_JOINT
            self.gripper_twist_joint = config.GRIPPER_TWIST_JOINT
            self.gripper_open_joint = config.GRIPPER_OPEN_JOINT

        self.curr_leap_hand_pos = list(self.gripper_open_joint)

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
        self.current_action = action.copy()

        self.nextpos = np.concatenate((self.cur_position, self.cur_oritation), axis=0)
        
        xyz_delta = action[:3]

        # print("action scaled = ", xyz_delta * self.action_scale[0])
        if self.exp_name == "tennis_ball_pick":
            if self.nextpos[2] < 0.03:
                self.nextpos[2] = 0.03
        elif self.exp_name == "tube_insertion":
            if self.nextpos[2] < 0.05:
                self.nextpos[2] = 0.05
            if self.nextpos[1] < -0.145:
                self.nextpos[1] = -0.145
        elif self.exp_name == "twist_bottle_cap":
            if self.nextpos[2] < 0.22 and (0.6 < self.nextpos[0] < 0.8) and (-0.13 < self.nextpos[1] < -0.05):
                action[:3] = np.clip(xyz_delta, -0.4, 0.4)
                
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

        if self.exp_name == "twist_bottle_cap" or self.exp_name == "tube_insertion":
            target_hand_pos = self.calculate_hand_pos_segmented(grip_action, current_hand_pos)
        elif self.exp_name == "tennis_ball_pick" or self.exp_name == "tennis_ball_place":
            target_hand_pos = self._cal_hand_close_open(grip_action, current_hand_pos)

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

        # t_end = time.time()
        # print(f"[update_position End] {t_end:.6f}, Step总耗时（含sleep）: {t_end - start_time:.4f}s, 实际频率: {1.0/(t_end - start_time):.2f}Hz")
        # print("after publish arm action cur_position = ", self.cur_position)
        self.frame_count += 1

        ob = self._get_obs()
        reward = self.compute_reward(ob)
        # self.save_training_frame()
        # print(f"reward in denso_env = {reward}")
        # done = self.curr_path_length >= self.max_episode_length or reward or self.terminate
        done = reward or self.terminate
        # t_end = time.time()
        # print(f"[Step End] {t_end:.6f}, Step总耗时（含sleep）: {t_end - start_time:.4f}s, 实际频率: {1.0/(t_end - start_time):.2f}Hz")
        # print("curr hand pos = ", self.curr_leap_hand_pos)
        # input("debug for hand pos")
        return ob, int(reward), done, False, {"succeed": reward}
    
    def _close_open_pose_init(self, current_hand_pos: np.ndarray):
        self._lower = np.minimum(self.gripper_open_joint, self.gripper_close_joint)
        self._upper = np.maximum(self.gripper_open_joint, self.gripper_close_joint)

        self._cmd_pos = np.asarray(current_hand_pos, dtype=np.float32)

        diff = self.gripper_close_joint - self.gripper_open_joint
        max_step = 10
        self.max_joint_delta = float(np.max(np.abs(diff))) / max_step

        self.is_close_open_pose_init = True
        

    def _cal_hand_close_open(self, grip_action: float, current_hand_pos: np.ndarray) -> np.ndarray:
        if not hasattr(self, "is_close_open_pose_init") or not self.is_close_open_pose_init:
            self._close_open_pose_init(current_hand_pos)

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

        new_pos = np.clip(new_pos, self._lower, self._upper)
        self._cmd_pos = new_pos.copy()
        return new_pos
    

    def _segmented_init(self, current_hand_pos: np.ndarray):
        self._waypoints = np.stack(
            [self.gripper_open_joint, self.gripper_close_joint, self.gripper_twist_joint],
            axis=0
        ).astype(np.float32)
        self._num_wp = int(self._waypoints.shape[0]) 
        self._loop = bool(self.grip_loop)
        self._num_seg = self._num_wp if self._loop else (self._num_wp - 1)
        # self._num_seg = self._num_wp - 1
        

        def segment_vectors() -> np.ndarray:
            wp = self._waypoints
            if self._loop:
                return wp[(np.arange(self._num_wp) + 1) % self._num_wp] - wp
            else:
                return wp[1:] - wp[:-1]
            
        seg_vecs = segment_vectors()  
        # seg_vecs = self._waypoints[1:] - self._waypoints[:-1]          # (2, D)
        seg_lens = np.linalg.norm(seg_vecs, axis=1) + 1e-12            # (2,)

        # joint bounds
        self._lower = np.minimum.reduce(self._waypoints)
        self._upper = np.maximum.reduce(self._waypoints)

        # choose nearest waypoint to initialize which segment we are on
        diffs_wp = self._waypoints - np.asarray(current_hand_pos, dtype=np.float32)
        nearest_wp = int(np.argmin(np.linalg.norm(diffs_wp, axis=1)))

        # segment start index must be 0 or 1
        if self._loop:
            self._seg_start_idx = nearest_wp
        else:
            self._seg_start_idx = max(0, min(self._num_wp - 2, nearest_wp))
            
            
        # if nearest_wp <= 0:
        #     self._seg_start_idx = 0
        # elif nearest_wp >= self._num_wp - 1:
        #     self._seg_start_idx = self._num_wp - 2
        # else:
        #     self._seg_start_idx = 0

        max_step = 10
        mean_len = float(np.mean(seg_lens))
        self._per_step_path_len = max(1e-6, mean_len / max_step)
        self._max_joint_delta = float(np.max(np.abs(seg_vecs))) / max_step
        self._snap_eps = 1e-6
        self.is_segmented_init = True
        self._cmd_pos = np.asarray(current_hand_pos, dtype=np.float32)
    
    
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
        action_scalar = float(np.clip(grip_action, -1.0, 1.0))
        if abs(action_scalar) < 1e-6:
            # 不推进，直接维持上次指令
            return np.asarray(self._cmd_pos, dtype=np.float32)

        # 决定方向
        seg_dir = (+1 if action_scalar > 0 else -1)
        waypoints = self._waypoints
        num_wp = self._num_wp
        num_seg = num_wp - 1
               
        # 用“指令位置”推进（不受真实到位与否的影响）
        pos = np.asarray(self._cmd_pos, dtype=np.float32)
        remaining_path_len = abs(action_scalar) * self._per_step_path_len
        
        def seg_end_idx(start_idx: int) -> int:
            if self._loop:
                return (start_idx + 1) % self._num_wp
            else:
                return start_idx + 1

        while remaining_path_len > 0:
            start_idx = int(self._seg_start_idx)

            # clamp segment index into valid range [0, num_seg-1]
            if not self._loop:
                start_idx = max(0, min(self._num_seg - 1, start_idx))
                self._seg_start_idx = start_idx
            else:
                start_idx = start_idx % self._num_seg
                self._seg_start_idx = start_idx

            end_idx = seg_end_idx(start_idx)

            seg_start_wp = waypoints[start_idx]
            seg_end_wp   = waypoints[end_idx]
            seg_vec = seg_end_wp - seg_start_wp
            seg_len = float(np.linalg.norm(seg_vec)) + 1e-12
            unit = seg_vec / seg_len

            segment_progress, proj = self._project_to_current_segment(pos, seg_start_wp, seg_vec)

            # 吸附端点，避免在端点附近数值抖动
            if segment_progress <= self._snap_eps and np.linalg.norm(pos - seg_start_wp) <= self._snap_eps:
                pos = seg_start_wp.copy()
                segment_progress = 0.0
            if (1.0 - segment_progress) <= self._snap_eps and np.linalg.norm(pos - seg_end_wp) <= self._snap_eps:
                pos = seg_end_wp.copy()
                segment_progress = 1.0

            unit = seg_vec / seg_len

            if seg_dir > 0:
                path_left = (1.0 - segment_progress) * seg_len
                if path_left <= self._snap_eps:
                    if self._loop:
                        self._seg_start_idx = (start_idx + 1) % self._num_seg
                        pos = seg_end_wp.copy()
                        continue
                    else:
                        if start_idx < num_seg - 1:
                            self._seg_start_idx = start_idx + 1
                            pos = seg_end_wp.copy()
                            continue
                        else:
                            # at final segment end (twist) -> saturate
                            pos = waypoints[-1].copy()
                            remaining_path_len = 0.0
                            break

                step_here = min(remaining_path_len, path_left)
                pos = pos + unit * step_here
                remaining_path_len -= step_here

                # if reached end of segment
                if abs(step_here - path_left) <= self._snap_eps:
                    pos = seg_end_wp.copy()
                    if self._loop:
                        self._seg_start_idx = (start_idx + 1) % self._num_seg
                        continue
                    else:
                        if start_idx < num_seg - 1:
                            self._seg_start_idx = start_idx + 1
                            continue
                        else:
                            # reached twist
                            remaining_path_len = 0.0
                            break

            else:
                path_left = segment_progress * seg_len
                if path_left <= self._snap_eps:
                    if self._loop:
                        self._seg_start_idx = (start_idx - 1) % self._num_seg
                        pos = seg_start_wp.copy()
                        continue
                    else:
                        if start_idx > 0:
                            self._seg_start_idx = start_idx - 1
                            pos = seg_start_wp.copy()
                            continue
                        else:
                            # at first segment start (open) -> saturate
                            pos = waypoints[0].copy()
                            remaining_path_len = 0.0
                            break

                step_here = min(remaining_path_len, path_left)
                pos = pos - unit * step_here
                remaining_path_len -= step_here

                # if reached start of segment
                if abs(step_here - path_left) <= self._snap_eps:
                    pos = seg_start_wp.copy()
                    if self._loop:
                        self._seg_start_idx = (start_idx - 1) % self._num_seg
                        continue
                    else:
                        if start_idx > 0:
                            self._seg_start_idx = start_idx - 1
                            continue
                        else:
                            remaining_path_len = 0.0
                            break

            # （可选）如果仍需要单步限幅，就把上面这段放到 pos 更新之后再裁剪
            delta = pos - self._cmd_pos
            norm = float(np.linalg.norm(delta))
            max_move = max(1e-9, abs(action_scalar) * self._max_joint_delta)
            if norm > max_move:
                pos = self._cmd_pos + delta * (max_move / norm)
                break

        # 更新“指令侧”的位置状态，并裁剪到合法范围
        pos = np.clip(pos, self._lower, self._upper)
        self._cmd_pos = pos.copy()

        return pos
    
    
    def _hand_progress_scalar(self, hand_joint_pos: np.ndarray) -> float:
        """
        Open polyline progress in [0,1]:
            open -> close -> twist
        """
        hand_joint_pos = np.asarray(hand_joint_pos, dtype=np.float32)

        # If waypoints not prepared, fallback to open<->close
        if not hasattr(self, "_waypoints"):
            open_joint = np.asarray(self.gripper_open_joint, dtype=np.float32)
            close_joint = np.asarray(self.gripper_close_joint, dtype=np.float32)
            seg_vec = close_joint - open_joint
            seg_len_sq = float(np.dot(seg_vec, seg_vec)) + 1e-12
            ratio = float(np.dot(hand_joint_pos - open_joint, seg_vec) / seg_len_sq)
            return float(np.clip(ratio, 0.0, 1.0))

        waypoints = np.asarray(self._waypoints, dtype=np.float32)  # (3, D)
        num_wp = int(waypoints.shape[0])                           # 3
        num_seg = num_wp - 1                                       # 2

        # segments: 0->1, 1->2
        segment_vectors = waypoints[1:] - waypoints[:-1]           # (2, D)
        segment_lengths = np.linalg.norm(segment_vectors, axis=1) + 1e-12  # (2,)

        prefix_lengths = np.concatenate([[0.0], np.cumsum(segment_lengths)])  # (3,)
        total_length = float(prefix_lengths[-1]) + 1e-12

        closest_seg = 0
        closest_ratio = 0.0
        min_dist_sq = float("inf")

        for seg_idx in range(num_seg):
            seg_start = waypoints[seg_idx]
            seg_vec = segment_vectors[seg_idx]
            seg_len_sq = float(np.dot(seg_vec, seg_vec)) + 1e-12

            ratio = float(np.dot(hand_joint_pos - seg_start, seg_vec) / seg_len_sq)
            ratio = float(np.clip(ratio, 0.0, 1.0))

            proj = seg_start + ratio * seg_vec
            dist_sq = float(np.sum((hand_joint_pos - proj) ** 2))

            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest_seg = seg_idx
                closest_ratio = ratio

        arc_length = float(prefix_lengths[closest_seg]) + float(closest_ratio) * float(segment_lengths[closest_seg])
        progress = float(np.clip(arc_length / total_length, 0.0, 1.0))
        return progress
    

    def compute_reward(self, obs) -> bool:
        state = obs["state"]
        pos = np.asarray(state["tcp_pos"], dtype=np.float32).reshape(-1)   # (3,)
        ori = np.asarray(state["tcp_ori"], dtype=np.float32).reshape(-1)
        # convert from quat to euler first
        current_rot = Rotation.from_quat(ori).as_matrix()
        target_rot = Rotation.from_quat(self._TARGET_POSE[3:7]).as_matrix()
        # target_rot = Rotation.from_euler("xyz", self._TARGET_POSE[3:]).as_matrix()
        diff_rot = current_rot.T  @ target_rot
        diff_euler = Rotation.from_matrix(diff_rot).as_euler("xyz")
        delta = np.abs(np.hstack([pos - self._TARGET_POSE[:3], [0, 0, 0]]))
        if np.all(delta < self._REWARD_THRESHOLD):
            return True
        else:
            # print(f'Goal not reached, the difference is {delta}, the desired threshold is {self._REWARD_THRESHOLD}')
            return False
    

    def get_im(self) -> Dict[str, np.ndarray]:
        """Get images from the realsense cameras."""
        images = {}
        display_images = {}
        full_res_images = {}
        video_images = {}
        for key, cap in self.cap.items():
            if key == "front_camera_2":
                continue
            try:
                frame = cap.read()
                if frame.ndim == 3 and frame.shape[2] == 4:
                    rgb = frame[..., :3] # 这里是bgr格式，cv2.imshow输入bgr,显示rgb图像
                else:
                    rgb = frame
                rgb = rgb.astype(np.uint8)
                video_images[key] = rgb
                cropped_rgb = self.config.IMAGE_CROP[key](rgb) if key in self.config.IMAGE_CROP else rgb
                # cropped_rgb = rgb #当前不需要裁剪
                resized = cv2.resize(
                    cropped_rgb, self.observation_space["images"][key].shape[:2][::-1]
                )
                images[key] = resized[..., ::-1]
                # display_images[key] = resized
                display_images[key + "_full"] = cropped_rgb
                full_res_images[key] = copy.deepcopy(cropped_rgb)  # Store the full resolution cropped image
            except queue.Empty:
                input(
                    f"{key} camera frozen. Check connect, then press enter to relaunch..."
                )
                cap.close()
                self.init_cameras(self.config.REALSENSE_CAMERAS, self.config.EXTRA_REALSENSE_CAMERAS)
                return self.get_im()
        # if not self.enable_tactile:
        #     if self.display_image:
        #         display_image = {
        #             "front_camera": display_images["front_camera_full"],
        #             "wrist_camera": display_images["wrist_camera_full"],
        #         }
        if self.enable_tactile:
            heat_map = cv2.hconcat([self.thumb_heat_map, self.index_heat_map])
            full_res_images["tactile_data"] = heat_map
            video_images["tactile_data"] = heat_map
            with self.tac_index_lock:
                display_images["heat_map"] = heat_map
        if self.display_image:
            self.img_queue.put(display_images)
        if self.save_video:
            self.recording_frames.append(video_images)
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
            thumb_raw_img, thumb_points, thumb_heat_map = self.process_tactile_data(self.thumb_tactile_sensor, self.tactile_size)

            with self.tac_thumb_lock:
                self.thumb_raw_img = thumb_raw_img
                self.thumb_points = thumb_points
                self.thumb_heat_map = thumb_heat_map
            time.sleep(0.01)


    def process_index_tactile(self):
        # Process index tactile data
        while True:
            index_raw_img, index_points, index_heat_map = self.process_tactile_data(self.index_tactile_sensor, self.tactile_size)

            with self.tac_index_lock:
                self.index_raw_img = index_raw_img
                self.index_points = index_points
                self.index_heat_map = index_heat_map
            time.sleep(0.01)
    

    def process_middle_tactile(self):
        # Process middle tactile data
        while True:
            middle_raw_img, middle_points, middle_heat_map = self.process_tactile_data(self.middle_tactile_sensor, self.tactile_size)

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

    def save_video_recording(self, count):
        try:
            if len(self.recording_frames):
                if not os.path.exists('./videos'):
                    os.makedirs('./videos')
                
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                
                for camera_key in self.recording_frames[0].keys():
                    if self.url == "http://127.0.0.1:5000/":
                        video_path = f'./videos/{count}/left_{camera_key}_{timestamp}.mp4'
                    else:
                        video_path = f'./videos/{count}/right_{camera_key}_{timestamp}.mp4'
                        
                    video_dir = os.path.dirname(video_path)
                    if not os.path.exists(video_dir):
                        os.makedirs(video_dir, exist_ok=True)

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
        self.hand_state = self._hand_progress_scalar(self.curr_leap_hand_pos)


    def _get_obs(self) -> dict:
        images = self.get_im()
        # if self.grip_loop:
        #     state_observation = {
        #         "tcp_pos": self.cur_position,
        #         "tcp_ori": self.cur_oritation,
        #     }
        # else:
        state_observation = {
            "tcp_pos": self.cur_position,
            "tcp_ori": self.cur_oritation,
            "gripper_pose": self.hand_state,
        }
        
        obs = {
            "images": {},
            "state": state_observation,
        }
        for cam_key, img in images.items():
            obs["images"][cam_key] = img
            
        if self.enable_tactile:
            heatmap_canvas = cv2.hconcat([self.thumb_heat_map, self.index_heat_map])
            obs["images"]["tactile_data"] = heatmap_canvas
                
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
        if hasattr(self, "_ros_spin_stop"):
            self._ros_spin_stop.set()

        # 2) 等 spin thread 退出来（先 join）
        if hasattr(self, "executor_thread") and self.executor_thread.is_alive():
            self.executor_thread.join(timeout=2)
            print("executor_thread alive?", self.executor_thread.is_alive(), flush=True)

        # 3) 再 shutdown executor + node + rclpy
        if hasattr(self, "executor"):
            self.executor.shutdown()

        if hasattr(self, "ros_interface"):
            self.ros_interface.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()

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
            self.action_buffer.append(copy.deepcopy(self.current_action))
            # 保存图像
            images, depth_img = self.get_rgb_and_dpth_im()
            for cam_name, img in images.items():
                depth = depth_img[cam_name]
                if cam_name == "front_camera":
                    self.front_color_buffer.append(copy.deepcopy(img[..., ::-1]))
                    self.front_depth_buffer.append(copy.deepcopy(depth))
                elif cam_name == "front_camera_2":
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

            if len(self.front_color_buffer) > frame_id:
                cv2.imwrite(os.path.join(frame_dir, "color_image.jpg"), self.front_color_buffer[frame_id])
                cv2.imwrite(os.path.join(frame_dir, "depth_image.png"), self.front_depth_buffer[frame_id])
            if len(self.side_color_buffer) > frame_id:
                cv2.imwrite(os.path.join(frame_dir, "color_image2.jpg"), self.side_color_buffer[frame_id])
                cv2.imwrite(os.path.join(frame_dir, "depth_image2.png"), self.side_depth_buffer[frame_id])
            if len(self.wrist_color_buffer) > frame_id:
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
            np.savetxt(os.path.join(frame_dir, "action.txt"), np.atleast_1d(self.action_buffer[frame_id]), fmt="%.6f")