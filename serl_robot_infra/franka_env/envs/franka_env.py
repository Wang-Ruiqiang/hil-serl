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

from datetime import datetime
from collections import OrderedDict
from typing import Dict, List, Tuple

from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Rotation as R, Slerp

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
import threading
from franka_env.camera.video_capture import VideoCapture
from franka_env.camera.rs_capture import RSCapture
try:
    from leap_hand.srv import LeapPosition, LeapPosVelEff
except ImportError:
    LeapPosition = None
    LeapPosVelEff = None
try:
    from dmrobotics import Sensor as DMTacSensor
except ImportError:
    DMTacSensor = None
from franka_env.gaze.display_markers import draw_gaze_display_markers, marker_points_for_size
from franka_env.recording.episode_recorder import EpisodeDataRecorder


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
                img = cv2.resize(v, (640, 480))  # 每个窗口显示 320×240
                # img = cv2.resize(v, (1280, 960))  # 每个窗口显示 320×240
                cv2.imshow(f"{self.name} - {k}", img)

            cv2.waitKey(1)


def _timing_enabled():
    return os.environ.get("HIL_TIMING", "0") == "1"


def _timing_log(label, start):
    if _timing_enabled():
        print(f"[timing][franka_env] {label}: {time.time() - start:.4f}s")


##############################################################################

class DefaultEnvConfig:
    """Default configuration for FrankaEnv. Fill in the values below."""

    REALSENSE_CAMERAS: Dict = {
        "front_camera": "151422254571",
    }
    IMAGE_CROP: dict[str, callable] = {}
    TARGET_POSE: np.ndarray = np.zeros((7,))
    REWARD_THRESHOLD: np.ndarray = np.zeros((6,))
    ACTION_SCALE = np.zeros((3,))
    CMD_POSE_RESYNC_THRESHOLD: float = 0.05
    DISPLAY_IMAGE: bool = True
    MAX_EPISODE_LENGTH: int = 200
    ENABLE_DATA_RECORDING: bool = False
    ENABLE_GAZE_COLLECTION: bool = False
    GAZE_FRAME_SAVE_PATH: str = "./recorded_gaze_data"
    PUPIL_HOST: str = "127.0.0.1"
    PUPIL_PORT: int = 50020
    GAZE_DISPLAY_MARKERS: bool = True
    GAZE_RS_SAVE_WIDTH: int = 640
    GAZE_RS_SAVE_HEIGHT: int = 480
    DM_TAC_THUMB_SERIAL_ID = "2501130504"
    DM_TAC_INDEX_SERIAL_ID = "2501130556"
    DM_TAC_MIDDLE_SERIAL_ID = "2501130530"
    ENABLE_DM_TAC_MIDDLE = False
    DM_TAC_DEPTH_SCALE: float = 0.25


##############################################################################

class ROSNodeInterface(Node):
    def __init__(self):
        if LeapPosition is None:
            raise ImportError(
                "leap_hand.srv is required for the Franka actor. "
                "Source/build the ROS2 workspace that provides leap_hand messages."
            )
        super().__init__('franka_env_node')

        # Publishers（发送）
        self.arm_pub = self.create_publisher(
            PoseStamped,
            '/target_pose',
            10
        )

        self.publisher_hand = self.create_publisher(
            JointState, 
            '/cmd_leap', 
            10
        )

        #franka
        self.robot_ee_sub = self.create_subscription(
            PoseStamped,
            '/franka_robot_state_broadcaster/current_pose',
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
        self.cur_orientation = np.array([0, 1, 0, 0], dtype=np.float32)


    def robot_ee_callback(self, msg):
        #用ros2 INFO打印接收到的数据 
        position = msg.pose.position
        # self.get_logger().info(f"robot_ee_received:{position}")
        self.cur_position = np.array([position.x, position.y, position.z])

        # 从 msg 中提取四元数方向数据（xyzw）
        orientation = msg.pose.orientation

        # self.get_logger().info(f"cur_orientation:{self.cur_orientation}")
        self.cur_orientation = np.array([
            orientation.w, orientation.x, orientation.y, orientation.z
        ])
        # 设置事件为已收到数据
        # self.robot_ee_event.set()


    def joint_callback(self, msg):

        # 将 joint name 和对应的位置打包为字典
        joint_dict = {name: pos for name, pos in zip(msg.name, msg.position)}

        # 按照你需要的顺序提取关节角：joint1 ~ joint7
        #franka
        ordered_joint_names = ["fr3_joint1", "fr3_joint2", "fr3_joint3", "fr3_joint4", "fr3_joint5", "fr3_joint6", "fr3_joint7"]
        ordered_joint_positions = [joint_dict.get(joint, 0.0) for joint in ordered_joint_names]

        # 保存为 numpy array
        self.joint_position = np.array(ordered_joint_positions, dtype=np.float32)

        # 设置事件为“数据已接收”
        # self.joint_event.set()


    def publish_arm_action(self, pose):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        x, y, z = map(float, pose[:3])
        msg.header.frame_id = "base"
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = z
        ori_w, ori_x, ori_y, ori_z = map(float, pose[3:7])
        # msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = pose[:3]
        msg.pose.orientation.w = ori_w
        msg.pose.orientation.x = ori_x
        msg.pose.orientation.y = ori_y
        msg.pose.orientation.z = ori_z
        self.arm_pub.publish(msg)


    def arm_interpolate_and_publish(self, start_pose, end_pose, step_time=0.02, steps=100):
        """
        start_pose: [x, y, z, qw, qx, qy, qz]
        end_pose:   [x, y, z, qw, qx, qy, qz]
        step_time:  每一步间隔时间，例如 0.02 = 50Hz
        steps:      插值步数，例如 100 步，整体大约 2 秒
        """

        start_pose = np.asarray(start_pose, dtype=np.float64)
        end_pose = np.asarray(end_pose, dtype=np.float64)

        start_pos = start_pose[:3]
        end_pos = end_pose[:3]

        # 你的 pose 是 [qw, qx, qy, qz]
        start_quat_wxyz = start_pose[3:7]
        end_quat_wxyz = end_pose[3:7]

        # scipy Rotation 需要 [qx, qy, qz, qw]
        start_quat_xyzw = np.array([
            start_quat_wxyz[1],
            start_quat_wxyz[2],
            start_quat_wxyz[3],
            start_quat_wxyz[0],
        ])

        end_quat_xyzw = np.array([
            end_quat_wxyz[1],
            end_quat_wxyz[2],
            end_quat_wxyz[3],
            end_quat_wxyz[0],
        ])

        # 姿态球面插值 Slerp
        key_times = [0.0, 1.0]
        key_rots = R.from_quat([start_quat_xyzw, end_quat_xyzw])
        slerp = Slerp(key_times, key_rots)

        for i in range(steps + 1):
            alpha = i / steps

            # 位置线性插值
            interpolated_pos = start_pos + alpha * (end_pos - start_pos)

            # 姿态 Slerp 插值
            interpolated_rot = slerp([alpha])[0]
            interpolated_quat_xyzw = interpolated_rot.as_quat()

            # 转回你的格式 [qw, qx, qy, qz]
            interpolated_quat_wxyz = np.array([
                interpolated_quat_xyzw[3],
                interpolated_quat_xyzw[0],
                interpolated_quat_xyzw[1],
                interpolated_quat_xyzw[2],
            ])

            interpolated_pose = np.concatenate([
                interpolated_pos,
                interpolated_quat_wxyz,
            ])

            self.publish_arm_action(interpolated_pose)

            time.sleep(step_time)


    def publish_hand_action(self, hand_joints):
        # self.get_logger().info('publish_hand_action')
        stater = JointState()
        # 1) 转成 1D
        joints = np.asarray(hand_joints).reshape(-1)

        # 2) 转 float64 + 转 Python float 列表
        joints = joints.astype(np.float64)
        joints_list = [float(v) for v in joints]

        stater.name = [f"joint_{i}" for i in range(len(joints_list))]
        stater.position = joints_list

        self.publisher_hand.publish(stater)
    
    def get_current_robot_ee(self, timeout=5.0):
        return self.cur_position, self.cur_orientation
    

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

class FrankaEnv(gym.Env):
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
        self.tact_base_path = getattr(config, "TACT_BASE_PATH", "")
        self.enable_tactile = config.ENABLE_TACTILE
        self.enable_gaze_collection = bool(getattr(config, "ENABLE_GAZE_COLLECTION", False))
        self.gaze_display_markers = bool(getattr(config, "GAZE_DISPLAY_MARKERS", True))
        self.gaze_rs_save_width = int(getattr(config, "GAZE_RS_SAVE_WIDTH", 640))
        self.gaze_rs_save_height = int(getattr(config, "GAZE_RS_SAVE_HEIGHT", 480))
        self.gaze_marker_points_realsense = marker_points_for_size(
            self.gaze_rs_save_width,
            self.gaze_rs_save_height,
        )
        self.gaze_prediction_overlay = None
        self.gaze_prediction_lock = threading.Lock()
        self.enable_data_recording = bool(
            getattr(config, "ENABLE_DATA_RECORDING", False)
        ) or self.enable_gaze_collection
        self.fake_env = fake_env
        self.exp_name = config.EXP_NAME

        self.last_gripper_act = time.time()
        self.lastsent = time.time()
        self.hz = hz
        self.grip_loop = self.config.LOOP_CONTROL if hasattr(self.config, 'LOOP_CONTROL') else False
        self.cmd_pose_resync_threshold = float(
            getattr(config, "CMD_POSE_RESYNC_THRESHOLD", 0.05)
        )
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
        
        image_spaces = {
            key: gym.spaces.Box(0, 255, shape=(128, 128, 3), dtype=np.uint8)
            for key in config.REALSENSE_CAMERAS
        }
        if self.enable_gaze_collection:
            image_spaces["gaze_mask"] = gym.spaces.Box(0, 255, shape=(128, 128, 3), dtype=np.uint8)

        if not self.enable_tactile:
            print("init arm without tactaile data")

            self.observation_space = gym.spaces.Dict(
                {
                    "state": gym.spaces.Dict(state_dict),
                    "images": gym.spaces.Dict(image_spaces),
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
                            **image_spaces,
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
        self.action_buffer = []
        self.hand_state = 0.0
        
        
        if fake_env:
            return
        
        self.cap = None

        if self.enable_tactile:
            if DMTacSensor is None:
                raise ImportError(
                    "dmrobotics is required for DM-TAC tactile sensing. "
                    "Please install the DM-TAC SDK package first."
                )

            self.enable_dm_tac_middle = bool(getattr(config, "ENABLE_DM_TAC_MIDDLE", False))
            thumb_serial = getattr(config, "DM_TAC_THUMB_SERIAL_ID", None)
            index_serial = getattr(config, "DM_TAC_INDEX_SERIAL_ID", None)
            middle_serial = getattr(config, "DM_TAC_MIDDLE_SERIAL_ID", None)
            if thumb_serial is None or index_serial is None:
                raise ValueError(
                    "DM_TAC_THUMB_SERIAL_ID and DM_TAC_INDEX_SERIAL_ID must be set "
                    "when ENABLE_TACTILE=True."
                )
            if self.enable_dm_tac_middle and middle_serial is None:
                raise ValueError("DM_TAC_MIDDLE_SERIAL_ID must be set when ENABLE_DM_TAC_MIDDLE=True.")

            self.thumb_tactile_sensor = DMTacSensor(str(thumb_serial))
            self.index_tactile_sensor = DMTacSensor(str(index_serial))
            if self.enable_dm_tac_middle:
                self.middle_tactile_sensor = DMTacSensor(str(middle_serial))

            self.tactile_size = (128, 128)
            self.tactile_depth_scale = float(getattr(config, "DM_TAC_DEPTH_SCALE", 0.25))
            self.thumb_raw_img = np.zeros((*self.tactile_size, 3), dtype=np.uint8)
            self.index_raw_img = np.zeros((*self.tactile_size, 3), dtype=np.uint8)
            self.middle_raw_img = np.zeros((*self.tactile_size, 3), dtype=np.uint8)

            self.thumb_depth_img = np.zeros((*self.tactile_size, 3), dtype=np.uint8)
            self.index_depth_img = np.zeros((*self.tactile_size, 3), dtype=np.uint8)
            self.middle_depth_img = np.zeros((*self.tactile_size, 3), dtype=np.uint8)

            self.rthumb_raw_buffer = []  # Right thumb raw tactile image
            self.rindex_raw_buffer = []  # Right index raw tactile image
            self.rmiddle_raw_buffer = []

            self.rthumb_depth_buffer = []  # Right thumb depth tactile image
            self.rindex_depth_buffer = []  # Right index depth tactile image
            self.rmiddle_depth_buffer = []

            self.tac_thumb_lock = threading.Lock()
            self.tac_index_lock = threading.Lock()
            self.tac_middle_lock = threading.Lock()
            self.tac_main_lock = threading.Lock()
            self.tac_running = True

            # Start threads for each tactile sensor
            self.start_tac_processing()

        try:
            self.init_cameras(self.config.REALSENSE_CAMERAS)
        except Exception:
            if getattr(self, "enable_tactile", False):
                self.close_tac_processing()
            raise
        if self.display_image:
            self.img_queue = queue.Queue()
            self.displayer = ImageDisplayer(self.img_queue, self.url)
            self.displayer.start()

        # Spin ROS callbacks in a background thread and keep references so we can
        # shut everything down cleanly when the environment closes.
        rclpy.init(args=None)

        self.ros_interface = ROSNodeInterface()

        self.executor = rclpy.executors.MultiThreadedExecutor()
        self.executor.add_node(self.ros_interface)
        self.executor_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.executor_thread.start()

        self.interpolation_thread = None
        self.thread_lock = threading.Lock()

        # self.frame_save_path = "/home/wrq/workspaces/HK_TACEXO_WANG/recorded_data/recorded_data_training-2-6-0"  # 可自行修改
        # os.makedirs(self.frame_save_path, exist_ok=True)
        self.frame_count = 0
        self.video_count = 0
        self.global_frame_id = 0
        self._cur_ep_start = None
        self.episode_frame_ranges = []
        self._episode_counter = 0
        self.frame_save_path = getattr(config, "GAZE_FRAME_SAVE_PATH", "./recorded_gaze_data")
        self.frame_root = self.frame_save_path
        self.et_mirror_dir = os.path.join(self.frame_root, "et_images")
        self.rs_mirror_dir = os.path.join(self.frame_root, "rs_images")
        self.data_recorder = None

        self.cur_position = np.zeros(3, dtype=np.float32)
        self.cur_orientation = np.array([0, 1, 0, 0], dtype=np.float32)
        # self.cmd_pose = np.concatenate([self.cur_position.copy(), self.cur_orientation.copy()], axis=0)


        if self.exp_name in ("tennis_ball_pick", "tennis_ball_place", "tennis_ball_pick_and_place"):
        # grip with index
            self.gripper_close_joint = config.GRIPPER_CLOSE_JOINT
            self.gripper_open_joint = config.GRIPPER_OPEN_JOINT

        elif self.exp_name == "twist_bottle_cap" or self.exp_name == "lid_grip" or self.exp_name == "tube_insertion":
            self.gripper_close_joint = config.GRIPPER_CLOSE_JOINT
            self.gripper_twist_joint = config.GRIPPER_TWIST_JOINT
            self.gripper_open_joint = config.GRIPPER_OPEN_JOINT

        self.curr_leap_hand_pos = list(self.gripper_open_joint)

        if self.enable_data_recording:
            self.data_recorder = EpisodeDataRecorder(
                self,
                frame_root=self.frame_root,
                enable_gaze=self.enable_gaze_collection,
                pupil_host=getattr(config, "PUPIL_HOST", "127.0.0.1"),
                pupil_port=int(getattr(config, "PUPIL_PORT", 50020)),
            )
            self.data_recorder.start()
            self.frame_save_path = self.data_recorder.frame_save_path
            self.frame_root = self.data_recorder.frame_root
            self.et_mirror_dir = self.data_recorder.et_mirror_dir
            self.rs_mirror_dir = self.data_recorder.rs_mirror_dir

        print("Initialized franka")


    def start_tac_processing(self):
        # Start threads for each tactile sensor
        self.thumb_thread = threading.Thread(target=self.process_thumb_tactile, daemon=True)
        self.thumb_thread.start()
        self.index_thread = threading.Thread(target=self.process_index_tactile, daemon=True)
        self.index_thread.start()
        if getattr(self, "enable_dm_tac_middle", False):
            self.middle_thread = threading.Thread(target=self.process_middle_tactile, daemon=True)
            self.middle_thread.start()


    def close_tac_processing(self):
        self.tac_running = False
        for thread_name in ("thumb_thread", "index_thread", "middle_thread"):
            if hasattr(self, thread_name):
                getattr(self, thread_name).join(timeout=1.0)

        for sensor_name in ("thumb_tactile_sensor", "index_tactile_sensor", "middle_tactile_sensor"):
            if hasattr(self, sensor_name):
                try:
                    getattr(self, sensor_name).disconnect()
                except Exception as exc:
                    print(f"[DM-TAC] failed to disconnect {sensor_name}: {exc}")


    def step(self, action: np.ndarray) -> tuple:
        """standard gym step function."""
        start_time = time.time()
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.current_action = action.copy()
        self.cur_position, self.cur_orientation = self.ros_interface.get_current_robot_ee()
        self.curpos = np.concatenate((self.cur_position, self.cur_orientation), axis=0)
        # self.nextpos = self.curpos.copy()

        self.nextpos = self.cmd_pose.copy()
        # print("self.nextpos before step = ", self.nextpos)
        # input("debug for step, press Enter to continue...")
        
        xyz_delta = action[:3]

        # print("action scaled = ", xyz_delta * self.action_scale[0])

        # if self.exp_name == "twist_bottle_cap" or self.exp_name == "lid_grip":
        #     if self.nextpos[2] < 0.24 and (0.6 < self.nextpos[0] < 0.8) and (-0.2 < self.nextpos[1] < -0.1):
        #         action[:3] = np.clip(xyz_delta, -0.4, 0.4)
                
        self.nextpos[:3] = self.nextpos[:3] + xyz_delta * self.action_scale[0]

        if self.exp_name in ("tennis_ball_pick", "tennis_ball_place", "tennis_ball_pick_and_place"):
            if self.nextpos[2] < 0.19:
                self.nextpos[2] = 0.19
        elif self.exp_name == "tube_insertion":
            if self.nextpos[2] < 0.20:
                self.nextpos[2] = 0.20
            # if self.nextpos[1] < -0.09:
            #     self.nextpos[1] = -0.09

        # GET ORIENTATION FROM ACTION
        rpy_delta = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        # rpy_delta[0] = action[3]
        self.nextpos[3:6] = self.nextpos[3:6] + rpy_delta * self.action_scale[1]
        # self.nextpos[3:] = (
        #     Rotation.from_euler("xyz", rpy_delta * self.action_scale[1])
        #     * Rotation.from_quat(self.cur_orientation)
        # ).as_quat()
        self.nextpos[3:] = self.cur_orientation.copy()

        self.nextpos[3:] = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

        current_hand_pos = np.asarray(self.curr_leap_hand_pos, dtype=np.float32)
        grip_action = float(np.clip(action[6], -1.0, 1.0))

        if self.exp_name == "twist_bottle_cap" or self.exp_name == "lid_grip" or self.exp_name == "tube_insertion":
            target_hand_pos = self.calculate_hand_pos_segmented(grip_action, current_hand_pos)
        elif self.exp_name in ("tennis_ball_pick", "tennis_ball_place", "tennis_ball_pick_and_place"):
            target_hand_pos = self._cal_hand_close_open(grip_action, current_hand_pos)

        # print("target_hand_pos = ", target_hand_pos)

        self.ros_interface.arm_interpolate_and_publish(self.cmd_pose, self.nextpos, 0.007, 10)
        # self.ros_interface.publish_arm_action(self.nextpos)
        self.cmd_pose = self.nextpos.copy()

        if -0.3 > grip_action or grip_action > 0.3:
            self._send_leap_hand_command(target_hand_pos.copy())

        
        # print(f"[publish End] {t_end:.6f}, Step总耗时(含sleep): {t_end - start_time:.4f}s, 实际频率: {1.0/(t_end - start_time):.2f}Hz")

        self.curr_path_length += 1
        t = time.time()
        self._update_cur_position(self.nextpos, wait=False)
        _timing_log("step.update_cur_position", t)
        tracking_error = float(np.linalg.norm(self.cmd_pose[:3] - self.cur_position))
        if tracking_error > self.cmd_pose_resync_threshold:
            print(
                "[WARN] cmd_pose tracking error too large; "
                f"resync cmd_pose position to robot topic. error={tracking_error:.4f}m"
            )
            self.cmd_pose[:3] = np.asarray(self.cur_position, dtype=np.float32).copy()
        # t_end = time.time()
        # print(f"[update_position End] {t_end:.6f}, Step总耗时（含sleep）: {t_end - start_time:.4f}s, 实际频率: {1.0/(t_end - start_time):.2f}Hz")
        # print("after publish arm action cur_position = ", self.cur_position)
        self.frame_count += 1
        if self.enable_data_recording:
            t = time.time()
            shared_images, shared_depth_images = self.get_rgb_and_dpth_im(show_display=True)
            _timing_log("step.read_shared_rgb_depth", t)
            t = time.time()
            ob = self._get_obs(shared_images)
            _timing_log("step.get_obs", t)
        else:
            shared_images = None
            shared_depth_images = None
            t = time.time()
            ob = self._get_obs()
            _timing_log("step.get_obs", t)
        t = time.time()
        frame_idx = self.save_training_frame(shared_images, shared_depth_images) if self.enable_data_recording else None
        _timing_log("step.save_training_frame", t)
        t = time.time()
        reward = self.compute_reward(ob)
        _timing_log("step.compute_reward", t)
        # self.save_training_frame()
        # print(f"reward in franka_env = {reward}")
        # done = self.curr_path_length >= self.max_episode_length or reward or self.terminate
        done = reward or self.terminate
        # t_end = time.time()
        # print(f"[Step End] {t_end:.6f}, Step总耗时（含sleep）: {t_end - start_time:.4f}s, 实际频率: {1.0/(t_end - start_time):.2f}Hz")
        # print("curr hand pos = ", self.curr_leap_hand_pos)
        # input("debug for hand pos")
        info = {"succeed": reward}
        if self.enable_data_recording and frame_idx is not None:
            info["frame_idx"] = int(frame_idx)
            info["episode_id"] = int(self._episode_counter)
        _timing_log("step.total", start_time)

        # time.sleep(1.5)
        dt = time.time() - start_time
        time.sleep(max(0, (1.0 / self.hz) - dt))
        t_end = time.time()
        return ob, int(reward), done, False, info
    
    def _close_open_pose_init(self, current_hand_pos: np.ndarray, max_steps: int=10):
        self._lower = np.minimum(self.gripper_open_joint, self.gripper_close_joint)
        self._upper = np.maximum(self.gripper_open_joint, self.gripper_close_joint)

        self._cmd_pos = np.asarray(current_hand_pos, dtype=np.float32)

        diff = self.gripper_close_joint - self.gripper_open_joint
        self.max_joint_delta = float(np.max(np.abs(diff))) / max_steps

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
        max_length = max(1e-9, abs(grip_action) * self.max_joint_delta)

        if dist <= max_length:
            new_pos = target.copy()
        else:
            direction = diff / dist
            new_pos = pos + direction * max_length

        new_pos = np.clip(new_pos, self._lower, self._upper)
        self._cmd_pos = new_pos.copy()
        return new_pos
    

    def _segmented_init(self, current_hand_pos: np.ndarray, max_steps: int=10):
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

        mean_len = float(np.mean(seg_lens))
        self._per_step_path_len = max(1e-6, mean_len / max_steps)
        self._max_joint_delta = float(np.max(np.abs(seg_vecs))) / max_steps
        self._snap_eps = 1e-6
        self.is_segmented_init = True
        self._cmd_pos = np.asarray(current_hand_pos, dtype=np.float32)

        # --- for phase / turns encoding ---
        # self._seg_lens = seg_lens.astype(np.float32)                 # (num_seg,)
        # self._total_len = float(np.sum(self._seg_lens)) + 1e-12
        # # cumulative length at each segment start, length = num_seg
        # self._cumlen = np.concatenate([[0.0], np.cumsum(self._seg_lens[:-1])]).astype(np.float32)

        # self._turns = 0
        # self.gripper_phase = 0.0               # in [0,1)
        # self.gripper_unwrapped_phase = 0.0     # turns + phase (optional)
    
    
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

            # # --- update phase cache ---
            # seg_idx_for_phase = int(self._seg_start_idx)
            # if not self._loop:
            #     seg_idx_for_phase = max(0, min(self._num_seg - 1, seg_idx_for_phase))
            # else:
            #     seg_idx_for_phase = seg_idx_for_phase % self._num_seg

            # s = float(self._cumlen[seg_idx_for_phase] + segment_progress * self._seg_lens[seg_idx_for_phase])
            # phase = s / self._total_len
            # # keep in [0,1)
            # if phase >= 1.0:
            #     phase = 0.0
            # if phase < 0.0:
            #     phase = 0.0
            # self.gripper_phase = float(phase)
            # self.gripper_unwrapped_phase = float(self._turns) + self.gripper_phase

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
                        prev_idx = (start_idx - 1) % self._num_seg
                        if start_idx == 0 and prev_idx == (self._num_seg - 1):
                            pos = waypoints[0].copy()
                            remaining_path_len = 0.0
                            break
                        self._seg_start_idx = prev_idx
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
                        prev_idx = (start_idx - 1) % self._num_seg
                        if start_idx == 0 and prev_idx == (self._num_seg - 1):
                            pos = waypoints[0].copy()
                            remaining_path_len = 0.0
                            break
                        self._seg_start_idx = prev_idx
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
    
    
    # def _hand_progress_scalar(self, hand_joint_pos: np.ndarray) -> float:
    #     hand_joint_pos = np.asarray(hand_joint_pos, dtype=np.float32)

    #     # If waypoints not prepared, fallback to open<->close
    #     if not hasattr(self, "_waypoints"):
    #         open_joint = np.asarray(self.gripper_open_joint, dtype=np.float32)
    #         close_joint = np.asarray(self.gripper_close_joint, dtype=np.float32)
    #         seg_vec = close_joint - open_joint
    #         seg_len_sq = float(np.dot(seg_vec, seg_vec)) + 1e-12
    #         ratio = float(np.dot(hand_joint_pos - open_joint, seg_vec) / seg_len_sq)
    #         return float(np.clip(ratio, 0.0, 1.0))

    #     waypoints = np.asarray(self._waypoints, dtype=np.float32)  # (3, D)
    #     num_wp = int(waypoints.shape[0])                           # 3
    #     num_seg = num_wp - 1                                       # 2

    #     # segments: 0->1, 1->2
    #     segment_vectors = waypoints[1:] - waypoints[:-1]           # (2, D)
    #     segment_lengths = np.linalg.norm(segment_vectors, axis=1) + 1e-12  # (2,)

    #     prefix_lengths = np.concatenate([[0.0], np.cumsum(segment_lengths)])  # (3,)
    #     total_length = float(prefix_lengths[-1]) + 1e-12

    #     closest_seg = 0
    #     closest_ratio = 0.0
    #     min_dist_sq = float("inf")

    #     for seg_idx in range(num_seg):
    #         seg_start = waypoints[seg_idx]
    #         seg_vec = segment_vectors[seg_idx]
    #         seg_len_sq = float(np.dot(seg_vec, seg_vec)) + 1e-12

    #         ratio = float(np.dot(hand_joint_pos - seg_start, seg_vec) / seg_len_sq)
    #         ratio = float(np.clip(ratio, 0.0, 1.0))

    #         proj = seg_start + ratio * seg_vec
    #         dist_sq = float(np.sum((hand_joint_pos - proj) ** 2))

    #         if dist_sq < min_dist_sq:
    #             min_dist_sq = dist_sq
    #             closest_seg = seg_idx
    #             closest_ratio = ratio

    #     arc_length = float(prefix_lengths[closest_seg]) + float(closest_ratio) * float(segment_lengths[closest_seg])
    #     progress = float(np.clip(arc_length / total_length, 0.0, 1.0))
    #     return progress
    
    def _hand_progress_scalar(self, hand_joint_pos: np.ndarray) -> float:
        """
        Non-loop: open polyline progress in [0,1] : open -> close -> twist
        Loop: closed polyline progress in [0,1)  : open -> close -> twist -> open
            plus an internal self._turns counter (optional) and forbids backward wrap
            from open->close path to twist->open edge.
        """
        hand_joint_pos = np.asarray(hand_joint_pos, dtype=np.float32)

        # If waypoints not prepared, fallback
        if not hasattr(self, "_waypoints"):
            open_joint = np.asarray(self.gripper_open_joint, dtype=np.float32)
            close_joint = np.asarray(self.gripper_close_joint, dtype=np.float32)
            seg_vec = close_joint - open_joint
            seg_len_sq = float(np.dot(seg_vec, seg_vec)) + 1e-12
            ratio = float(np.dot(hand_joint_pos - open_joint, seg_vec) / seg_len_sq)
            return float(np.clip(ratio, 0.0, 1.0))

        waypoints = np.asarray(self._waypoints, dtype=np.float32)  # (3, D)
        num_wp = int(waypoints.shape[0])                           # 3

        # ---------------------------
        # choose segments depending on loop
        # ---------------------------
        if getattr(self, "grip_loop", False):
            # segments: 0->1, 1->2, 2->0 (closed)
            seg_starts = waypoints
            seg_ends = waypoints[(np.arange(num_wp) + 1) % num_wp]
            segment_vectors = seg_ends - seg_starts                # (3, D)
            num_seg = num_wp                                       # 3
        else:
            # segments: 0->1, 1->2 (open)
            seg_starts = waypoints[:-1]
            seg_ends = waypoints[1:]
            segment_vectors = seg_ends - seg_starts                # (2, D)
            num_seg = num_wp - 1                                   # 2

        segment_lengths = np.linalg.norm(segment_vectors, axis=1) + 1e-12  # (num_seg,)
        prefix_lengths = np.concatenate([[0.0], np.cumsum(segment_lengths)])  # (num_seg+1,)
        total_length = float(prefix_lengths[-1]) + 1e-12

        # find closest segment projection
        closest_seg = 0
        closest_ratio = 0.0
        min_dist_sq = float("inf")

        for seg_idx in range(num_seg):
            seg_start = seg_starts[seg_idx]
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
        progress = float(arc_length / total_length)

        if not getattr(self, "grip_loop", False):
            # open path: [0,1]
            return float(np.clip(progress, 0.0, 1.0))

        # ---------------------------
        # loop path: [0,1) + turns (internal) + forbid backward wrap
        # ---------------------------
        # keep progress in [0,1)
        progress = float(progress % 1.0)

        # init memory
        if not hasattr(self, "_turns"):
            self._turns = 0
        if not hasattr(self, "_prev_hand_progress"):
            self._prev_hand_progress = progress

        prev = float(self._prev_hand_progress)
        curr = float(progress)

        # thresholds (can tune)
        hi = 0.85
        lo = 0.3

        if prev < 0.3 and (curr - prev) > 0.6:
            curr = prev

        # forward wrap: near 1 -> near 0
        if prev > hi and curr < lo:
            self._turns += 1

        self._prev_hand_progress = curr
        return curr + self._turns

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

    def set_gaze_prediction_overlay(
        self,
        image_key: str = "front_camera",
        xy_norm: Tuple[float, float] | None = None,
        conf: float | None = None,
    ):
        """Set the gaze prediction point drawn by the RGB display thread."""
        with self.gaze_prediction_lock:
            if xy_norm is None:
                self.gaze_prediction_overlay = None
                return
            self.gaze_prediction_overlay = {
                "image_key": image_key,
                "xy_norm": (float(xy_norm[0]), float(xy_norm[1])),
                "conf": None if conf is None else float(conf),
            }

    def _draw_gaze_prediction_overlay(self, image_bgr: np.ndarray, image_key: str):
        with self.gaze_prediction_lock:
            overlay = None if self.gaze_prediction_overlay is None else dict(self.gaze_prediction_overlay)
        if overlay is None or overlay.get("image_key") != image_key:
            return image_bgr

        xy_norm = overlay.get("xy_norm")
        if xy_norm is None:
            return image_bgr

        x_norm, y_norm = xy_norm
        if not (np.isfinite(x_norm) and np.isfinite(y_norm)):
            return image_bgr

        image_bgr = image_bgr.copy()
        height, width = image_bgr.shape[:2]
        x = int(np.clip(round(x_norm * (width - 1)), 0, width - 1))
        y = int(np.clip(round(y_norm * (height - 1)), 0, height - 1))

        color = (0, 0, 255)
        cv2.drawMarker(
            image_bgr,
            (x, y),
            color,
            markerType=cv2.MARKER_CROSS,
            markerSize=32,
            thickness=3,
            line_type=cv2.LINE_AA,
        )
        cv2.circle(image_bgr, (x, y), 7, (0, 255, 0), 2, cv2.LINE_AA)
        label = "gaze pred"
        if overlay.get("conf") is not None:
            label += f" conf={overlay['conf']:.2f}"
        cv2.putText(
            image_bgr,
            label,
            (min(x + 12, max(0, width - 220)), max(24, y - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
        return image_bgr
    

    def get_im(self, return_full_images: bool = False) -> Dict[str, np.ndarray]:
        """Get images from the realsense cameras."""
        images = {}
        display_images = {}
        full_res_images = {}
        video_images = {}
        for key, cap in self.cap.items():
            if key == "front_camera_2":
                continue
            try:
                t = time.time()
                frame = cap.read()
                _timing_log(f"get_im.{key}.cap_read", t)
                t = time.time()
                if frame.ndim == 3 and frame.shape[2] == 4:
                    rgb = frame[..., :3] # 这里是bgr格式，cv2.imshow输入bgr,显示rgb图像
                else:
                    rgb = frame
                rgb = rgb.astype(np.uint8)
                video_images[key] = rgb
                # cropped_rgb = self.config.IMAGE_CROP[key](rgb) if key in self.config.IMAGE_CROP else rgb
                cropped_rgb = rgb #当前不需要裁剪
                resized = cv2.resize(
                    cropped_rgb, self.observation_space["images"][key].shape[:2][::-1]
                )
                images[key] = resized[..., ::-1]
                # display_images[key] = resized
                display_rgb = cropped_rgb
                if (
                    key == "front_camera"
                    and self.enable_gaze_collection
                    and self.gaze_display_markers
                ):
                    display_rgb = draw_gaze_display_markers(display_rgb)
                display_rgb = self._draw_gaze_prediction_overlay(display_rgb, key)
                display_images[key + "_full"] = display_rgb
                full_res_images[key] = copy.deepcopy(cropped_rgb)  # Store the full resolution cropped image
                _timing_log(f"get_im.{key}.process", t)
            except queue.Empty:
                input(
                    f"{key} camera frozen. Check connect, then press enter to relaunch..."
                )
                cap.close()
                self.init_cameras(self.config.REALSENSE_CAMERAS)
                return self.get_im(return_full_images=return_full_images)
        # if not self.enable_tactile:
        #     if self.display_image:
        #         display_image = {
        #             "front_camera": display_images["front_camera_full"],
        #             "wrist_camera": display_images["wrist_camera_full"],
        #         }
        if self.enable_tactile:
            tactile_depth = np.concatenate([self.thumb_depth_img, self.index_depth_img], axis=1)
            full_res_images["tactile_data"] = tactile_depth
            video_images["tactile_data"] = tactile_depth
            with self.tac_index_lock:
                display_images["tactile_depth"] = tactile_depth
        if self.display_image:
            t = time.time()
            self.img_queue.put(display_images)
            _timing_log("get_im.display_queue_put", t)
        if self.save_video:
            self.recording_frames.append(video_images)
        if self.enable_gaze_collection and "gaze_mask" in self.observation_space["images"].spaces:
            h, w, c = self.observation_space["images"]["gaze_mask"].shape
            images["gaze_mask"] = np.zeros((h, w, c), dtype=np.uint8)
        if return_full_images:
            record_images = {key: img[..., ::-1] for key, img in full_res_images.items()}
            return images, record_images
        return images


    def get_rgb_and_dpth_im(
        self,
        show_display: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Get images from the realsense cameras."""
        images = {}
        depth_images = {}
        display_images = {}
        full_res_images = {}  # New dictionary to store full resolution cropped images
        for key, cap in self.cap.items():
            try:
                t = time.time()
                frame = cap.read()
                _timing_log(f"get_rgb_and_dpth_im.{key}.cap_read", t)
                t = time.time()
                if frame.ndim == 3 and frame.shape[2] == 4:
                    rgb = frame[..., :3]   # BGR 彩色
                    depth = frame[..., 3]    # 深度
                else:
                    rgb = frame
                    depth = None
                rgb = rgb.astype(np.uint8, copy=False)
                cropped_rgb = rgb #当前不需要裁剪
                resized = cv2.resize(
                    cropped_rgb, (640, 480)
                )
                images[key] = resized[..., ::-1]
                depth_images[key] = depth
                display_rgb = cropped_rgb
                if (
                    key == "front_camera"
                    and self.enable_gaze_collection
                    and self.gaze_display_markers
                ):
                    display_rgb = draw_gaze_display_markers(display_rgb)
                display_rgb = self._draw_gaze_prediction_overlay(display_rgb, key)
                display_images[key + "_full"] = display_rgb
                full_res_images[key] = copy.deepcopy(cropped_rgb)  # Store the full resolution cropped image
                _timing_log(f"get_rgb_and_dpth_im.{key}.process", t)
            except queue.Empty:
                input(
                    f"{key} camera frozen. Check connect, then press enter to relaunch..."
                )
                cap.close()
                self.init_cameras(self.config.REALSENSE_CAMERAS)
                return self.get_rgb_and_dpth_im(show_display=show_display)

        if show_display and self.enable_tactile:
            tactile_depth = np.concatenate([self.thumb_depth_img, self.index_depth_img], axis=1)
            display_images["tactile_depth"] = tactile_depth

        if show_display and self.display_image:
            t = time.time()
            self.img_queue.put(display_images)
            _timing_log("get_rgb_and_dpth_im.display_queue_put", t)
        return images, depth_images


    def process_tactile_data(self, sensor, img_size):
        raw_img = sensor.getRawImage()
        depth = sensor.getDepth()

        raw_img = np.asarray(raw_img)
        raw_img = raw_img.astype(np.uint8, copy=False)
        if raw_img.shape[:2] != img_size[::-1]:
            raw_img = cv2.resize(raw_img, img_size, interpolation=cv2.INTER_LINEAR)

        depth = np.asarray(depth, dtype=np.float32)
        depth_input = np.nan_to_num(
            depth * self.tactile_depth_scale * 255.0,
            nan=0.0,
            posinf=255.0,
            neginf=0.0,
        )
        depth_input = np.clip(depth_input, 0, 255).astype(np.uint8)
        depth_img = cv2.applyColorMap(depth_input, cv2.COLORMAP_HOT)
        depth_img = cv2.resize(depth_img, img_size, interpolation=cv2.INTER_LINEAR)

        return raw_img, depth_img
    

    def process_thumb_tactile(self):
        while getattr(self, "tac_running", True):
            try:
                thumb_raw_img, thumb_depth_img = self.process_tactile_data(self.thumb_tactile_sensor, self.tactile_size)
            except Exception as exc:
                print(f"[DM-TAC][thumb] {exc}")
                time.sleep(0.1)
                continue

            with self.tac_thumb_lock:
                self.thumb_raw_img = thumb_raw_img
                self.thumb_depth_img = thumb_depth_img
            time.sleep(0.01)


    def process_index_tactile(self):
        # Process index tactile data
        while getattr(self, "tac_running", True):
            try:
                index_raw_img, index_depth_img = self.process_tactile_data(self.index_tactile_sensor, self.tactile_size)
            except Exception as exc:
                print(f"[DM-TAC][index] {exc}")
                time.sleep(0.1)
                continue

            with self.tac_index_lock:
                self.index_raw_img = index_raw_img
                self.index_depth_img = index_depth_img
            time.sleep(0.01)
    

    def process_middle_tactile(self):
        # Middle DM-TAC is kept off by default and is not included in tactile_data.
        while getattr(self, "tac_running", True):
            try:
                middle_raw_img, middle_depth_img = self.process_tactile_data(self.middle_tactile_sensor, self.tactile_size)
            except Exception as exc:
                print(f"[DM-TAC][middle] {exc}")
                time.sleep(0.1)
                continue

            with self.tac_middle_lock:
                self.middle_raw_img = middle_raw_img
                self.middle_depth_img = middle_depth_img
            time.sleep(0.01)
    

    def reset(self, joint_reset=False, **kwargs):
        print("franka_env reset")
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

        # for cam_name, kwargs in extra_cameras_dict.items():
        #     cap = VideoCapture(
        #         RSCapture(name=cam_name, **kwargs)
        #     )
        #     self.cap[cam_name] = cap

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
            

    def _update_cur_position(self, arm_action=None, timeout=10.0, wait_threshold=0.05, wait=True):
        """
        Internal function to get the latest state of the robot and its gripper.
        """
        start = time.time()
        self.cur_position, self.cur_orientation = self.ros_interface.get_current_robot_ee()
        # joint_position = self.ros_interface.get_current_joint()
        # self.joint_position = np.asarray(joint_position, dtype=np.float32).copy()

        hand_joint_msg = self.ros_interface.get_current_leap_position()
        self.curr_leap_hand_pos = np.asarray(hand_joint_msg, dtype=np.float32).copy()
        # print("in _update_cur_position curr_leap_hand_pos = ", self.curr_leap_hand_pos)

        if wait and arm_action is not None:
            diff = np.asarray(arm_action[:3], dtype=np.float32) - np.asarray(self.cur_position, dtype=np.float32)
        else:
            diff = np.zeros(3, dtype=np.float32)

        while wait and np.max(np.abs(diff)) > wait_threshold:
            if time.time() - start > timeout:
                print("[WARN] 等待机械臂到位超时")
                break
            time.sleep(0.02)
            self.cur_position, self.cur_orientation = self.ros_interface.get_current_robot_ee()
            # joint_position = self.ros_interface.get_current_joint()
            # self.joint_position = np.asarray(joint_position, dtype=np.float32).copy()
            hand_joint_msg = self.ros_interface.get_current_leap_position()
            self.curr_leap_hand_pos = np.asarray(hand_joint_msg, dtype=np.float32).copy()
            diff = np.asarray(arm_action[:3], dtype=np.float32) - np.asarray(self.cur_position, dtype=np.float32)
        
        # if self.exp_name =="twist_bottle_cap":
        #     self.hand_state = self.gripper_unwrapped_phase
        # else:
        self.hand_state = self._hand_progress_scalar(self.curr_leap_hand_pos)
        # self.hand_state = 0


    def _get_obs(self, source_images=None, return_record_images: bool = False) -> dict:
        record_images = None
        if source_images is None:
            if return_record_images:
                images, record_images = self.get_im(return_full_images=True)
            else:
                images = self.get_im()
        else:
            images = {}
            image_spaces = self.observation_space["images"].spaces
            for cam_key, img in source_images.items():
                if cam_key not in image_spaces:
                    continue
                target_hw = image_spaces[cam_key].shape[:2]
                images[cam_key] = cv2.resize(img, target_hw[::-1])
            if self.enable_gaze_collection and "gaze_mask" in image_spaces:
                h, w, c = image_spaces["gaze_mask"].shape
                images["gaze_mask"] = np.zeros((h, w, c), dtype=np.uint8)
        # if self.grip_loop:
        #     state_observation = {
        #         "tcp_pos": self.cur_position,
        #         "tcp_ori": self.cur_orientation,
        #     }
        # else:
        state_observation = {
            "tcp_pos": self.cur_position,
            "tcp_ori": self.cur_orientation,
            "gripper_pose": self.hand_state,
        }
        
        obs = {
            "images": {},
            "state": state_observation,
        }
        for cam_key, img in images.items():
            obs["images"][cam_key] = img
            
        if self.enable_tactile:
            depth_canvas = np.concatenate([self.thumb_depth_img, self.index_depth_img], axis=1)
            obs["images"]["tactile_data"] = depth_canvas
                
        obs = copy.deepcopy(obs)
        if return_record_images:
            return obs, record_images
        return obs


    def close(self):
        self.enable_gaze_collection = False
        data_recorder = getattr(self, "data_recorder", None)
        if data_recorder is not None:
            data_recorder.close()
        if getattr(self, "enable_tactile", False):
            self.close_tac_processing()
        if hasattr(self, 'listener'):
            self.listener.stop()
        self.close_cameras()
        if self.display_image and hasattr(self, "img_queue"):
            try:
                self.img_queue.put(None)
            except Exception:
                pass
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
            if hasattr(self, "displayer"):
                self.displayer.join(timeout=1.0)

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


    def pause_display(self):
        if not getattr(self, "display_image", False):
            return
        try:
            if hasattr(self, "img_queue"):
                self.img_queue.put(None, timeout=0.1)
            if hasattr(self, "displayer"):
                self.displayer.join(timeout=1.0)
            cv2.destroyAllWindows()
        except Exception:
            pass


    def resume_display(self):
        if not getattr(self, "display_image", False):
            return
        try:
            self.img_queue = queue.Queue()
            self.displayer = ImageDisplayer(self.img_queue, self.url)
            self.displayer.start()
        except Exception as exc:
            print(f"[FrankaEnv][resume_display] failed: {exc}")

    def pose_callback(self, msg):
        self.arm_position = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        self.arm_orientation = np.array([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
        # self.arm_pose = np.concatenate((position, orientation), axis=0)

    def stop_cur_command(self):
        print("stop current command")
        pos = self.cur_position.copy()
        ori = self.cur_orientation.copy()
        nextpos = np.concatenate((pos, ori), axis=0)
        self.ros_interface.arm_interpolate_and_publish(nextpos, nextpos)
        # time.sleep(2.0)
        # self.get_im()


    def save_training_frame(self, images=None, depth_images=None):
        if self.data_recorder is None:
            return None
        frame_id = self.data_recorder.save_frame(images=images, depth_images=depth_images)
        self.global_frame_id = self.data_recorder.global_frame_id
        return frame_id


    def save_all_data_on_exit(self):
        if self.data_recorder is None:
            return
        self.data_recorder.save_all_data_on_exit()


    def end_episode_and_collect(self):
        if self.data_recorder is None:
            return None
        self.data_recorder.cur_ep_start = self._cur_ep_start
        frame_range = self.data_recorder.end_episode()
        self._cur_ep_start = self.data_recorder.cur_ep_start
        self.episode_frame_ranges = self.data_recorder.episode_frame_ranges
        return frame_range


    def mark_episode_start_for_recording(self):
        if self.data_recorder is None:
            return None
        start = self.data_recorder.mark_episode_start()
        self.global_frame_id = self.data_recorder.global_frame_id
        self._cur_ep_start = self.data_recorder.cur_ep_start
        self._episode_counter = self.data_recorder.episode_counter
        return start
