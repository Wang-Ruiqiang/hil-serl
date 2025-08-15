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

from franka_env.utils.rotations import euler_2_quat, quat_2_euler
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64MultiArray
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

            frame = np.concatenate(
                [cv2.resize(v, (128, 128)) for k, v in img_array.items() if "full" not in k], axis=1
            )
            cv2.imshow(self.name, frame)
            cv2.waitKey(1)


##############################################################################

class DefaultEnvConfig:
    """Default configuration for FrankaEnv. Fill in the values below."""

    SERVER_URL: str = "http://127.0.0.1:5000/"
    REALSENSE_CAMERAS: Dict = {
        "front_camera": "242422303461",
        # "side_camera": "234222300515",
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
        # print("msg = ", msg)

        # 将 joint name 和对应的位置打包为字典
        joint_dict = {name: pos for name, pos in zip(msg.name, msg.position)}

        # 按照你需要的顺序提取关节角：joint1 ~ joint6
        ordered_joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        ordered_joint_positions = [joint_dict.get(joint, 0.0) for joint in ordered_joint_names]

        # 保存为 numpy array
        self.joint_position = np.array(ordered_joint_positions, dtype=np.float32)

        # 设置事件为“数据已接收”
        self.joint_event.set()


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
        hz=20,
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
        self.is_arm_only = config.IS_ARM_ONLY
        self.tact_base_path = config.TACT_BASE_PATH
        self.enable_tactile = config.ENABLE_TACTILE



        # convert last 3 elements from euler to quat, from size (6,) to (7,)
        # self.resetpos = np.concatenate(
        #     [config.RESET_POSE[:3], euler_2_quat(config.RESET_POSE[3:])]
        # )
        self.last_gripper_act = time.time()
        self.lastsent = time.time()
        # self.randomreset = config.RANDOM_RESET
        # self.random_xy_range = config.RANDOM_XY_RANGE
        # self.random_rz_range = config.RANDOM_RZ_RANGE
        self.hz = hz
        # self.joint_reset_cycle = config.JOINT_RESET_PERIOD  # reset the robot joint every 200 cycles

        self.save_video = save_video
        if self.save_video:
            print("Saving videos!")
            self.recording_frames = []

        # boundary box
        # self.xyz_bounding_box = gym.spaces.Box(
        #     config.ABS_POSE_LIMIT_LOW[:3],
        #     config.ABS_POSE_LIMIT_HIGH[:3],
        #     dtype=np.float64,
        # )
        # self.rpy_bounding_box = gym.spaces.Box(
        #     config.ABS_POSE_LIMIT_LOW[3:],
        #     config.ABS_POSE_LIMIT_HIGH[3:],
        #     dtype=np.float64,
        # )
        # Action/Observation Space
        if not self.is_arm_only:
            print("init arm with hand")
            self.action_space = gym.spaces.Box(
                np.ones((22,), dtype=np.float32) * -1,
                np.ones((22,), dtype=np.float32),
            )

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
                            "gripper_pose": gym.spaces.Box(-np.inf, np.inf, shape=(16,)),
                        }
                    ),
                    "images": gym.spaces.Dict(
                        {key: gym.spaces.Box(0, 255, shape=(240, 320, 3), dtype=np.uint8) 
                                    for key in config.REALSENSE_CAMERAS}
                    ),
                }
            )
        elif self.enable_tactile:
            self.action_space = gym.spaces.Box(
                np.ones((6,), dtype=np.float32) * -1,
                np.ones((6,), dtype=np.float32),
            )
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
                                -np.inf, np.inf, shape=(1,), dtype=np.int32
                            )
                        }
                    ),
                    "images": gym.spaces.Dict(
                        {
                            **{key: gym.spaces.Box(0, 255, shape=(240, 320, 3), dtype=np.uint8) 
                                    for key in config.REALSENSE_CAMERAS},
                            "tactile_data": gym.spaces.Box(0, 255, shape=(480, 1920, 3), dtype=np.uint8),
                        }
                    ),
                }
            )
        else:
            print("init arm only")
            self.action_space = gym.spaces.Box(
                np.ones((6,), dtype=np.float32) * -1,
                np.ones((6,), dtype=np.float32),
            )
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
                            # "gripper_pose": gym.spaces.Box(
                            #     -np.inf, np.inf, shape=(1,), dtype=np.int32
                            # )
                        }
                    ),
                    "images": gym.spaces.Dict(
                        {key: gym.spaces.Box(0, 255, shape=(240, 320, 3), dtype=np.uint8) 
                                    for key in config.REALSENSE_CAMERAS}
                    ),
                }
            )


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
        
        self.cycle_count = 0

        self.front_color_buffer = []            #  D435i color 
        self.front_depth_buffer = []            #  D435i depth 
        self.side_color_buffer = []           #  D435i 2 color
        self.side_depth_buffer = []           #  D435i 2 depth
        self.joint_buffer = []

        # robot_urdf_path = "/home/qiangqiang/workspaces/HK_TACTEXO_DATA/denso_robot_with_ati_4.urdf"
        # self.data_count = 0
        # self.data = read_utils.read_data(robot_urdf_path,True)
        # self._update_cur_position()

        if fake_env:
            return

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

        executor = rclpy.executors.MultiThreadedExecutor()
        executor.add_node(self.ros_interface)
        executor_thread = threading.Thread(target=executor.spin, daemon=True)
        executor_thread.start()

        self.interpolation_thread = None
        self.thread_lock = threading.Lock()

        self.print_action = True
        self._last_step_time = None

        self.frame_save_path = "/home/ruiqiang/workspaces/HK_TACEXO_WANG/recorded_data/recorded_data_training-8-15-0"  # 可自行修改
        os.makedirs(self.frame_save_path, exist_ok=True)
        self.frame_count = 0

        self.cur_position = np.zeros(3, dtype=np.float32)
        self.cur_oritation = np.zeros(4, dtype=np.float32)

        # initial close pose
        # self.gripper_close_joint = [
        #     3.552699565887451172, 3.572641372680664062, 4.193903446197509766, 3.380893707275390625,
        #     3.423845052719116211, 3.796602487564086914, 3.713767528533935547, 3.592582941055297852,
        #     3.144660711288452148, 3.288854837417602539, 2.890019893646240234, 3.325670242309570312,
        #     4.592738628387451172, 3.472932577133178711, 3.713767528533935547, 3.051087856292724609
        # ]

        # grip with index
        self.gripper_close_joint = [
            3.546563625335693359, 4.127942085266113281, 3.413689804077148438, 3.641670465469360352,
            3.626330614089965820, 3.529689788818359375, 2.931437253952026367, 3.782796621322631836,
            3.144660711288452148, 3.181476116180419922, 3.144660711288452148, 3.140058755874633789,
            4.661767482757568359, 3.366175127029418945, 3.260291767120361328, 3.566796636581420898
        ]

        #grip with middle
        # self.gripper_close_joint = [
        #     3.132388830184936523, 3.186078071594238281, 3.153864383697509766, 3.147728681564331055,
        #     3.201417922973632812, 4.543651103973388672, 2.943709135055541992, 3.740427684783935547,
        #     3.144660711288452148, 3.181476116180419922, 3.144660711288452148, 3.140058755874633789,
        #     4.825903415679931641, 3.525670242309570312, 3.230563640594482422, 3.240349960327148438
        # ]

        # self.gripper_open_joint = [
        #     2.989728450775146484, 3.231437253952026367, 3.438389015197753906, 3.96806390762329102,    #index
        #     2.904854822158813477, 3.202951908111572266, 3.466796636581420898, 3.969689750671386719,   #middle
        #     3.218291759490966797, 3.238233327865600586, 2.867010116577148438, 3.325670242309570312,
        #     4.312019824981689453, 3.905515193939208984, 3.374757766723632812, 3.597184896469116211  #thumb
        # ]

        self.last_hand_pos = [
            3.546563625335693359, 4.127942085266113281, 3.413689804077148438, 3.641670465469360352,
            3.626330614089965820, 3.529689788818359375, 2.931437253952026367, 3.782796621322631836,
            3.144660711288452148, 3.181476116180419922, 3.144660711288452148, 3.140058755874633789,
            4.661767482757568359, 3.366175127029418945, 3.260291767120361328, 3.566796636581420898
        ]

        self.changed_hand_pos = [
            # 3.140058755874633789, 3.187612056732177734, 3.155398368835449219, 3.150796651840209961,
            2.989728450775146484, 3.231437253952026367, 3.438389015197753906, 3.96806390762329102,    #index
            2.904854822158813477, 3.202951908111572266, 3.466796636581420898, 3.969689750671386719,   #middle
            3.218291759490966797, 3.238233327865600586, 2.867010116577148438, 3.325670242309570312,
            4.312019824981689453, 3.905515193939208984, 3.374757766723632812, 3.597184896469116211
        ]

        self.hand_state = 0 #hand opened lable

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

        print("action = ", action)

        xyz_delta = action[:3]

        self.nextpos = np.concatenate((self.cur_position, self.cur_oritation), axis=0)
        self.nextpos[:3] = self.nextpos[:3] + xyz_delta * self.action_scale[0]


        # GET ORIENTATION FROM ACTION
        rpy_delta = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        # self.nextpos[3:] = (
        #     Rotation.from_euler("xyz", action[3:6] * self.action_scale[1])
        #     * Rotation.from_quat(self.cur_oritation)
        # ).as_quat()
        self.nextpos[3:] = (
            Rotation.from_euler("xyz", rpy_delta * self.action_scale[1])
            * Rotation.from_quat(self.cur_oritation)
        ).as_quat()
        
        print("nextpos = ", self.nextpos)
        self.ros_interface.publish_arm_action(self.nextpos)


        if not np.array_equal(self.last_hand_pos, self.changed_hand_pos):
            self._send_leap_hand_command(self.changed_hand_pos)
            self.last_hand_pos = self.changed_hand_pos

        # time.sleep(2.1)
        dt = time.time() - start_time
        time.sleep(max(0, (1.0 / self.hz) - dt))
        t_end = time.time()
        print(f"[publish End] {t_end:.6f}, Step总耗时(含sleep): {t_end - start_time:.4f}s, 实际频率: {1.0/(t_end - start_time):.2f}Hz")

        self.curr_path_length += 1
        self._update_cur_position(self.nextpos)

        t_end = time.time()
        # print(f"[update_position End] {t_end:.6f}, Step总耗时（含sleep）: {t_end - start_time:.4f}s, 实际频率: {1.0/(t_end - start_time):.2f}Hz")
        # print("after publish arm action cur_position = ", self.cur_position)
        self.frame_count += 1
        self.save_training_frame()

        ob = self._get_obs()
        reward = self.compute_reward(ob)
        print(f"reward in denso_env = {reward}")
        # done = self.curr_path_length >= self.max_episode_length or reward or self.terminate
        done = reward or self.terminate
        t_end = time.time()
        # print(f"[Step End] {t_end:.6f}, Step总耗时（含sleep）: {t_end - start_time:.4f}s, 实际频率: {1.0/(t_end - start_time):.2f}Hz")
        return ob, int(reward), done, False, {"succeed": reward}
    

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

        # Store full resolution cropped images separately
        if self.save_video:
            self.recording_frames.append(full_res_images)

        if self.display_image:
            with self.tac_index_lock:
                # index_heat_map_resized = cv2.resize(self.index_heat_map, (128, 128))  # 如果想缩放
                heat_map = cv2.hconcat([self.thumb_heat_map, self.index_heat_map])
                display_image = {"heat_map": heat_map}
            self.img_queue.put(display_image)
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
                # cropped_rgb = self.config.IMAGE_CROP[key](rgb) if key in self.config.IMAGE_CROP else rgb
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
        # Add subtitles to each image
        # cv2.putText(heat_map, "Thumb", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        # Resize images for display
        target_size = img_size
        heat_map = cv2.resize(heat_map, target_size, interpolation=cv2.INTER_LINEAR)
        points, gradients = sensor.height_map_2_point_cloud_gradients(height_map)

        return raw_img, points, heat_map
    

    def process_thumb_tactile(self):
        while True:
            thumb_raw_img, thumb_points, thumb_heat_map = self.process_tactile_data(self.thumb_tactile_sensor, (640, 480))

            with self.tac_thumb_lock:
                self.thumb_raw_img = thumb_raw_img
                self.thumb_points = thumb_points
                self.thumb_heat_map = thumb_heat_map
            time.sleep(0.01)


    def process_index_tactile(self):
        # Process index tactile data
        while True:
            index_raw_img, index_points, index_heat_map = self.process_tactile_data(self.index_tactile_sensor, (640, 480))

            with self.tac_index_lock:
                self.index_raw_img = index_raw_img
                self.index_points = index_points
                self.index_heat_map = index_heat_map
            time.sleep(0.01)
    

    def process_middle_tactile(self):
        # Process middle tactile data
        while True:
            middle_raw_img, middle_points, middle_heat_map = self.process_tactile_data(self.middle_tactile_sensor, (640, 480))

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

        self.cycle_count += 1
        if self.joint_reset_cycle!=0 and self.cycle_count % self.joint_reset_cycle == 0:
            self.cycle_count = 0
            joint_reset = True

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


    def _send_leap_hand_command(self, leap_hand_action: np.ndarray):
        """Internal function to send leap hand command to the robot."""
        hand_action = leap_hand_action
        step_time = 0.05  # Example step time
        steps = 5       # Example number of steps

        if self.interpolation_thread and self.interpolation_thread.is_alive():
            return

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
            self.ros_interface.publish_hand_action(interpolated_position)
            time.sleep(step_time)
            

    def _update_cur_position(self, arm_action, timeout=10.0):
        """
        Internal function to get the latest state of the robot and its gripper.
        """
        start = time.time()
        threshold = 0.002
        while np.linalg.norm(arm_action[:3] - self.cur_position) > threshold:
            position, orientation = self.ros_interface.get_current_robot_ee()
            joint_position = self.ros_interface.get_current_joint()
            # print(" before update cur_position = ", self.cur_position)
            self.cur_position = position
            self.cur_oritation = orientation
            # print("cur_position = ", self.cur_position)

            self.joint_position = joint_position

            # if not self.is_arm_only:
            hand_joint_msg = self.ros_interface.get_current_leap_position()
            self.curr_leap_hand_pos = np.array(hand_joint_msg)
            
            joint_pose = np.concatenate([
                    self.joint_position, self.curr_leap_hand_pos])
            
            self.joint_buffer.append(
                            copy.deepcopy(joint_pose))

            dist_close = np.linalg.norm(self.curr_leap_hand_pos - self.gripper_close_joint)

            if dist_close < 0.05:
                self.hand_state = 1
            else:
                self.hand_state = 0

            if time.time() - start > timeout:
                print("[WARN] 等待机械臂到位超时")
                break
            time.sleep(0.02)


    def _get_obs(self) -> dict:
        images = self.get_im()
        front_camera_image = images["front_camera"]
        # side_camera_image = images["side_camera"]

        if not self.is_arm_only:
            state_flattened = np.concatenate([
                np.array(self.cur_position, dtype=np.float32).flatten(),  # TCP 位置 (3,)
                np.array(self.cur_oritation, dtype=np.float32).flatten(),  # TCP 旋转 (4,)
                np.array(self.curr_leap_hand_pos, dtype=np.float32).flatten()  # 夹爪 (n,)
            ])
        else :
            state_flattened = np.concatenate([
                np.array(self.cur_position, dtype=np.float32).flatten(),  # TCP 位置 (3,)
                np.array(self.cur_oritation, dtype=np.float32).flatten(),  # TCP 旋转 (4,)
                np.array(self.hand_state, dtype=np.int32).flatten(),  # TCP 旋转 (4,)
            ])
        if self.enable_tactile:
            heatmap_canvas = cv2.hconcat([self.thumb_heat_map, self.index_heat_map, self.middle_heat_map])
            obs = copy.deepcopy({
            "front_camera": front_camera_image,
            # "side_camera": side_camera_image,
            "tactle_data":heatmap_canvas,
            "state": state_flattened
        })
        else:
            obs = copy.deepcopy({
            "front_camera": front_camera_image,
            # "side_camera": side_camera_image,
            "state": state_flattened
        })
            
        return obs


    def close(self):
        if hasattr(self, 'listener'):
            self.listener.stop()
        self.close_cameras()
        if self.display_image:
            self.img_queue.put(None)
            cv2.destroyAllWindows()
            self.displayer.join()


    def set_data_count(self, data_count):
        self.data_count = data_count + 1

    def pose_callback(self, msg):
        self.arm_position = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        self.arm_orientation = np.array([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
        # self.arm_pose = np.concatenate((position, orientation), axis=0)


    def save_training_frame(self):
        print("save_training_frame")
        try:
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

            if self.enable_tactile:
                cv2.imwrite(os.path.join(frame_dir, "thumb_raw_image.jpg"), self.rthumb_raw_buffer[frame_id])
                cv2.imwrite(os.path.join(frame_dir, "thumb_heat_map.jpg"), self.rthumb_heatmap_buffer[frame_id])
            
                cv2.imwrite(os.path.join(frame_dir, "index_raw_image.jpg"), self.rindex_raw_buffer[frame_id])
                cv2.imwrite(os.path.join(frame_dir, "index_heat_map.jpg"), self.rindex_heatmap_buffer[frame_id])
            
                cv2.imwrite(os.path.join(frame_dir, "middle_raw_image.jpg"), self.rmiddle_raw_buffer[frame_id])
                cv2.imwrite(os.path.join(frame_dir, "middle_heat_map.jpg"), self.rmiddle_heatmap_buffer[frame_id])
            # 保存 state（TCP + orientation + hand joints）
            # if not self.is_arm_only:
            np.savetxt(os.path.join(frame_dir, "right_arm_joint.txt"), self.joint_buffer[frame_id])