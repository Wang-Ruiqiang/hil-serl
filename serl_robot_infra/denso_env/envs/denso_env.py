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
from typing import Dict

from franka_env.utils.rotations import euler_2_quat, quat_2_euler

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
import threading
from denso_env.camera.video_capture import VideoCapture
from denso_env.camera.rs_capture import RSCapture
from leap_hand.srv import LeapPosition, LeapPosVelEff

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
        "side_camera": "234222300515",
    }
    IMAGE_CROP: dict[str, callable] = {}
    TARGET_POSE: np.ndarray = np.zeros((6,))
    GRASP_POSE: np.ndarray = np.zeros((6,))
    REWARD_THRESHOLD: np.ndarray = np.zeros((6,))
    ACTION_SCALE = np.zeros((3,))
    RESET_POSE = np.zeros((6,))
    RANDOM_RESET = False
    RANDOM_XY_RANGE = (0.0,)
    RANDOM_RZ_RANGE = (0.0,)
    ABS_POSE_LIMIT_HIGH = np.zeros((6,))
    ABS_POSE_LIMIT_LOW = np.zeros((6,))
    COMPLIANCE_PARAM: Dict[str, float] = {}
    RESET_PARAM: Dict[str, float] = {}
    PRECISION_PARAM: Dict[str, float] = {}
    LOAD_PARAM: Dict[str, float] = {
        "mass": 0.0,
        "F_x_center_load": [0.0, 0.0, 0.0],
        "load_inertia": [0, 0, 0, 0, 0, 0, 0, 0, 0]
    }
    DISPLAY_IMAGE: bool = True
    GRIPPER_SLEEP: float = 0.6
    MAX_EPISODE_LENGTH: int = 100
    JOINT_RESET_PERIOD: int = 0


##############################################################################

class ROSNodeInterface(Node):
    def __init__(self):
        super().__init__('denso_env_node')

        # Publishers（发送）
        self.arm_pub = self.create_publisher(
            PoseStamped,
            '/cloth_folding/robot_control',
            10
        )

        self.publisher_hand = self.create_publisher(
            JointState, 
            '/cmd_leap', 
            10
        )

        # Subscribers（接收）
        self.joint_sub = self.create_subscription(
            PoseStamped,
            '/cartesian_compliance_controller/current_pose',  # 接收机械臂关节角
            self.joint_callback,
            10
        )
        
        self.leap_position_client = self.create_client(LeapPosition, '/leap_position')

        # Wait for the service to be available
        while not self.leap_position_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().info('Waiting for /leap_position service...')

        # 同步事件
        self.joint_event = threading.Event()
        self.hand_joint_event = threading.Event()

        # 数据存储
        self.current_joints = None
        self.current_hand_joints = None

    def joint_callback(self, msg):
        #用ros2 INFO打印接收到的数据 
        # self.get_logger().info("ROS回调已接收到机械臂位姿数据")
        position = msg.pose.position
        self.cur_position = np.array([position.x, position.y, position.z])

        # 从 msg 中提取四元数方向数据（xyzw）
        orientation = msg.pose.orientation
        self.cur_oritation = np.array([
            orientation.x, orientation.y, orientation.z, orientation.w
        ])

        # 设置事件为已收到数据
        self.joint_event.set()

    def publish_arm_action(self, pose):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = pose[:3]
        msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w = pose[3:7]
        self.arm_pub.publish(msg)


    def publish_hand_action(self, hand_joints):
        stater = JointState()
        stater.name = [f"joint_{i}" for i in range(len(hand_joints))]
        stater.position = hand_joints
        self.publisher_hand.publish(stater)
        

    def get_current_joints(self, timeout=5.0):
        # success = self.joint_event.wait(timeout=timeout)
        # if not success:
        #     raise TimeoutError("等待机械臂数据超时，请检查ROS话题是否正常发布。")
        self.joint_event.wait()
        self.joint_event.clear()
        return self.cur_position, self.cur_oritation

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
        self._RESET_POSE = config.RESET_POSE
        self._REWARD_THRESHOLD = config.REWARD_THRESHOLD
        self.url = config.SERVER_URL
        self.config = config
        self.max_episode_length = config.MAX_EPISODE_LENGTH
        self.display_image = config.DISPLAY_IMAGE
        self.gripper_sleep = config.GRIPPER_SLEEP



        # convert last 3 elements from euler to quat, from size (6,) to (7,)
        self.resetpos = np.concatenate(
            [config.RESET_POSE[:3], euler_2_quat(config.RESET_POSE[3:])]
        )
        self.last_gripper_act = time.time()
        self.lastsent = time.time()
        self.randomreset = config.RANDOM_RESET
        self.random_xy_range = config.RANDOM_XY_RANGE
        self.random_rz_range = config.RANDOM_RZ_RANGE
        self.hz = hz
        self.joint_reset_cycle = config.JOINT_RESET_PERIOD  # reset the robot joint every 200 cycles

        self.save_video = save_video
        if self.save_video:
            print("Saving videos!")
            self.recording_frames = []

        # boundary box
        self.xyz_bounding_box = gym.spaces.Box(
            config.ABS_POSE_LIMIT_LOW[:3],
            config.ABS_POSE_LIMIT_HIGH[:3],
            dtype=np.float64,
        )
        self.rpy_bounding_box = gym.spaces.Box(
            config.ABS_POSE_LIMIT_LOW[3:],
            config.ABS_POSE_LIMIT_HIGH[3:],
            dtype=np.float64,
        )
        # Action/Observation Space
        self.action_space = gym.spaces.Box(
            np.ones((23,), dtype=np.float32) * -1,
            np.ones((23,), dtype=np.float32),
        )

        # self.action_space = gym.spaces.Box(
        #     np.ones((7,), dtype=np.float32) * -1,
        #     np.ones((7,), dtype=np.float32),
        # )
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
        
        self.hand_joint_offset = np.array([3.14, 3.14, 3.14, 3.14,
            3.14, 3.14, 3.14, 0,
            3.14, 3.14, 3.14, 3.14,
            3.14, 3.14, 3.14, 3.14
        ] )
        
        self.cycle_count = 0

        # robot_urdf_path = "/home/qiangqiang/workspaces/HK_TACTEXO_DATA/denso_robot_with_ati_4.urdf"
        # self.data_count = 0
        # self.data = read_utils.read_data(robot_urdf_path,True)
        # self._update_cur_position()

        if fake_env:
            return

        self.cap = None
        self.init_cameras(config.REALSENSE_CAMERAS)
        if self.display_image:
            self.img_queue = queue.Queue()
            self.displayer = ImageDisplayer(self.img_queue, self.url)
            self.displayer.start()

        if set_load:
            input("Put arm into programing mode and press enter.")
            requests.post(self.url + "set_load", json=self.config.LOAD_PARAM)
            input("Put arm into execution mode and press enter.")
            for _ in range(2):
                self._recover()
                time.sleep(1)

        # if not fake_env:
        #     from pynput import keyboard
        #     self.terminate = False
        #     def on_press(key):
        #         if key == keyboard.Key.esc:
        #             self.terminate = True
        #     self.listener = keyboard.Listener(on_press=on_press)
        #     self.listener.start()

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

        print("Initialized Denso")
    

    def clip_safety_box(self, pose: np.ndarray) -> np.ndarray:
        """Clip the pose to be within the safety box."""
        pose[:3] = np.clip(
            pose[:3], self.xyz_bounding_box.low, self.xyz_bounding_box.high
        )
        euler = Rotation.from_quat(pose[3:]).as_euler("xyz")

        # Clip first euler angle separately due to discontinuity from pi to -pi
        sign = np.sign(euler[0])
        euler[0] = sign * (
            np.clip(
                np.abs(euler[0]),
                self.rpy_bounding_box.low[0],
                self.rpy_bounding_box.high[0],
            )
        )

        euler[1:] = np.clip(
            euler[1:], self.rpy_bounding_box.low[1:], self.rpy_bounding_box.high[1:]
        )
        pose[3:] = Rotation.from_euler("xyz", euler).as_quat()

        return pose

    def step(self, action: np.ndarray) -> tuple:
        """standard gym step function."""
        start_time = time.time()
        # action = np.clip(action, self.action_space.low, self.action_space.high)


        # 前7维为机械臂位姿
        arm_action = action[:7]
        # self.ros_interface.publish_arm_action(arm_action)

        # 后16维为灵巧手关节角
        leap_hand_action = action[7:]
        # self.ros_interface.publish_hand_action(hand_action)
        self._send_leap_hand_command(leap_hand_action)

        input("debug")

        self.curr_path_length += 1
        dt = time.time() - start_time
        time.sleep(max(0, (1.0 / self.hz) - dt))

        self._update_cur_position()
        ob = self._get_obs()
        reward = self.compute_reward(ob)
        done = self.curr_path_length >= self.max_episode_length or reward or self.terminate
        return ob, int(reward), done, False, {"succeed": reward}
    

    def compute_reward(self, obs) -> bool:
        current_pose = obs["state"]
        # convert from quat to euler first
        current_rot = Rotation.from_quat(current_pose[3:7]).as_matrix()
        target_rot = Rotation.from_euler("xyz", self._TARGET_POSE[3:]).as_matrix()
        diff_rot = current_rot.T  @ target_rot
        diff_euler = Rotation.from_matrix(diff_rot).as_euler("xyz")
        delta = np.abs(np.hstack([current_pose[:3] - self._TARGET_POSE[:3], diff_euler]))
        # print(f"Delta: {delta}")
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
            try:
                rgb = cap.read()
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
                self.init_cameras(self.config.REALSENSE_CAMERAS)
                return self.get_im()

        # Store full resolution cropped images separately
        if self.save_video:
            self.recording_frames.append(full_res_images)

        if self.display_image:
            self.img_queue.put(display_images)
        return images
    

    def reset(self, joint_reset=False, **kwargs):
        self.data_count = 0
        self.last_gripper_act = time.time()
        requests.post(self.url + "update_param", json=self.config.COMPLIANCE_PARAM)
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

    def init_cameras(self, name_serial_dict=None):
        """Init both wrist cameras."""
        if self.cap is not None:  # close cameras if they are already open
            self.close_cameras()

        self.cap = OrderedDict()
        for cam_name, kwargs in name_serial_dict.items():
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

    def _send_pos_command(self, pos: np.ndarray):
        """Internal function to send position command to the robot."""
        self._recover()
        arr = np.array(pos).astype(np.float32)
        data = {"arr": arr.tolist()}
        requests.post(self.url + "pose", json=data)

    def _send_leap_hand_command(self, leap_hand_action: np.ndarray):
        """Internal function to send leap hand command to the robot."""
        index = leap_hand_action[:4]
        index[:2] = index[1::-1]    # Flip the first two elements 
        thumb = leap_hand_action[4:8]
        middle = leap_hand_action[8:12]
        middle[:2] = middle[1::-1]
        ring = leap_hand_action[12:]
        ring[:2] = ring[1::-1]

        leap_hand_j_cmd = np.concatenate((index, middle, ring, thumb))
        hand_action = leap_hand_j_cmd + self.hand_joint_offset
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
            

    def _update_cur_position(self):
        """
        Internal function to get the latest state of the robot and its gripper.
        """
        position, orientation = self.ros_interface.get_current_joints()
        self.cur_position = position
        self.cur_oritation = orientation
        hand_joint_msg = self.ros_interface.get_current_leap_position()
        print("hand_joint_msg = ", hand_joint_msg)

        if hand_joint_msg is not None:
            self.curr_leap_hand_pos = np.array(hand_joint_msg)
        else:
            print("Warning: Hand joint data unavailable.")


    def _get_obs(self) -> dict:
        images = self.get_im()
        front_camera_image = images["front_camera"]
        side_camera_image = images["side_camera"]
        state_flattened = np.concatenate([
            np.array(self.cur_position, dtype=np.float32).flatten(),  # TCP 位置 (3,)
            np.array(self.cur_oritation, dtype=np.float32).flatten(),  # TCP 旋转 (4,)
            np.array(self.curr_leap_hand_pos, dtype=np.float32).flatten()  # 夹爪 (n,)
        ])

        return copy.deepcopy({
            "front_camera": front_camera_image,
            "side_camera": side_camera_image,
            "state": state_flattened
        })

    def close(self):
        if hasattr(self, 'listener'):
            self.listener.stop()
        self.close_cameras()
        if self.display_image:
            self.img_queue.put(None)
            cv2.destroyAllWindows()
            self.displayer.join()


    # def spin_ros(self):
    #     rclpy.spin_once(self.node, timeout_sec=0.1)  # 每次只 spin 一次，这样可以在任务中间运行 ROS
    #     time.sleep(1)

    def set_data_count(self, data_count):
        self.data_count = data_count + 1

    def pose_callback(self, msg):
        self.arm_position = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        self.arm_orientation = np.array([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
        # self.arm_pose = np.concatenate((position, orientation), axis=0)