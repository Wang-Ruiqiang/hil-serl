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
        "front_camera": "130322274175",
        # "side_camera": "127122270572",
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
        self.cycle_count = 0

        robot_urdf_path = "/home/qiangqiang/workspaces/HK_TACTEXO_DATA/denso_robot_with_ati_4.urdf"
        self.data_count = 0
        self.data, _ = read_utils.read_data(robot_urdf_path,True)
        self._update_cur_position()

        if fake_env:
            return

        self.cap = None
        # self.init_cameras(config.REALSENSE_CAMERAS)
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
        # self.ros_param_init()

        print("Initialized Denso")


    # def ros_param_init(self):
    #     rclpy.init(args=None)
    #     self.node = rclpy.create_node("ros_publisher_node")

    #     self.publisher_arm = self.node.create_publisher(Float32MultiArray, 'wrist_cmd', 1)

    #     self.publisher_hand = self.node.create_publisher(Float32MultiArray, 'hand_cmd', 1)

    #     self.robot_subscription = self.node.create_subscription(
    #         PoseStamped,
    #         '/cartesian_compliance_controller/current_pose',
    #         self.pose_callback,
    #         10)
        
    #     # TODO:订阅leap_hand关节角话题
    

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
        # xyz_delta = action[:3]
        
        # arm_action = action[:7]
        # wrist_cmd = Float32MultiArray()
        # wrist_cmd.data = np.array(arm_action, dtype=np.float32).tolist()
        # # wrist_cmd.header.stamp = self.get_clock().now().to_msg()
        # self.publisher_arm.publish(wrist_cmd)

        # hand_action = action[7:]
        # hand_cmd = Float32MultiArray()
        # hand_cmd.data = np.array(hand_action, dtype=np.float32).tolist()
        # # hand_cmd.header.stamp = self.get_clock().now().to_msg()
        # self.publisher_hand.publish(hand_cmd)


        self.nextpos = np.concatenate([self.cur_position.copy(), np.zeros(4)])
        # self.nextpos[:3] = self.nextpos[:3] + xyz_delta * self.action_scale[0]
        self.nextpos[:3] = action[:3]

        # GET ORIENTATION FROM ACTION
        # self.nextpos[3:] = (
        #     Rotation.from_euler("xyz", action[3:6] * self.action_scale[1])
        #     * Rotation.from_quat(self.cur_position[3:])
        # ).as_quat()
        self.nextpos[3:] = action[3:7]

        self.curr_path_length += 1
        dt = time.time() - start_time
        time.sleep(max(0, (1.0 / self.hz) - dt))

        self._update_cur_position()
        ob = self._get_obs()
        # reward = self.compute_reward(ob)
        # done = self.curr_path_length >= self.max_episode_length or reward or self.terminate
        # return ob, int(reward), done, False, {"succeed": reward}
        done = 0
        return ob, 0, done, False, {"succeed": 0}

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

    # def get_im(self) -> Dict[str, np.ndarray]:
    #     """Get images from the realsense cameras."""
    #     images = {}
    #     display_images = {}
    #     full_res_images = {}  # New dictionary to store full resolution cropped images
    #     for key, cap in self.cap.items():
    #         try:
    #             rgb = cap.read()
    #             cropped_rgb = self.config.IMAGE_CROP[key](rgb) if key in self.config.IMAGE_CROP else rgb
    #             resized = cv2.resize(
    #                 cropped_rgb, self.observation_space["images"][key].shape[:2][::-1]
    #             )
    #             images[key] = resized[..., ::-1]
    #             display_images[key] = resized
    #             display_images[key + "_full"] = cropped_rgb
    #             full_res_images[key] = copy.deepcopy(cropped_rgb)  # Store the full resolution cropped image
    #         except queue.Empty:
    #             input(
    #                 f"{key} camera frozen. Check connect, then press enter to relaunch..."
    #             )
    #             cap.close()
    #             self.init_cameras(self.config.REALSENSE_CAMERAS)
    #             return self.get_im()

    #     # Store full resolution cropped images separately
    #     if self.save_video:
    #         self.recording_frames.append(full_res_images)

    #     if self.display_image:
    #         self.img_queue.put(display_images)
    #     return images
    
    def get_im_test(self) -> Dict[str, np.ndarray]:
        """Get images from the realsense cameras."""
        images_dict = {} 
        obs = self.data[self.data_count]["observations"]
        for key in obs:
            if key == "state":
                continue
            img = np.array(obs[key])

            images_dict[key] = copy.deepcopy(img)

        return images_dict

    def interpolate_move(self, goal: np.ndarray, timeout: float):
        """Move the robot to the goal position with linear interpolation."""
        if goal.shape == (6,):
            goal = np.concatenate([goal[:3], euler_2_quat(goal[3:])])
        steps = int(timeout * self.hz)
        self._update_cur_position()
        path = np.linspace(self.cur_position, goal, steps)
        for p in path:
            self._send_pos_command(p)
            time.sleep(1 / self.hz)
        self.nextpos = p
        self._update_cur_position()

    def go_to_reset(self, joint_reset=False):
        """
        The concrete steps to perform reset should be
        implemented each subclass for the specific task.
        Should override this method if custom reset procedure is needed.
        """
        # Change to precision mode for reset        # Use compliance mode for coupled reset
        self._update_cur_position()
        self._send_pos_command(self.cur_position)
        time.sleep(0.3)
        requests.post(self.url + "update_param", json=self.config.PRECISION_PARAM)
        time.sleep(0.5)

        # Perform joint reset if needed
        if joint_reset:
            print("JOINT RESET")
            requests.post(self.url + "jointreset")
            time.sleep(0.5)

        # Perform Carteasian reset
        if self.randomreset:  # randomize reset position in xy plane
            reset_pose = self.resetpos.copy()
            reset_pose[:2] += np.random.uniform(
                -self.random_xy_range, self.random_xy_range, (2,)
            )
            euler_random = self._RESET_POSE[3:].copy()
            euler_random[-1] += np.random.uniform(
                -self.random_rz_range, self.random_rz_range
            )
            reset_pose[3:] = euler_2_quat(euler_random)
            self.interpolate_move(reset_pose, timeout=1)
        else:
            reset_pose = self.resetpos.copy()
            self.interpolate_move(reset_pose, timeout=1)

        # Change to compliance mode
        requests.post(self.url + "update_param", json=self.config.COMPLIANCE_PARAM)

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

    # def init_cameras(self, name_serial_dict=None):
    #     """Init both wrist cameras."""
    #     if self.cap is not None:  # close cameras if they are already open
    #         self.close_cameras()

    #     self.cap = OrderedDict()
    #     for cam_name, kwargs in name_serial_dict.items():
    #         cap = VideoCapture(
    #             RSCapture(name=cam_name, **kwargs)
    #         )
    #         self.cap[cam_name] = cap

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

    def _send_gripper_command(self, pos: float, mode="binary"):
        """Internal function to send gripper command to the robot."""
        if mode == "binary":
            if (pos <= -0.5) and (self.curr_gripper_pos > 0.85) and (time.time() - self.last_gripper_act > self.gripper_sleep):  # close gripper
                requests.post(self.url + "close_gripper")
                self.last_gripper_act = time.time()
                time.sleep(self.gripper_sleep)
            elif (pos >= 0.5) and (self.curr_gripper_pos < 0.85) and (time.time() - self.last_gripper_act > self.gripper_sleep):  # open gripper
                requests.post(self.url + "open_gripper")
                self.last_gripper_act = time.time()
                time.sleep(self.gripper_sleep)
            else: 
                return
        elif mode == "continuous":
            raise NotImplementedError("Continuous gripper control is optional")

    def _update_cur_position(self):
        """
        Internal function to get the latest state of the robot and its gripper.
        """
        # self.spin_ros()
        self.cur_position = self.data[self.data_count]["observations"]["state"][:3]
        self.cur_oritation = self.data[self.data_count]["observations"]["state"][3:7]
        # self.cur_position = self.arm_position
        # self.cur_oritation = self.arm_orientation

        # TODO:保存leap_hand当前关节角
        # self.curr_gripper_pos = self.data[self.data_count]["observations"]["state"][7:]

    # def update_cur_position(self):
    #     """
    #     Internal function to get the latest state of the robot and its gripper.
    #     """
    #     ps = requests.post(self.url + "getstate").json()
    #     self.cur_position = np.array(ps["pose"])
    #     self.currvel = np.array(ps["vel"])

    #     self.currforce = np.array(ps["force"])
    #     self.currtorque = np.array(ps["torque"])
    #     self.currjacobian = np.reshape(np.array(ps["jacobian"]), (6, 7))

    #     self.q = np.array(ps["q"])
    #     self.dq = np.array(ps["dq"])

    #     self.curr_gripper_pos = np.array(ps["gripper_pos"])

    def _get_obs(self) -> dict:
        images = self.get_im_test()
        front_camera_image = images["front_camera"]
        # side_camera_image = images["side_camera"]
        state_flattened = np.concatenate([
            np.array(self.cur_position, dtype=np.float32).flatten(),  # TCP 位置 (3,)
            np.array(self.cur_oritation, dtype=np.float32).flatten(),  # TCP 旋转 (4,)
            # np.array(self.curr_gripper_pos, dtype=np.float32).flatten()  # 夹爪 (n,)
        ])
        # state_observation = {
        #     "tcp_pos": self.cur_position,
        #     "tcp_ori": self.cur_oritation,
        #     "gripper_pose": self.curr_gripper_pos,
        # }
        return copy.deepcopy({
            "front_camera": front_camera_image,
            # "side_camera": side_camera_image,
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