#!/usr/bin/env python3
"""Check FR3 feedback and test joint-target/EE-pose mode switching.

Default mode only observes /franka/joint_states and does not move the robot.
Use --sequence to run a repeatable joint-target reset / EE-pose Servo cycle.
"""

import argparse
import copy
import threading
import time

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


JOINT_NAMES = [
    "fr3_joint1",
    "fr3_joint2",
    "fr3_joint3",
    "fr3_joint4",
    "fr3_joint5",
    "fr3_joint6",
    "fr3_joint7",
]

RESET_JOINT = np.array(
    [
        -0.18156941995945866,
        0.1296803145320053,
        0.0,
        -1.9407152154140208,
        0.0,
        2.1285232352375885,
        -0.23116278766225817,
    ],
    dtype=np.float32,
)


class JointServoFeedbackTest(Node):
    def __init__(self):
        super().__init__("joint_servo_feedback_test")
        self._lock = threading.Lock()
        self.latest_joint = None
        self.latest_pose = None
        self.last_stamp = None
        self.message_count = 0

        self.create_subscription(
            JointState,
            "/franka/joint_states",
            self._joint_state_callback,
            10,
        )
        self.create_subscription(
            PoseStamped,
            "/franka_robot_state_broadcaster/current_pose",
            self._pose_callback,
            10,
        )
        self.pose_target_pub = self.create_publisher(
            PoseStamped,
            "/target_pose",
            10,
        )
        self.trajectory_pub = self.create_publisher(
            JointTrajectory,
            "/joint_pose",
            10,
        )

    def log_control_connections(self):
        """Print the connections needed by the external pose-tracking node."""
        self.get_logger().info(
            "control connections "
            f"/joint_pose(pub={self.count_publishers('/joint_pose')}, "
            f"sub={self.count_subscribers('/joint_pose')}) "
            f"/target_pose(pub={self.count_publishers('/target_pose')}, "
            f"sub={self.count_subscribers('/target_pose')})"
        )

    def _joint_state_callback(self, msg):
        values = dict(zip(msg.name, msg.position))
        missing = [name for name in JOINT_NAMES if name not in values]
        if missing:
            self.get_logger().warning(
                f"Missing joints: {missing}; received names={list(msg.name)}"
            )
            return False

        ordered = np.asarray([values[name] for name in JOINT_NAMES], dtype=np.float64)
        now = time.monotonic()
        with self._lock:
            self.latest_joint = ordered
            self.last_stamp = now
            self.message_count += 1

    def _pose_callback(self, msg):
        with self._lock:
            self.latest_pose = msg

    def wait_for_feedback(self, timeout=5.0):
        with self._lock:
            previous_count = self.message_count
        deadline = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < deadline:
            with self._lock:
                if (
                    self.latest_joint is not None
                    and self.message_count > previous_count
                ):
                    return self.latest_joint.copy()
            rclpy.spin_once(self, timeout_sec=0.05)
        raise RuntimeError(
            "No fresh /franka/joint_states message received within the timeout."
        )

    def wait_for_pose(self, timeout=5.0):
        deadline = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < deadline:
            with self._lock:
                if self.latest_pose is not None:
                    return self.latest_pose
            rclpy.spin_once(self, timeout_sec=0.05)
        raise RuntimeError(
            "No usable /franka_robot_state_broadcaster/current_pose message received."
        )

    def publish_pose_target(self, pose):
        target = PoseStamped()
        target.header = pose.header
        target.header.stamp = self.get_clock().now().to_msg()
        target.pose = pose.pose
        self.pose_target_pub.publish(target)

    def publish_interpolated_pose(self, start_pose, target_pose, step_time=0.007, steps=10):
        """Publish EE targets using the same position/Slerp interpolation as RL."""
        start_pos = np.array(
            [start_pose.pose.position.x, start_pose.pose.position.y, start_pose.pose.position.z],
            dtype=np.float64,
        )
        target_pos = np.array(
            [target_pose.pose.position.x, target_pose.pose.position.y, target_pose.pose.position.z],
            dtype=np.float64,
        )

        start_quat_xyzw = np.array(
            [
                start_pose.pose.orientation.x,
                start_pose.pose.orientation.y,
                start_pose.pose.orientation.z,
                start_pose.pose.orientation.w,
            ],
            dtype=np.float64,
        )
        target_quat_xyzw = np.array(
            [
                target_pose.pose.orientation.x,
                target_pose.pose.orientation.y,
                target_pose.pose.orientation.z,
                target_pose.pose.orientation.w,
            ],
            dtype=np.float64,
        )
        slerp = Slerp([0.0, 1.0], R.from_quat([start_quat_xyzw, target_quat_xyzw]))

        steps = max(1, int(steps))

        # ==================== EE pose 插值位置 ====================
        # 位置使用线性插值，姿态使用四元数 Slerp，然后逐点发布
        # /target_pose。RL 中对应的是 arm_interpolate_and_publish()。
        for index in range(steps + 1):
            alpha = index / steps
            rotation = slerp([alpha])[0].as_quat()
            target = PoseStamped()
            target.header = target_pose.header
            target.pose.position.x = float(start_pos[0] + alpha * (target_pos[0] - start_pos[0]))
            target.pose.position.y = float(start_pos[1] + alpha * (target_pos[1] - start_pos[1]))
            target.pose.position.z = float(start_pos[2] + alpha * (target_pos[2] - start_pos[2]))
            target.pose.orientation.x = float(rotation[0])
            target.pose.orientation.y = float(rotation[1])
            target.pose.orientation.z = float(rotation[2])
            target.pose.orientation.w = float(rotation[3])
            self.publish_pose_target(target)
            if index < steps:
                time.sleep(float(step_time))

    def publish_joint_trajectory(self, start_joint, target_joint, duration):
        start_joint = np.asarray(start_joint, dtype=np.float64).reshape(7)
        target_joint = np.asarray(target_joint, dtype=np.float64).reshape(7)
        duration = max(float(duration), 0.1)

        trajectory = JointTrajectory()
        trajectory.header.stamp = self.get_clock().now().to_msg()
        trajectory.joint_names = list(JOINT_NAMES)

        # ==================== Joint trajectory 插值位置 ====================
        # 先在测试脚本中按 0.01s 生成多个关节角中间点，再一次性发送到
        # /joint_pose。Servo 节点收到后还会继续执行
        # Butterworth 平滑。
        joint_interpolation_step = 0.01
        steps = max(2, int(np.ceil(duration / joint_interpolation_step)))
        for index in range(steps + 1):
            alpha = index / steps
            point = JointTrajectoryPoint()
            point.positions = (
                start_joint + alpha * (target_joint - start_joint)
            ).tolist()
            elapsed = duration * alpha
            point.time_from_start = Duration(
                sec=int(np.floor(elapsed)),
                nanosec=int((elapsed - np.floor(elapsed)) * 1e9),
            )
            if index == 0:
                point.time_from_start = Duration(sec=0, nanosec=1)
            trajectory.points.append(point)
        self.trajectory_pub.publish(trajectory)
        self.get_logger().info(
            "joint target published start=%s target=%s duration=%.2fs"
            % (np.round(start_joint, 6), np.round(target_joint, 6), duration)
        )

    def wait_for_joint_target(
        self,
        target_joint,
        timeout,
        tolerance,
        minimum_wait=0.0,
    ):
        target_joint = np.asarray(target_joint, dtype=np.float64).reshape(7)
        start_time = time.monotonic()
        deadline = start_time + float(timeout)
        minimum_deadline = start_time + float(minimum_wait)
        last_current = None
        while rclpy.ok() and time.monotonic() < deadline:
            last_current = self.wait_for_feedback(timeout=1.0)
            error = target_joint - last_current
            if (
                time.monotonic() >= minimum_deadline
                and float(np.max(np.abs(error))) <= float(tolerance)
            ):
                self.get_logger().info(
                    "joint target reached error=%s" % np.round(error, 6)
                )
                return True, last_current
            rclpy.spin_once(self, timeout_sec=0.02)

        if last_current is None:
            last_current = self.wait_for_feedback(timeout=1.0)
        error = target_joint - last_current
        self.get_logger().warning(
            "joint target not reached current=%s target=%s error=%s"
            % (
                np.round(last_current, 6),
                np.round(target_joint, 6),
                np.round(error, 6),
            )
        )
        return False, last_current


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sequence",
        "--move",
        dest="sequence",
        action="store_true",
        help="Run a repeatable reset -> move -> Enter -> reset -> Enter -> move cycle.",
    )
    parser.add_argument("--duration", type=float, default=4.0,
                        help="Joint-target reset trajectory duration in seconds.")
    parser.add_argument(
        "--wait-time",
        type=float,
        default=5.0,
        help="Maximum time to wait for each joint-target reset.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.03,
        help="Maximum absolute joint error considered reached, in radians.",
    )
    parser.add_argument(
        "--pose-offset-z",
        type=float,
        default=0.03,
        help="EE z offset used during the pose-tracking Servo test, in meters.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rclpy.init()
    node = JointServoFeedbackTest()
    try:
        # This is the first diagnostic to check if a command appears to have
        # no effect. The pose-tracking node must subscribe to both inputs.
        node.log_control_connections()
        initial = node.wait_for_feedback()
        node.get_logger().info(
            "initial feedback names=%s values=%s"
            % (JOINT_NAMES, np.round(initial, 6))
        )

        if not args.sequence:
            node.get_logger().info(
                "Feedback-only test passed. Use --sequence to run the mode-switch test."
            )
            deadline = time.monotonic() + 3.0
            while rclpy.ok() and time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=0.1)
            return

        def reset_arm(reset_index):
            # A fresh sample is important after input(): do not reuse the
            # cached joint state from before the previous mode switch.
            start_joint = node.wait_for_feedback()
            node.get_logger().info(
                "reset %d start_joint=%s" % (reset_index, np.round(start_joint, 6))
            )
            node.publish_joint_trajectory(start_joint, RESET_JOINT, args.duration)
            reached, end_joint = node.wait_for_joint_target(
                RESET_JOINT,
                args.wait_time,
                args.tolerance,
                minimum_wait=args.duration,
            )
            node.get_logger().info(
                "reset %d end_joint=%s" % (reset_index, np.round(end_joint, 6))
            )
            if not reached:
                raise RuntimeError(
                    f"Joint-target reset {reset_index} did not reach RESET_JOINT."
                )

        def move_ee(move_index):
            # A new target_pose switches the same pose-tracking node back to
            # EE-pose mode after the preceding joint reset.
            pose_start = node.wait_for_pose()
            pose_target = PoseStamped()
            pose_target.header = pose_start.header
            pose_target.pose = copy.deepcopy(pose_start.pose)
            pose_target.pose.position.z += float(args.pose_offset_z)
            node.get_logger().info(
                "move %d EE-pose start_xyz=%s target_xyz=%s"
                % (
                    move_index,
                    np.round(
                        [
                            pose_start.pose.position.x,
                            pose_start.pose.position.y,
                            pose_start.pose.position.z,
                        ],
                        4,
                    ),
                    np.round(
                        [
                            pose_target.pose.position.x,
                            pose_target.pose.position.y,
                            pose_target.pose.position.z,
                        ],
                        4,
                    ),
                )
            )
            node.publish_interpolated_pose(
                pose_start,
                pose_target,
                step_time=0.007,
                steps=10,
            )
            node.get_logger().info(
                "move %d published; pose tracking keeps the latest target."
                % move_index
            )

        # The first cycle preserves the old behavior: reset, then move.
        reset_index = 1
        move_index = 1
        reset_arm(reset_index)
        move_ee(move_index)

        # Thereafter each Enter advances exactly one phase:
        # move -> reset -> move -> reset -> ... until Ctrl+C.
        while rclpy.ok():
            input(
                "Press Enter to reset the arm "
                f"(reset {reset_index + 1}), or Ctrl+C to exit. "
            )
            reset_index += 1
            reset_arm(reset_index)

            input(
                "Press Enter to move the EE "
                f"(move {move_index + 1}), or Ctrl+C to exit. "
            )
            move_index += 1
            move_ee(move_index)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
