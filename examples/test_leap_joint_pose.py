#!/usr/bin/env python3

import argparse
import time

import numpy as np
import rclpy
from leap_hand.srv import LeapPosition
from rclpy.node import Node
from sensor_msgs.msg import JointState


# 1-4 index, 5-8 middle, 9-12 ring, 13-16 thumb.
# Script uses 0-based indices: 0-3 index, 4-7 middle, 8-11 ring, 12-15 thumb.
MODIFIED_GRIPPER_CLOSE_JOINT = np.array([
    3.225961685180664062, 4.575495624542236328, 2.794913053512573242, 3.514874553680419922,
    3.146330614089965820, 3.529689788818359375, 3.438389015197753906, 3.969689750671386719,
    3.218291759490966797, 3.238233327865600586, 2.867010116577148438, 3.325670242309570312,
    4.674039363861083984, 3.307262659072875977, 2.844000339508056641, 3.736408138275146484,
], dtype=np.float32)
MODIFIED_GRIPPER_CLOSE_JOINT
# Edit this pose directly, then run the script to compare it with GRIPPER_CLOSE_JOINT.
GRIPPER_CLOSE_JOINT = np.array([
    3.225961685180664062, 4.575495624542236328, 2.794913053512573242, 4.114874553680419922,
    3.146330614089965820, 3.529689788818359375, 3.438389015197753906, 3.969689750671386719,
    3.218291759490966797, 3.238233327865600586, 2.867010116577148438, 3.325670242309570312,
    4.674039363861083984, 3.307262659072875977, 2.844000339508056641, 3.736408138275146484,
], dtype=np.float32)




GRIPPER_OPEN_JOINT = np.array([
    3.173806190490722656, 4.548253059387207031, 2.851670265197753906, 3.597184896469116211,
    3.146330614089965820, 3.529689788818359375, 3.438389015197753906, 3.969689750671386719,
    3.218291759490966797, 3.238233327865600586, 2.867010116577148438, 3.325670242309570312,
    4.703185081481933594, 3.296524763107299805, 2.872485685348510742, 3.718369483947753906,
], dtype=np.float32)

POSES = {
    "close": (GRIPPER_CLOSE_JOINT, MODIFIED_GRIPPER_CLOSE_JOINT),
    "open": (GRIPPER_OPEN_JOINT, GRIPPER_OPEN_JOINT),
}


class LeapPosePublisher(Node):
    def __init__(self):
        super().__init__("test_leap_joint_pose")
        self.publisher = self.create_publisher(JointState, "/cmd_leap", 10)
        self.leap_position_client = self.create_client(LeapPosition, "/leap_position")

    def publish_pose(self, pose):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        joints = np.asarray(pose, dtype=np.float64).reshape(-1)
        msg.name = [f"joint_{i}" for i in range(len(joints))]
        msg.position = [float(v) for v in joints]
        self.publisher.publish(msg)

    def get_current_pose(self, timeout=2.0):
        if not self.leap_position_client.wait_for_service(timeout_sec=timeout):
            print("Warning: /leap_position service is not available.")
            return None

        future = self.leap_position_client.call_async(LeapPosition.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        if future.result() is None:
            print("Warning: failed to read current Leap hand pose.")
            return None

        pose = np.asarray(list(future.result().position), dtype=np.float32)
        if pose.shape != (16,):
            print(f"Warning: expected current pose shape (16,), got {pose.shape}.")
            return None
        return pose


def _send_for_duration(node, pose, duration, hz):
    period = 1.0 / hz
    end_time = time.time() + duration
    while time.time() < end_time:
        node.publish_pose(pose)
        rclpy.spin_once(node, timeout_sec=0.0)
        time.sleep(period)


def _send_interpolated(node, start_pose, target_pose, duration, hz):
    start_pose = np.asarray(start_pose, dtype=np.float32)
    target_pose = np.asarray(target_pose, dtype=np.float32)
    n_steps = max(1, int(duration * hz))
    period = 1.0 / hz

    for i in range(n_steps + 1):
        alpha = i / n_steps
        pose = (1.0 - alpha) * start_pose + alpha * target_pose
        node.publish_pose(pose)
        rclpy.spin_once(node, timeout_sec=0.0)
        time.sleep(period)


def main():
    parser = argparse.ArgumentParser(
        description="Send original and modified Leap hand poses for visual inspection."
    )
    parser.add_argument("--pose", default="close", choices=sorted(POSES), help="Pose pair to test.")
    parser.add_argument("--hold", type=float, default=3.0, help="Seconds to keep publishing each pose.")
    parser.add_argument("--move_time", type=float, default=3.0, help="Seconds for interpolated movement.")
    parser.add_argument("--pause", type=float, default=2.0, help="Seconds to wait between poses.")
    parser.add_argument("--hz", type=float, default=20.0, help="Publish frequency.")
    args = parser.parse_args()

    base_pose, modified_pose = POSES[args.pose]
    base_pose = np.asarray(base_pose, dtype=np.float32)
    modified_pose = np.asarray(modified_pose, dtype=np.float32)

    if base_pose.shape != (16,) or modified_pose.shape != (16,):
        raise ValueError(
            f"Pose arrays must have shape (16,), got {base_pose.shape} and {modified_pose.shape}."
        )

    diff = modified_pose - base_pose
    changed_indices = np.where(np.abs(diff) > 1e-6)[0]

    print(f"pose pair: {args.pose}")
    if len(changed_indices) == 0:
        print("No modified joints detected. Edit MODIFIED_GRIPPER_CLOSE_JOINT to test changes.")
    else:
        print("modified joints:")
        for idx in changed_indices:
            print(
                f"  joint_{idx}: {base_pose[idx]:.6f} -> {modified_pose[idx]:.6f} "
                f"({diff[idx]:+.6f})"
            )
    print("")
    print("About to send ORIGINAL pose.")
    input("Press Enter to start...")

    rclpy.init()
    node = LeapPosePublisher()
    try:
        start_pose = node.get_current_pose()
        if start_pose is None:
            start_pose = GRIPPER_OPEN_JOINT.copy()
            print("Using GRIPPER_OPEN_JOINT as interpolation start pose.")

        print(f"Moving to OPEN pose over {args.move_time} seconds.")
        _send_interpolated(node, start_pose, GRIPPER_OPEN_JOINT, args.move_time, args.hz)

        print(f"Holding OPEN pose for {args.hold} seconds.")
        _send_for_duration(node, GRIPPER_OPEN_JOINT, args.hold, args.hz)

        print(f"Waiting {args.pause} seconds before original pose.")
        time.sleep(args.pause)

        print(f"Moving to ORIGINAL pose over {args.move_time} seconds.")
        _send_interpolated(node, GRIPPER_OPEN_JOINT, base_pose, args.move_time, args.hz)

        print(f"Holding ORIGINAL pose for {args.hold} seconds.")
        _send_for_duration(node, base_pose, args.hold, args.hz)

        print(f"Waiting {args.pause} seconds before modified pose.")
        time.sleep(args.pause)

        print(f"Moving to MODIFIED pose over {args.move_time} seconds.")
        _send_interpolated(node, base_pose, modified_pose, args.move_time, args.hz)

        print(f"Holding MODIFIED pose for {args.hold} seconds.")
        _send_for_duration(node, modified_pose, args.hold, args.hz)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
