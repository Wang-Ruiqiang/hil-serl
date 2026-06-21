#!/usr/bin/env python3

import math
import re
import sys
from pathlib import Path

import numpy as np
from absl import app, flags


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from utils.kinematics_utils import comupute_forward_kinematics


FLAGS = flags.FLAGS

flags.DEFINE_string(
    "frame_root",
    "/media/user/data3/wrq/recorded_data/tennis_ball_pick/tennis_ball_pick-6-17-0",
    "Recorded data root containing frame_xxx folders.",
)
flags.DEFINE_string(
    "robot_urdf_path",
    "",
    "Franka URDF path. Defaults to examples/urdf/fr3_moveit_servo.urdf.",
)
flags.DEFINE_string(
    "base_frame",
    "",
    "Optional base frame override for Pinocchio FK.",
)
flags.DEFINE_string(
    "ee_frame",
    "",
    "Optional end-effector frame override for Pinocchio FK.",
)
flags.DEFINE_integer(
    "max_frames",
    0,
    "Optional max number of frames to verify. 0 means all frames.",
)
flags.DEFINE_integer(
    "top_k",
    10,
    "Number of worst frames to print.",
)


def _frame_number(path: Path) -> int:
    match = re.search(r"frame_(\d+)", path.name)
    return int(match.group(1)) if match else 10**12


def _load_vector(path: Path, expected_min_len: int):
    values = np.loadtxt(path, dtype=np.float64).reshape(-1)
    if values.shape[0] < expected_min_len:
        raise ValueError(f"{path} has {values.shape[0]} values, expected at least {expected_min_len}")
    return values


def _normalize_quat_wxyz(quat):
    quat = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(quat)
    if norm < 1e-12:
        raise ValueError(f"Invalid near-zero quaternion: {quat}")
    return quat / norm


def _quat_angle_error_deg(q_topic_wxyz, q_fk_wxyz):
    q_topic = _normalize_quat_wxyz(q_topic_wxyz)
    q_fk = _normalize_quat_wxyz(q_fk_wxyz)
    dot = abs(float(np.dot(q_topic, q_fk)))
    dot = float(np.clip(dot, -1.0, 1.0))
    return math.degrees(2.0 * math.acos(dot))


def _summarize(name, values):
    values = np.asarray(values, dtype=np.float64)
    print(
        f"[{name}] mean={values.mean():.6f} median={np.median(values):.6f} "
        f"p95={np.percentile(values, 95):.6f} max={values.max():.6f}"
    )


def main(_):
    if not FLAGS.frame_root:
        raise ValueError("--frame_root is required")

    frame_root = Path(FLAGS.frame_root).expanduser().resolve()
    robot_urdf_path = Path(FLAGS.robot_urdf_path).expanduser().resolve() if FLAGS.robot_urdf_path else (
        REPO_ROOT / "examples" / "urdf" / "fr3_moveit_servo.urdf"
    )
    if not frame_root.exists():
        raise FileNotFoundError(f"Missing frame_root: {frame_root}")
    if not robot_urdf_path.exists():
        raise FileNotFoundError(f"Missing URDF: {robot_urdf_path}")

    frame_dirs = sorted(
        [path for path in frame_root.iterdir() if path.is_dir() and path.name.startswith("frame_")],
        key=_frame_number,
    )
    if FLAGS.max_frames > 0:
        frame_dirs = frame_dirs[: FLAGS.max_frames]

    results = []
    skipped = 0
    for frame_dir in frame_dirs:
        joint_path = frame_dir / "right_arm_joint.txt"
        topic_pose_path = frame_dir / "robot_ee_pose.txt"
        if not joint_path.exists() or not topic_pose_path.exists():
            skipped += 1
            continue
        try:
            joint = _load_vector(joint_path, 7)
            topic_pose = _load_vector(topic_pose_path, 7)[:7]
            fk_xyz, fk_quat = comupute_forward_kinematics(
                joint,
                str(robot_urdf_path),
                base_frame=FLAGS.base_frame or None,
                ee_frame=FLAGS.ee_frame or None,
            )
            fk_pose = np.concatenate([np.asarray(fk_xyz), np.asarray(fk_quat)])
            pos_err_m = float(np.linalg.norm(topic_pose[:3] - fk_pose[:3]))
            quat_err_deg = _quat_angle_error_deg(topic_pose[3:7], fk_pose[3:7])
            results.append(
                {
                    "frame": frame_dir.name,
                    "topic_pose": topic_pose,
                    "fk_pose": fk_pose,
                    "pos_err_m": pos_err_m,
                    "quat_err_deg": quat_err_deg,
                }
            )
        except Exception as exc:
            skipped += 1
            print(f"[warn] skipped {frame_dir}: {exc}")

    print(f"[source] frame_root={frame_root}")
    print(f"[source] robot_urdf_path={robot_urdf_path}")
    print(f"[source] base_frame={FLAGS.base_frame or 'auto'} ee_frame={FLAGS.ee_frame or 'auto'}")
    print(f"[verify] checked={len(results)} skipped={skipped}")
    if not results:
        print("[verify] no comparable frames found. Did you record with the updated code?")
        return

    pos_err_m = np.asarray([item["pos_err_m"] for item in results])
    quat_err_deg = np.asarray([item["quat_err_deg"] for item in results])
    _summarize("position_error_m", pos_err_m)
    _summarize("position_error_mm", pos_err_m * 1000.0)
    _summarize("quaternion_angle_error_deg", quat_err_deg)

    worst = sorted(
        results,
        key=lambda item: (item["pos_err_m"], item["quat_err_deg"]),
        reverse=True,
    )[: max(0, FLAGS.top_k)]
    print("[worst] largest position errors:")
    for item in worst:
        print(
            f"  {item['frame']}: pos_err={item['pos_err_m'] * 1000.0:.3f}mm "
            f"quat_err={item['quat_err_deg']:.3f}deg "
            f"topic_xyz={item['topic_pose'][:3].tolist()} fk_xyz={item['fk_pose'][:3].tolist()}"
        )


if __name__ == "__main__":
    app.run(main)
