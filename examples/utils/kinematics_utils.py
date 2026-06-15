import pinocchio as pin
import numpy as np


def _first_existing_frame(robot_model, frame_names):
    for frame_name in frame_names:
        if frame_name and robot_model.existFrame(frame_name):
            return frame_name
    return None


def comupute_forward_kinematics(joint_poistions, robot_urdf_path, base_frame=None, ee_frame=None):
    robot_model = pin.buildModelFromUrdf(robot_urdf_path)
    robot_data = robot_model.createData()

    q = np.asarray(joint_poistions, dtype=np.float64).reshape(-1)
    if q.shape[0] > robot_model.nq:
        # Recorded Franka+hand vectors contain arm joints first, then hand joints.
        q = q[: robot_model.nq]
    elif q.shape[0] < robot_model.nq:
        raise ValueError(
            f"Joint vector is too short for URDF: expected at least {robot_model.nq}, "
            f"got {q.shape[0]}."
        )

    pin.forwardKinematics(robot_model, robot_data, q)
    pin.updateFramePlacements(robot_model, robot_data)

    base_frame = base_frame or _first_existing_frame(
        robot_model,
        (
            "base",
            "fr3_link0",
            "panda_link0",
        ),
    )
    ee_frame = ee_frame or _first_existing_frame(
        robot_model,
        (
            "fr3_link8",
            "fr3_link7",
            "panda_link8",
            "tool0",
            "ee_link",
        ),
    )
    if base_frame is None:
        raise ValueError("Could not find a supported base frame in the URDF.")
    if ee_frame is None:
        raise ValueError("Could not find a supported end-effector frame in the URDF.")

    base_frame_id = robot_model.getFrameId(base_frame)
    ee_frame_id = robot_model.getFrameId(ee_frame)

    oMb = robot_data.oMf[base_frame_id]
    oMe = robot_data.oMf[ee_frame_id]
    bMe = oMb.inverse() * oMe

    xyz = bMe.translation

    quaternion = bMe.rotation
    quaternion = pin.Quaternion(quaternion)
    return xyz, [quaternion.w, quaternion.x, quaternion.y, quaternion.z]
