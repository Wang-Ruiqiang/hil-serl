import pinocchio as pin
import numpy as np
from transforms3d.quaternions import mat2quat, quat2mat

def comupute_forward_kinematics(joint_poistions, robot_urdf_path):
    robot_model = pin.buildModelFromUrdf(robot_urdf_path)
    robot_data = robot_model.createData()


    q = joint_poistions
    pin.forwardKinematics(robot_model, robot_data, q)
    pin.updateFramePlacements(robot_model, robot_data)

    # Define frame names
    base_frame = "base_link"    # Base frame in your URDF
    ee_frame = 'palm_lower'     # ee frame in your URDF   
    # Get frame IDs
    base_frame_id = robot_model.getFrameId(base_frame)
    ee_frame_id = robot_model.getFrameId(ee_frame)

    # Check if frames exist
    if base_frame_id == -1:
        print(f"Frame '{base_frame}' not found in the model.")
        return
    if ee_frame_id == -1:
        print(f"Frame '{ee_frame}' not found in the model.")
        return

    # Get the transforms (poses) of each frame in the world coordinate system
    oMb = robot_data.oMf[base_frame_id]     # Pose of base frame
    oMe = robot_data.oMf[ee_frame_id]     # Pose of ee frame

    xyz = oMe.translation

    quaternion = oMe.rotation.T  # Rotation matrix
    quaternion = pin.Quaternion(quaternion)
    return xyz, [quaternion.w, quaternion.x, quaternion.y, quaternion.z]


# for converting plam lower pose to denso end link ( a predefined tf)
def apply_transformation(position, orientation, transformation_matrix):
    """Apply a 4x4 transformation matrix to a position and orientation."""
    # Construct a 4x4 transformation matrix for the input position and orientation
    rotation_matrix = quat2mat(orientation)  # Convert quaternion to rotation matrix
    input_transformation_matrix = np.eye(4)
    input_transformation_matrix[:3, :3] = rotation_matrix
    input_transformation_matrix[:3, 3] = position

    # Apply the transformation using the @ operator
    result_transformation_matrix = input_transformation_matrix @ transformation_matrix

    # Extract the transformed position
    transformed_position = result_transformation_matrix[:3, 3]

    # Extract the transformed rotation matrix and convert back to quaternion
    transformed_rotation_matrix = result_transformation_matrix[:3, :3]
    transformed_orientation = mat2quat(transformed_rotation_matrix)

    # print("iput pose: ", position, orientation)
    # print("output pose: ", transformed_position, transformed_orientation)
    return transformed_position, transformed_orientation