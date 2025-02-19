import pinocchio as pin

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
    return xyz, [quaternion.x, quaternion.y, quaternion.z, quaternion.w]