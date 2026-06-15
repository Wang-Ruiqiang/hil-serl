import copy
import json
import os
import threading
import time

import cv2
import numpy as np

from franka_env.gaze.display_markers import detect_gaze_display_markers, marker_points_for_size

try:
    import msgpack
    import zmq
except Exception:
    msgpack = None
    zmq = None


class EpisodeDataRecorder:
    """Record per-frame robot/camera/tactile data plus optional Pupil gaze packets."""

    def __init__(
        self,
        env,
        frame_root: str,
        enable_gaze: bool = False,
        pupil_host: str = "127.0.0.1",
        pupil_port: int = 50020,
    ):
        self.env = env
        self.enable_gaze = bool(enable_gaze)
        self.frame_root = frame_root
        self.frame_save_path = frame_root
        self.et_mirror_dir = os.path.join(frame_root, "et_images")
        self.rs_mirror_dir = os.path.join(frame_root, "rs_images")
        self.et_gaze_dir = os.path.join(frame_root, "et_images_gaze")
        self.pupil_host = pupil_host
        self.pupil_port = int(pupil_port)

        self.global_frame_id = 0
        self.cur_ep_start = None
        self.episode_frame_ranges = []
        self.episode_counter = 0

        self.front_color_buffer = []
        self.front_depth_buffer = []
        self.side_color_buffer = []
        self.side_depth_buffer = []
        self.wrist_color_buffer = []
        self.wrist_depth_buffer = []
        self.joint_buffer = []
        self.robot_ee_pose_buffer = []
        self.hand_state_buffer = []
        self.action_buffer = []
        self.rthumb_raw_buffer = []
        self.rindex_raw_buffer = []
        self.rmiddle_raw_buffer = []
        self.rthumb_heatmap_buffer = []
        self.rindex_heatmap_buffer = []
        self.rmiddle_heatmap_buffer = []
        self.et_world_payload_buffer = []
        self.pupil_gaze_buffer = []

        self._pupil_ctx = None
        self._pupil_sub = None
        self._pupil_last_world_payload = None
        self._pupil_last_gaze = None
        self._running = False
        self._timing_enabled = os.environ.get("HIL_TIMING", "0") == "1"

    def _time_log(self, frame_id, label, start):
        if self._timing_enabled:
            print(f"[timing][recorder][frame={frame_id}] {label}: {time.time() - start:.4f}s")

    def start(self):
        os.makedirs(self.frame_root, exist_ok=True)
        if self.enable_gaze:
            os.makedirs(self.et_mirror_dir, exist_ok=True)
            os.makedirs(self.rs_mirror_dir, exist_ok=True)
            os.makedirs(self.et_gaze_dir, exist_ok=True)
        print(f"[EpisodeDataRecorder] enabled, frame_root={self.frame_root}")

        if not self.enable_gaze:
            return
        if zmq is None or msgpack is None:
            print("[EpisodeDataRecorder][WARN] pyzmq/msgpack unavailable; Pupil stream disabled.")
            return

        self._running = True
        try:
            self._pupil_ctx = zmq.Context()
            ctrl = self._pupil_ctx.socket(zmq.REQ)
            ctrl.connect(f"tcp://{self.pupil_host}:{self.pupil_port}")
            ctrl.send_string("SUB_PORT")
            sub_port = ctrl.recv_string()
            ctrl.close()

            self._pupil_sub = self._pupil_ctx.socket(zmq.SUB)
            self._pupil_sub.connect(f"tcp://{self.pupil_host}:{sub_port}")
            self._pupil_sub.setsockopt(zmq.RCVTIMEO, 500)
            self._pupil_sub.setsockopt_string(zmq.SUBSCRIBE, "gaze.")
            self._pupil_sub.setsockopt_string(zmq.SUBSCRIBE, "frame.world")
            threading.Thread(target=self._pupil_listen, daemon=True).start()
            print("[EpisodeDataRecorder] Pupil subscriber started.")
        except Exception as exc:
            print(f"[EpisodeDataRecorder][WARN] Pupil init failed: {exc}")
            self._pupil_sub = None

    def close(self):
        self._running = False
        if self._pupil_sub is not None:
            try:
                self._pupil_sub.close(0)
            except Exception:
                pass
        if self._pupil_ctx is not None:
            try:
                self._pupil_ctx.term()
            except Exception:
                pass

    def mark_episode_start(self):
        self.cur_ep_start = int(self.global_frame_id)
        self.episode_counter += 1
        return self.cur_ep_start

    def end_episode(self):
        try:
            if self.cur_ep_start is None:
                return None
            start = int(np.asarray(self.cur_ep_start).reshape(-1)[0])
            end = int(self.global_frame_id) - 1
            if end >= start:
                frame_range = (start, end)
                self.episode_frame_ranges.append(frame_range)
                return frame_range
            return None
        finally:
            self.cur_ep_start = None

    def save_frame(self, images=None, depth_images=None):
        frame_id = int(self.global_frame_id)
        t_total = time.time()
        images = {} if images is None else images
        depth_images = {} if depth_images is None else depth_images
        try:
            t = time.time()
            self._append_robot_state()
            self._time_log(frame_id, "append_robot_state", t)
            if not images:
                t = time.time()
                images, depth_images = self.env.get_rgb_and_dpth_im()
                self._time_log(frame_id, "get_rgb_and_dpth_im", t)
            t = time.time()
            self._append_camera_frames(images, depth_images)
            self._time_log(frame_id, "append_camera_frames", t)
            t = time.time()
            self._append_tactile_frames()
            self._time_log(frame_id, "append_tactile_frames", t)
        except Exception as exc:
            print("An error occurred while processing frames")
            print(exc)

        try:
            t = time.time()
            self._append_mirror_and_gaze_buffers(images)
            self._time_log(frame_id, "append_mirror_and_gaze_buffers", t)
        except Exception as exc:
            print(f"[EpisodeDataRecorder][mirror/gaze] {exc}")

        self.global_frame_id = frame_id + 1
        self._time_log(frame_id, "save_frame_total", t_total)
        return frame_id

    def save_all_data_on_exit(self):
        for frame_id in range(self.global_frame_id):
            print("save frame ", frame_id)
            frame_dir = os.path.join(self.frame_save_path, f"frame_{frame_id}")
            os.makedirs(frame_dir, exist_ok=True)

            if len(self.front_color_buffer) > frame_id:
                cv2.imwrite(os.path.join(frame_dir, "color_image.jpg"), self.front_color_buffer[frame_id])
                if self.enable_gaze:
                    cv2.imwrite(os.path.join(self.rs_mirror_dir, f"{frame_id}.jpg"), self.front_color_buffer[frame_id])
                if len(self.front_depth_buffer) > frame_id and self.front_depth_buffer[frame_id] is not None:
                    cv2.imwrite(os.path.join(frame_dir, "depth_image.png"), self.front_depth_buffer[frame_id])
            if len(self.side_color_buffer) > frame_id:
                cv2.imwrite(os.path.join(frame_dir, "color_image3.jpg"), self.side_color_buffer[frame_id])
                if len(self.side_depth_buffer) > frame_id and self.side_depth_buffer[frame_id] is not None:
                    cv2.imwrite(os.path.join(frame_dir, "depth_image3.png"), self.side_depth_buffer[frame_id])
            if len(self.wrist_color_buffer) > frame_id:
                cv2.imwrite(os.path.join(frame_dir, "color_image2.jpg"), self.wrist_color_buffer[frame_id])
                if len(self.wrist_depth_buffer) > frame_id and self.wrist_depth_buffer[frame_id] is not None:
                    cv2.imwrite(os.path.join(frame_dir, "depth_image2.png"), self.wrist_depth_buffer[frame_id])

            if self.env.enable_tactile:
                if len(self.rthumb_raw_buffer) > frame_id:
                    cv2.imwrite(os.path.join(frame_dir, "thumb_raw_image.jpg"), self.rthumb_raw_buffer[frame_id])
                if len(self.rthumb_heatmap_buffer) > frame_id:
                    cv2.imwrite(os.path.join(frame_dir, "thumb_heat_map.jpg"), self.rthumb_heatmap_buffer[frame_id])
                if len(self.rindex_raw_buffer) > frame_id:
                    cv2.imwrite(os.path.join(frame_dir, "index_raw_image.jpg"), self.rindex_raw_buffer[frame_id])
                if len(self.rindex_heatmap_buffer) > frame_id:
                    cv2.imwrite(os.path.join(frame_dir, "index_heat_map.jpg"), self.rindex_heatmap_buffer[frame_id])
                if getattr(self.env, "enable_dm_tac_middle", False) and len(self.rmiddle_raw_buffer) > frame_id:
                    cv2.imwrite(os.path.join(frame_dir, "middle_raw_image.jpg"), self.rmiddle_raw_buffer[frame_id])
                if getattr(self.env, "enable_dm_tac_middle", False) and len(self.rmiddle_heatmap_buffer) > frame_id:
                    cv2.imwrite(os.path.join(frame_dir, "middle_heat_map.jpg"), self.rmiddle_heatmap_buffer[frame_id])

            if len(self.joint_buffer) > frame_id:
                np.savetxt(os.path.join(frame_dir, "right_arm_joint.txt"), self.joint_buffer[frame_id])
            if len(self.robot_ee_pose_buffer) > frame_id:
                np.savetxt(
                    os.path.join(frame_dir, "robot_ee_pose.txt"),
                    np.atleast_1d(self.robot_ee_pose_buffer[frame_id]),
                    fmt="%.9f",
                )
            if len(self.hand_state_buffer) > frame_id:
                np.savetxt(
                    os.path.join(frame_dir, "hand_state.txt"),
                    np.atleast_1d(self.hand_state_buffer[frame_id]),
                    fmt="%.6f",
                )

            if self.enable_gaze:
                et_img = None
                gaze = self._gaze_for_frame(frame_id)
                if (
                    len(self.et_world_payload_buffer) > frame_id
                    and self.et_world_payload_buffer[frame_id] is not None
                ):
                    et_img = self._decode_et_payload(self.et_world_payload_buffer[frame_id])
                    if et_img is not None:
                        cv2.imwrite(os.path.join(self.et_mirror_dir, f"{frame_id}.jpg"), et_img)
                        et_gaze = self._draw_gaze_point(et_img, gaze)
                        cv2.imwrite(os.path.join(self.et_gaze_dir, f"{frame_id}.jpg"), et_gaze)
                if gaze is not None:
                    with open(os.path.join(frame_dir, "pupil_gaze.json"), "w") as f:
                        json.dump(gaze, f, indent=2)
                    gaze_contact = self._build_screen_marker_gaze_contact(et_img, gaze)
                    if gaze_contact is not None:
                        with open(os.path.join(frame_dir, "gaze_contact.json"), "w") as f:
                            json.dump(gaze_contact, f, indent=2)
            if len(self.action_buffer) > frame_id:
                np.savetxt(
                    os.path.join(frame_dir, "action.txt"),
                    np.atleast_1d(self.action_buffer[frame_id]),
                    fmt="%.6f",
                )

    def _pupil_listen(self):
        if self._pupil_sub is None or zmq is None or msgpack is None:
            return
        while self._running:
            try:
                parts = self._pupil_sub.recv_multipart()
                if not parts:
                    continue
                topic = parts[0].decode("utf-8")
                if topic.startswith("gaze."):
                    data = msgpack.loads(parts[1], raw=False)
                    self._pupil_last_gaze = {
                        "topic": topic,
                        "ts": data.get("timestamp", None),
                        "data": data,
                    }
                elif topic == "frame.world":
                    self._pupil_last_world_payload = parts[2] if len(parts) >= 3 else parts[1]
            except zmq.Again:
                continue
            except Exception as exc:
                print(f"[EpisodeDataRecorder][PupilListen] {exc}")

    def _gaze_for_frame(self, frame_id):
        if len(self.pupil_gaze_buffer) <= frame_id:
            return None
        return self.pupil_gaze_buffer[frame_id]

    def _decode_et_payload(self, payload):
        if payload is None:
            return None
        try:
            return cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        except Exception as exc:
            print(f"[EpisodeDataRecorder][ET-DECODE] {exc}")
            return None

    def _gaze_uv(self, image_shape, gaze):
        if gaze is None:
            return None
        payload = gaze.get("data", gaze)
        norm_pos = payload.get("norm_pos")
        if norm_pos is None or len(norm_pos) < 2:
            return None
        h, w = image_shape[:2]
        try:
            x_norm, y_norm = float(norm_pos[0]), float(norm_pos[1])
        except Exception:
            return None
        u = int(round(x_norm * (w - 1)))
        v = int(round((1.0 - y_norm) * (h - 1)))
        if not (0 <= u < w and 0 <= v < h):
            return None
        return u, v

    def _draw_gaze_point(self, image_bgr, gaze=None):
        if gaze is None:
            gaze = self._pupil_last_gaze
        uv = self._gaze_uv(image_bgr.shape, gaze)
        if uv is None:
            return image_bgr
        vis = image_bgr.copy()
        u, v = uv
        cv2.circle(vis, (u, v), 14, (0, 0, 255), 3)
        cv2.circle(vis, (u, v), 4, (0, 255, 255), -1)
        cv2.line(vis, (u - 22, v), (u + 22, v), (0, 0, 255), 2)
        cv2.line(vis, (u, v - 22), (u, v + 22), (0, 0, 255), 2)
        cv2.putText(
            vis,
            f"gaze ({u},{v})",
            (max(0, u + 12), max(24, v - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )
        return vis

    def _build_screen_marker_gaze_contact(self, et_img, gaze):
        if et_img is None or gaze is None:
            return None

        eye_uv = self._gaze_uv(et_img.shape, gaze)
        if eye_uv is None:
            return None

        marker_points_eye = detect_gaze_display_markers(et_img)
        if marker_points_eye is None:
            return None

        dst_w = int(getattr(self.env, "gaze_rs_save_width", 640))
        dst_h = int(getattr(self.env, "gaze_rs_save_height", 480))
        marker_points_rs = np.asarray(
            getattr(self.env, "gaze_marker_points_realsense", marker_points_for_size(dst_w, dst_h)),
            dtype=np.float32,
        )
        homography = cv2.getPerspectiveTransform(marker_points_eye, marker_points_rs)
        eye_pt = np.asarray(eye_uv, dtype=np.float32).reshape(1, 1, 2)
        rs_xy = cv2.perspectiveTransform(eye_pt, homography)[0, 0]
        rs_xy[0] = np.clip(rs_xy[0], 0.0, max(0.0, dst_w - 1))
        rs_xy[1] = np.clip(rs_xy[1], 0.0, max(0.0, dst_h - 1))

        return {
            "hit": True,
            "source": "screen_marker_homography",
            "gaze_uv_in_eye": [float(eye_uv[0]), float(eye_uv[1])],
            "gaze_uv_in_realsense": [float(rs_xy[0]), float(rs_xy[1])],
            "realsense_size": [dst_w, dst_h],
            "marker_points_eye": marker_points_eye.tolist(),
            "marker_points_realsense": marker_points_rs.tolist(),
            "homography_eye_to_realsense": homography.tolist(),
        }

    def _append_robot_state(self):
        try:
            joint_position = self.env.ros_interface.get_current_joint()
            self.env.joint_position = np.asarray(joint_position, dtype=np.float32).copy()
        except Exception:
            if not hasattr(self.env, "joint_position"):
                self.env.joint_position = np.zeros(6, dtype=np.float32)

        joint_pose = np.concatenate(
            [self.env.joint_position, self.env.curr_leap_hand_pos],
            dtype=np.float32,
        )
        try:
            ee_pos, ee_quat = self.env.ros_interface.get_current_robot_ee()
        except Exception:
            ee_pos = getattr(self.env, "cur_position", np.zeros(3, dtype=np.float32))
            ee_quat = getattr(self.env, "cur_orientation", np.array([0, 1, 0, 0], dtype=np.float32))
        ee_pose = np.concatenate(
            [
                np.asarray(ee_pos, dtype=np.float32).reshape(3),
                np.asarray(ee_quat, dtype=np.float32).reshape(4),
            ],
            dtype=np.float32,
        )
        self.joint_buffer.append(copy.deepcopy(joint_pose))
        self.robot_ee_pose_buffer.append(copy.deepcopy(ee_pose))
        self.hand_state_buffer.append(copy.deepcopy(self.env.hand_state))
        self.action_buffer.append(copy.deepcopy(self.env.current_action))

    def _append_camera_frames(self, images, depth_images):
        for cam_name, img in images.items():
            depth = depth_images.get(cam_name)
            if cam_name == "front_camera":
                self.front_color_buffer.append(copy.deepcopy(img[..., ::-1]))
                self.front_depth_buffer.append(copy.deepcopy(depth))
            elif cam_name == "front_camera_2":
                self.side_color_buffer.append(copy.deepcopy(img[..., ::-1]))
                self.side_depth_buffer.append(copy.deepcopy(depth))
            elif cam_name == "wrist_camera":
                self.wrist_color_buffer.append(copy.deepcopy(img[..., ::-1]))
                self.wrist_depth_buffer.append(copy.deepcopy(depth))

    def _append_tactile_frames(self):
        if not self.env.enable_tactile:
            return
        self.rthumb_raw_buffer.append(copy.deepcopy(self.env.thumb_raw_img))
        self.rindex_raw_buffer.append(copy.deepcopy(self.env.index_raw_img))
        self.rthumb_heatmap_buffer.append(copy.deepcopy(self.env.thumb_heat_map))
        self.rindex_heatmap_buffer.append(copy.deepcopy(self.env.index_heat_map))
        if getattr(self.env, "enable_dm_tac_middle", False):
            self.rmiddle_raw_buffer.append(copy.deepcopy(self.env.middle_raw_img))
            self.rmiddle_heatmap_buffer.append(copy.deepcopy(self.env.middle_heat_map))

    def _append_mirror_and_gaze_buffers(self, images):
        if not self.enable_gaze:
            return

        t = time.time()
        payload = bytes(self._pupil_last_world_payload) if self._pupil_last_world_payload is not None else None
        self.et_world_payload_buffer.append(payload)
        self.pupil_gaze_buffer.append(copy.deepcopy(self._pupil_last_gaze))
        self._time_log(self.global_frame_id, "append_gaze_buffers", t)
