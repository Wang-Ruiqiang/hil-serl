import numpy as np
import cv2
import pyrealsense2 as rs  # Intel RealSense cross-platform open-source API


class RSCapture:
    def get_device_serial_numbers(self):
        devices = rs.context().devices
        return [d.get_info(rs.camera_info.serial_number) for d in devices]

    def __init__(
        self,
        name,
        serial_number,
        dim=(640, 480),
        fps=15,
        depth=False,
        exposure=40000,
        color_format="bgr8",
    ):
        self.name = name
        assert serial_number in self.get_device_serial_numbers()
        self.serial_number = serial_number
        self.depth = depth
        self.color_format = self._parse_color_format(color_format)
        self.pipe = rs.pipeline()
        self.cfg = rs.config()
        self.cfg.enable_device(self.serial_number)
        self.cfg.enable_stream(rs.stream.color, dim[0], dim[1], self.color_format, fps)
        if self.depth:
            self.cfg.enable_stream(rs.stream.depth, dim[0], dim[1], rs.format.z16, fps)
        self.profile = self.pipe.start(self.cfg)
        self.s = self.profile.get_device().query_sensors()[0]
        self.s.set_option(rs.option.exposure, exposure)

        # Create an align object
        # rs.align allows us to perform alignment of depth frames to others frames
        # The "align_to" is the stream type to which we plan to align depth frames.
        align_to = rs.stream.color
        self.align = rs.align(align_to)

    @staticmethod
    def _parse_color_format(color_format):
        if color_format == "bgr8":
            return rs.format.bgr8
        if color_format == "yuyv":
            return rs.format.yuyv
        raise ValueError(f"Unsupported RealSense color_format: {color_format}")

    @staticmethod
    def _yuyv_to_bgr(image, color_frame):
        height = color_frame.get_height()
        width = color_frame.get_width()
        if image.dtype != np.uint8:
            image = image.view(np.uint8)
        image = image.reshape(height, width, 2)
        return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_YUY2)

    def read(self):
        frames = self.pipe.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        if self.depth:
            depth_frame = aligned_frames.get_depth_frame()

        if color_frame.is_video_frame():
            image = np.asarray(color_frame.get_data())
            if self.color_format == rs.format.yuyv:
                image = self._yuyv_to_bgr(image, color_frame)
            if self.depth and depth_frame.is_depth_frame():
                depth = np.expand_dims(np.asarray(depth_frame.get_data()), axis=2)
                return True, np.concatenate((image, depth), axis=-1)
            else:
                return True, image
        else:
            return False, None

    def close(self):
        self.pipe.stop()
        self.cfg.disable_all_streams()
