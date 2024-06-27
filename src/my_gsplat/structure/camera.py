import logging
from collections.abc import Generator

import cv2
import numpy as np

from .dataset import DataLoaderBase
from .Image import RGBDImage


class Camera(DataLoaderBase):
    def __init__(self, input_folder: str, cfg_file: str):
        super().__init__(input_folder, cfg_file)
        openni2.initialize(
            "/home/pixiu/Downloads/OrbbecViewer_1.1.13_202207221544_Linux"
        )
        self.device = openni2.Device.open_any()
        self.depth_stream = self.device.create_depth_stream()
        self.depth_stream.start()
        self.capture = cv2.VideoCapture(2)

    def __iter__(self) -> Generator[RGBDImage, None, None]:
        """
        get RGBDImage via camera
        :return: RGBDImage
        """
        try:
            color = self._get_rgb()
            depth = self._get_depth()
            yield RGBDImage(color, depth, self.K, self.scale)
        except Exception as e:
            logging.exception(e)
        finally:
            self.shut_down()

    def _get_rgb(self, index: int | None = None) -> np.ndarray:
        """
        :return: rgb_frame.shape=(height,width,color)
        """
        ret, rgb_frame = self.capture.read()
        if not ret:
            raise ValueError("got None RGBDFrame")
        return rgb_frame

    def _get_depth(self, index: int | None = None) -> np.ndarray:
        """
        :return: depth_array.shape = (height,width)
        """
        frame = self.depth_stream.read_frame()
        frame_data = frame.get_buffer_as_uint16()
        depth_array = np.ndarray(
            (frame.height, frame.width), dtype=np.uint16, buffer=frame_data
        )
        return depth_array

    def shut_down(self):
        """
        release resource...
        """
        cv2.destroyAllWindows()
        self.capture.release()
        self.depth_stream.stop()
        openni2.unload()
