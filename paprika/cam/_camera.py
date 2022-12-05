import random
from typing import Union
import cv2
import os
import queue
import threading
from abc import ABC, abstractmethod
import numpy as np

# necessary to work with Qt5: https://stackoverflow.com/questions/63829991/qt-qpa-plugin-could-not-load-the-qt-platform-plugin-xcb-in-even-though-it
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")
from PIL import Image


class Camera(ABC):
    """
    Webcam class that reads frames from an external webcam.

    Images are accessible via the read() method.
    """

    @abstractmethod
    def __init__(self, identifier: Union[int, str] = 0):
        """
        Initializes the webcam. The identifier can either be a name (for example an online webcam)
        or a number (for example a local webcam).

        The default value 0 attempts to open the first local webcam.
        """
        pass

    @abstractmethod
    def read(self) -> np.ndarray:
        """
        Returns a frame from the webcam.
        """
        pass


class BufferlessVideoCapture(Camera):
    """
    A class that represents a webcam. The webcam is bufferless, i.e.
    the read method always returns the latest frame.

    Based on:
    https://stackoverflow.com/questions/43665208/how-to-get-the-latest-frame-from-capture-device-camera-in-opencv
    """

    def __init__(self, name: Union[str, int]):
        self.cap = cv2.VideoCapture(name)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open webcam {name}")
        self.q = queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()  # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self) -> np.ndarray:
        return self.q.get()


class DummyCamera(Camera):
    """
    A dummy that returns a random image from the _dummy_camera folder.
    """

    def __init__(self, name: Union[str, int]):
        super().__init__()

    def read(self) -> np.ndarray:
        # source_folder = "../paprika/cam/_dummy_camera"
        # images = os.listdir(source_folder)
        # image_name = random.choice(images)
        # image = cv2.imread(f"{source_folder}/{image_name}")
        # return image
        return np.random.random((1080, 1920, 3)) * 255
