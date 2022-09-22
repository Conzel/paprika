from typing import List, Tuple

from PyQt5.QtCore import QThread, pyqtSignal, QTimer
import numpy as np

from paprika.ml import NeuralNetworkAnalysis, DummyAnalysis


class RunningCameraThread(QThread):
    """
    Thread signaling new images for the continuous camera feed.
    """

    new_capture_signal = pyqtSignal(np.ndarray)

    def __init__(self, parent, camera):
        QThread.__init__(self, parent)
        self.camera = camera

    def obtain_camera_capture(self):
        camera_image = self.camera.read()
        self.new_capture_signal.emit(camera_image)

    def run(self):
        timer = QTimer(self)
        timer.setInterval(100)
        timer.timeout.connect(self.obtain_camera_capture)
        timer.start()


class FrozenCameraThread(QThread):
    """
    Thread signaling new images as the analysed image.
    """

    new_capture_signal = pyqtSignal(np.ndarray)

    def __init__(self, parent, camera, refresh_seconds):
        QThread.__init__(self, parent)
        self.camera = camera
        self.refresh_rate = refresh_seconds * 1000

    def obtain_camera_capture(self):
        camera_image = self.camera.read()
        self.new_capture_signal.emit(camera_image)

    def run(self):
        self.obtain_camera_capture()
        timer = QTimer(self)
        timer.setInterval(self.refresh_rate)
        timer.timeout.connect(self.obtain_camera_capture)
        timer.start()


# class AnalysisThread(QThread):
#     """
#     Thread signaling the completion of the analysis of an image.
#     """
#
#     new_analysis_signal = pyqtSignal(results_dto)
#
#     def __init__(self, parent):
#         QThread.__init__(self, parent)
#
#     def compute_everything():
#         pass
