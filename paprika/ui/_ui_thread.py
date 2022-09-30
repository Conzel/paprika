from typing import List, Tuple

from PyQt5.QtCore import QThread, pyqtSignal, QTimer, QEventLoop, QObject
import numpy as np

from paprika.ml import NeuralNetworkAnalysis, DummyAnalysis
from paprika.ui._config import *


class AnalysisResultsDTO:
    """
    DTO containing all the ML analysis information that the UI needs.
    """

    def __init__(self, image, layers):
        self.original_image = image
        self.layers = layers
        self.layer_filters = {}

    def add_layer_filters(self, layer, filters):
        self.layer_filters[layer] = filters


class RunningCameraWorker(QObject):
    """
    Worker signaling new images for the running camera feed.
    It has to be run in a separate QThread.
    """

    new_capture_signal = pyqtSignal(np.ndarray)

    def __init__(self, camera):
        super().__init__()
        self.camera = camera

    def start(self):
        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.obtain_camera_capture)
        self.timer.start()

    def obtain_camera_capture(self):
        camera_image = self.camera.read()
        self.new_capture_signal.emit(camera_image)


class FrozenCameraWorker(QObject):
    """
    Worker signaling new images for the frozen camera feed.
    It has to be run in a separate QThread.
    """

    new_capture_signal = pyqtSignal(np.ndarray)

    def __init__(self, camera, refresh_seconds):
        super().__init__()
        self.camera = camera
        self.refresh_seconds = refresh_seconds * 1000

    def start(self):
        self.obtain_camera_capture()
        self.timer = QTimer(self)
        self.timer.setInterval(self.refresh_seconds)
        self.timer.timeout.connect(self.obtain_camera_capture)
        self.timer.start()

    def obtain_camera_capture(self):
        camera_image = self.camera.read()
        self.new_capture_signal.emit(camera_image)


class AnalysisWorker(QObject):
    """
    Worker signaling new ML analysis of an image.
    It has to be run in a separate QThread.
    """

    new_analysis_signal = pyqtSignal(AnalysisResultsDTO)

    def __init__(self, analysis_class: NeuralNetworkAnalysis, image: np.ndarray):
        super().__init__()
        self.analysis_class = analysis_class
        self.image = image

    def start(self):
        results_dto = AnalysisResultsDTO(self.image, selected_layers)
        # image might need to be cropped and converted to RGB first
        analysis = self.analysis_class(self.image)
        for layer in selected_layers:
            activated_filters = analysis.get_most_activated_filters(
                layer, filter_column_length * filter_row_length
            )
            results_dto.add_layer_filters(layer, activated_filters)
        self.new_analysis_signal.emit(results_dto)
