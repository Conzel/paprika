from typing import List, Tuple

from PyQt5.QtCore import QThread, pyqtSignal, QTimer
import numpy as np

from paprika.ml import NeuralNetworkAnalysis, DummyAnalysis
from paprika.ui._config import *


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


class AnalysisThread(QThread):
    """
    Thread signaling the completion of the analysis of an image.
    """

    new_analysis_signal = pyqtSignal(AnalysisResultsDTO)

    def __init__(self, parent, analysis_class):
        QThread.__init__(self, parent)
        self.analysis_class = analysis_class

    def compute_everything(self, image: np.ndarray):
        results_dto = AnalysisResultsDTO(image, selected_layers)
        # image might need to be cropped and converted to RGB first
        analysis = self.analysis_class(image)
        for layer in selected_layers:
            activated_filters = analysis.get_most_activated_filters(
                layer, filter_column_length * filter_row_length
            )
            results_dto.add_layer_filters(layer, activated_filters)

        self.new_analysis_signal.emit(results_dto)
