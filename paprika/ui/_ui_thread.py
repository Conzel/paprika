from typing import List, Tuple

from PyQt5.QtCore import QThread, pyqtSignal, QTimer, QEventLoop, QObject
import numpy as np

from paprika.ml import NeuralNetworkAnalysis, DummyAnalysis
from paprika.ui._config import *
from paprika.ui._helper import middle_cropped_image, bgr_to_rgb, image_for_analysis


class AnalysisResultsDTO:
    """
    DTO containing all the ML analysis information that the UI needs.
    """

    def __init__(self, image, layers):
        self.original_image = image
        self.layers = layers
        self.layer_filters = {}
        self.saliency_map = None
        self.class_predictions = []

    def add_layer_filters(self, layer, filters):
        self.layer_filters[layer] = filters

    def add_saliency_map(self, saliency_map):
        self.saliency_map = saliency_map


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


class AnalysisWorker(QObject):
    """
    Worker capturing new frozen camera images and signaling new ML analyses.
    It has to be run in a separate QThread.
    """

    new_analysis_signal = pyqtSignal(AnalysisResultsDTO)

    def __init__(self, camera, refresh_seconds, analysis_class: NeuralNetworkAnalysis):
        super().__init__()
        self.analysis_class = analysis_class
        self.camera = camera
        self.refresh_mseconds = refresh_seconds * 1000

    def obtain_new_analysis(self):
        camera_image = self.camera.read()

        results_dto = AnalysisResultsDTO(camera_image, selected_layers)
        # make image suitable for analysis
        analysis_image = image_for_analysis(camera_image)
        analysis = self.analysis_class(analysis_image)

        # filter visualisations
        for layer in selected_layers:
            activated_filters = analysis.get_most_activated_filters(
                layer, filter_column_length * filter_row_length
            )
            results_dto.add_layer_filters(layer, activated_filters)

        # saliency map
        saliency_map = analysis.get_saliency_map()
        results_dto.add_saliency_map(saliency_map)

        # predictions
        predictions = analysis.get_class_predictions(nr_predictions, nr_imagenet_images)
        results_dto.class_predictions = predictions

        self.new_analysis_signal.emit(results_dto)

    def start(self):
        self.obtain_new_analysis()
        self.timer = QTimer(self)
        self.timer.setInterval(self.refresh_mseconds)
        self.timer.timeout.connect(self.obtain_new_analysis)
        self.timer.start()
