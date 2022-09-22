import os
import random
import time
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple

from paprika.ml._config import *


class ClassPrediction:
    """
    Represents a prediction of the neural network.

    label: the name of the class that the neural network predicted.
    score: the confidence of the prediction in percent.
    similar_images: list of flexible length containing the full paths of images that are similar to the input image
    """

    def __init__(self, label: str, score: float, similar_images: List[str]):
        self.label = label
        self.score = score
        self.similar_images = []


class NeuralNetworkAnalysis(ABC):
    """
    This interface represents the pass of an image through our Neural Network.

    The analysis is performed on the image that is passed to the constructor.
    It may be performed lazily only when the corresponding method (f.e. get_most_activated_filters)
    are called. All results are received via class-specific methods.

    The interface can be implemented for different neural networks.
    """

    @abstractmethod
    def __init__(self, img: np.ndarray):
        """Performs the analysis on the given image."""
        pass

    @abstractmethod
    def get_most_activated_filters(
        self, layer: str, n: int
    ) -> List[Tuple[str, int, float]]:
        """
        Returns a list of length n containing (image_path, filter_id, filter_activation)
        elements, where
        - image_path is the full path of the visualisation image
        - filter_id is the order of the filter in the layer
        - filter_activation is a percentage signifying the activation of the filter
        (the filter_activations of all the filters in a layer should add up to 100%)

        The elements are ordered by filter_activation in decreasing order.
        """
        pass

    @abstractmethod
    def get_saliency_map(self) -> np.ndarray:
        """
        Returns a saliency map of the image. The saliency map has the same dimensions
        as the input image and is a heatmap of the most important pixels in the image.
        """
        pass

    @abstractmethod
    def get_class_predictions(
        self, n_predictions: int, n_images: int
    ) -> List[ClassPrediction]:
        """
        Returns a list of the n_predictions most likely classes for the image.
        Each class contains the score (percentage), the label (name), and a list of n_images similar images.

        The images are returned in descending order of likelihood.
        """
        pass


class Inceptionv1Analysis(NeuralNetworkAnalysis):
    """
    Performs image analysis using Inceptionv1.
    """

    def __init__(self, img: np.ndarray):
        """Performs the analysis on the given image."""
        pass

    def get_most_activated_filters(
        self, layer: str, n: int
    ) -> List[Tuple[str, int, float]]:
        """
        Returns a list of length n containing (image_path, filter_id, filter_activation)
        elements, where
        - image_path is the full path of the visualisation image
        - filter_id is the order of the filter in the layer
        - filter_activation is a percentage signifying the activation of the filter
        (the filter_activations of all the filters in a layer should add up to 100%)

        The elements are ordered by filter_activation in decreasing order.
        """
        pass

    def get_saliency_map(self) -> np.ndarray:
        """
        Returns a saliency map of the image. The saliency map has the same dimensions
        as the input image and is a heatmap of the most important pixels in the image.
        """
        pass

    def get_class_predictions(
        self, n_predictions: int, n_images: int
    ) -> List[ClassPrediction]:
        """
        Returns a list of the n_predictions most likely classes for the image.
        Each class contains the score (percentage), the label (name), and a list of n_images similar images.

        The images are returned in descending order of likelihood.
        """
        pass


class DummyAnalysis(NeuralNetworkAnalysis):
    """
    Returns dummy data.
    """

    def __init__(self, img: np.ndarray, delay: int = 0):
        """
        delay: the number of seconds of delay added in all of the class's methods
        """
        super().__init__(img)
        self.delay = delay

    def get_most_activated_filters(
        self, layer: str, n: int
    ) -> List[Tuple[str, int, float]]:
        """
        Returns a list of length n containing (image_path, filter_id, filter_activation)
        elements, where
        - image_path is the full path of the visualisation image
        - filter_id is the order of the filter in the layer
        - filter_activation is a percentage signifying the activation of the filter
        (the filter_activations of all the filters in a layer should add up to 100%)

        The elements are ordered by filter_activation in decreasing order.
        """
        time.sleep(self.delay)
        filters = []
        folder_path = "../" + visualisations_relative_path
        nr_filters = filter_numbers_per_layer[layer]
        possible_activations = np.linspace(0.08, 4.532).tolist()
        activations = random.choices(possible_activations, k=n)
        activations.sort(reverse=True)
        for i in range(n):
            filter_id = random.randint(0, nr_filters - 1)
            image_path = os.path.abspath(
                os.path.expanduser(
                    os.path.expandvars(f"{folder_path}{layer}/{filter_id}.jpg")
                )
            )
            filter_activation = activations[i]
            filters.append((image_path, filter_id, filter_activation))
        return filters

    def get_saliency_map(self) -> np.ndarray:
        """
        Returns a saliency map of the image. The saliency map has the same dimensions
        as the input image and is a heatmap of the most important pixels in the image.
        """
        pass

    def get_class_predictions(
        self, n_predictions: int, n_images: int
    ) -> List[ClassPrediction]:
        """
        Returns a list of the n_predictions most likely classes for the image.
        Each class contains the score (percentage), the label (name), and a list of n_images similar images.

        The images are returned in descending order of likelihood.
        """
        pass
