import os
import random
import time
import numpy as np
import torch
from ._label_converter import labelConverter
from pytorch_grad_cam import (
    GradCAM,
    ScoreCAM,
    GradCAMPlusPlus,
    AblationCAM,
    XGradCAM,
    EigenCAM,
    FullGrad,
    EigenGradCAM,
)
import torchvision.transforms as T
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt
from ._imagenet_class_list import IMAGENET_CLASS_LIST
from lucent.modelzoo import inceptionv1
from PIL import Image
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
import csv

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
        self, layer_string: str, n: int
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


# hook function gets executed during forward pass
# this is used to obtain the feature map aka layer output
class SaveFeatures:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output

    def close(self):
        self.hook.remove()


class Inceptionv1Analysis(NeuralNetworkAnalysis):
    """
    Performs image analysis using Inceptionv1.
    """

    def __init__(self, img: np.ndarray):
        """Performs the analysis on the given image."""
        self.model = inceptionv1(pretrained=True).eval()
        self.image = img
        self.preprocess_image = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def get_most_activated_filters(
        self, layer_string: str, n: int
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
        layer_number = filter_strings_to_numbers[layer_string]
        layer = list(self.model.children())[layer_number]
        activations = SaveFeatures(layer)
        image = self.preprocess_image(self.image).unsqueeze(0)
        predictions = self.model(image)[0]
        mean_act = [
            activations.features[0, i].mean()
            for i in range(activations.features.shape[1])
        ]

        # select the filters with the highest score
        most_activated_filters = (
            torch.topk(torch.tensor(mean_act), n).indices.cpu().detach().numpy()
        )

        filters = []
        folder_path = "../" + visualisations_relative_path

        for i in range(n):
            filter_id = most_activated_filters[i]
            filter_activation = (
                mean_act[filter_id] / torch.sum(torch.tensor(mean_act))
            ).item()
            image_path = os.path.abspath(
                os.path.expanduser(
                    os.path.expandvars(f"{folder_path}{layer_string}/{filter_id}.jpg")
                )
            )
            filters.append((image_path, filter_id, filter_activation))

        return filters

    def get_saliency_map(self) -> np.ndarray:
        """
        Returns a saliency map of the image. The saliency map has the same dimensions
        as the input image and is a heatmap of the most important pixels in the image.
        """
        target_layers = [list(self.model.children())[136]]  # lucent implementation

        # Construct the CAM object once, and then re-use it on many images:
        cam = GradCAMPlusPlus(
            model=self.model, target_layers=target_layers, use_cuda=False
        )
        # targets = [ClassifierOutputTarget(281)]
        # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
        img_to_tensor = T.PILToTensor()(self.image)
        input_tensor = img_to_tensor[None, :] / 255.0
        grayscale_cam = cam(
            input_tensor=input_tensor, targets=None, eigen_smooth=False, aug_smooth=True
        )
        # In this example grayscale_cam has only one image in the batch:
        grayscale_cam = grayscale_cam[0, :]
        rgb_image = np.asarray(T.ToPILImage()(img_to_tensor)) / 255.0
        visualization = show_cam_on_image(rgb_image, grayscale_cam)
        return visualization

    def get_class_predictions(
        self, n_predictions: int, n_images: int
    ) -> List[ClassPrediction]:
        """
        Returns a list of the n_predictions most likely classes for the image.
        Each class contains the score (percentage), the label (name), and a list of n_images similar images.

        The images are returned in descending order of likelihood.
        """
        image = self.preprocess_image(self.image).unsqueeze(0)
        predictions = self.model(image)[0]
        translations = self.read_csv("translations.csv")

        # Create Empty Dict with all Categories in it:
        translated_classes = self.create_translation_class_dict()

        # For every prediction: add the percentage to the translated class
        for idx in range(0, len(predictions)):
            label = labelConverter()[
                idx
                + 1  # idx+1 because predictions start with index 0 but labels with index 1
            ]
            prediction = predictions[idx]
            translated_class = translations[label]
            translated_classes[translated_class] += prediction

        ordered_predictions = {
            k: v
            for k, v in sorted(
                translated_classes.items(), key=lambda x: x[1], reverse=True
            )
        }

        final_predictions = []
        i = 0
        for pred_label in ordered_predictions:
            if i <= n_predictions - 1:
                class_prediction = ClassPrediction(
                    label=pred_label,  # labelConverter() needed for lucent implementation
                    score=ordered_predictions[pred_label],
                    similar_images=None,
                )
                final_predictions.append(class_prediction)
                i += 1
            else:
                break

        return final_predictions

    def read_csv(self, file_path) -> Dict[str]:
        """
        Reads all translations from translations.csv and puts them into a dictionary

        returns dictionary of translations. {'kit fox':'Fuchs', ...}
        """
        # Write all old labels into csv
        input_file = csv.DictReader(open(file_path, encoding="utf-8"))
        translateDict = {}
        for item in input_file:
            key = item["old_label"]
            val = item["new_label"]
            translateDict[key] = val

        return translateDict

    def create_translation_class_dict(self) -> Dict:
        """
        Returns a dictionary where each class and a respective percentage (initialized with 0) is saved
        """
        translations = self.read_csv("translations.csv")

        returnDict = {}

        # Item
        for item in translations:
            translation = translations[item]
            returnDict[translation] = 0

        return returnDict


class DummyAnalysis(NeuralNetworkAnalysis):
    """
    Returns dummy data.
    """

    def __init__(self, img: np.ndarray, delay: int = 2):
        """
        Initialisation sleeps for delay seconds.
        """
        super().__init__(img)
        time.sleep(delay)
        self.image = img

    def get_most_activated_filters(
        self, layer_string: str, n: int
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
        filters = []
        folder_path = "../" + visualisations_relative_path
        nr_filters = filter_numbers_per_layer[layer_string]
        possible_activations = np.linspace(0.08, 4.532).tolist()
        activations = random.choices(possible_activations, k=n)
        activations.sort(reverse=True)
        for i in range(n):
            filter_id = random.randint(0, nr_filters - 1)
            image_path = os.path.abspath(
                os.path.expanduser(
                    os.path.expandvars(f"{folder_path}{layer_string}/{filter_id}.jpg")
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
        img = self.image * 255
        return np.clip(1 * (img - 128) + 128, 0, 255)

    def get_class_predictions(
        self, n_predictions: int, n_images: int
    ) -> List[ClassPrediction]:
        """
        Returns a list of the n_predictions most likely classes for the image.
        Each class contains the score (percentage), the label (name), and a list of n_images similar images.

        The images are returned in descending order of likelihood.
        """
        pass
