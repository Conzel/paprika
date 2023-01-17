import copy
import csv
import os
import random
import json
import time
import numpy as np
import torch
from ._label_converter import *
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
from skimage.transform import resize

from paprika.ml._config import *
from ..ui._config import selected_layers


class ClassPrediction:
    """
    Represents a prediction of the neural network.

    german_label: the name of the class that the neural network predicted, in German.
    english_label: the name of the class that the neural network predicted, in English.
    score: the confidence of the prediction in percent.
    similar_images: list of flexible length containing the full paths of images that are similar to the input image
    """

    def __init__(
        self,
        german_label: str,
        english_label: str,
        score: float,
        similar_images: List[str],
    ):
        self.german_label = german_label
        self.english_label = english_label
        self.score = score
        self.similar_images = similar_images


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


class Inceptionv1FasterAnalysis(NeuralNetworkAnalysis):
    """
    Performs image analysis using Inceptionv1. The image is only passed through the network once.
    """

    def get_device():
        if device_name == "cuda" and not torch.cuda.is_available():
            print(
                "WARNING: Cuda is not enable, but specified in config. Switching to CPU."
            )
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")

    # use this same dictionary for every image analysis
    subclass_group_dict = construct_subclass_group_dict()
    # create a copy of this dictionary for every image analysis
    group_score_dict = construct_group_score_dict(subclass_group_dict)
    # load model
    device = get_device()
    model = inceptionv1(pretrained=True).eval().to(device)
    # init CAM objects
    target_layers = [
        list(model.children())[136],
        list(model.children())[119],
        list(model.children())[128],
    ]
    cam = GradCAMPlusPlus(
        model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available()
    )
    # init image preprocessor
    preprocess_image = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    def __init__(self, img: np.ndarray):
        """Performs the analysis on the given image."""
        self.original_image = img
        # resize image to size 224x224
        img = img / 255
        img = resize(img, (224, 224), anti_aliasing=True)
        img = img * 255
        img = np.require(img, np.uint8, "C")
        self.image = Image.fromarray(img)

        # get the layer activations for each selected layer
        layers = selected_layers.copy()
        self.layer_for_feature_vector = "mixed5b"
        if self.layer_for_feature_vector not in layers:
            layers.append(self.layer_for_feature_vector)
        self.layer_activations = {}
        for layer_name in layers:
            layer_number = filter_strings_to_numbers[layer_name]
            layer = list(self.model.children())[layer_number]
            self.layer_activations[layer_name] = SaveFeatures(layer)
        # obtain predictions
        image = self.preprocess_image(self.image).unsqueeze(0).to(self.device)
        self.predictions = self.model(image)[0]

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
        activations = self.layer_activations[layer_string]
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
            ).item() * 100
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
        targets = None
        img_to_tensor = T.PILToTensor()(self.image)
        input_tensor = (img_to_tensor[None, :] / 255.0).to(self.device)
        grayscale_cam = self.cam(
            input_tensor=input_tensor,
            targets=targets,
            eigen_smooth=False,
            aug_smooth=False,
        )
        grayscale_cam = grayscale_cam[0, :]
        rgb_image = self.original_image / 255.0
        grayscale_cam_resized = resize(
            grayscale_cam, (rgb_image.shape[1], rgb_image.shape[1]), anti_aliasing=True
        )
        visualization = show_cam_on_image(
            rgb_image, grayscale_cam_resized, use_rgb=True
        )
        return visualization

    def get_similar_images(self, path, class_id, nr_images) -> List[str]:
        """
        Returns a list of size nr_images containing the full paths of most similar images
        """
        image_full_paths = []
        enough_images_in_class = True

        activations = self.layer_activations[self.layer_for_feature_vector]
        mean_act = [
            activations.features[0, i].mean()
            for i in range(activations.features.shape[1])
        ]
        act_sum = torch.sum(torch.tensor(mean_act))
        mean_act = (torch.tensor(mean_act) / act_sum).to(self.device)

        if nr_images != 0:
            tensor = (
                torch.load(f"{path}{class_id}/{class_id}_activation_tensor.pt")
                .float()
                .to(self.device)
            )
            dictionary = json.load(open(f"{path}{class_id}/{class_id}_dictionary.json"))

            if len(dictionary) < nr_images:
                print(
                    f"class {class_id} does not contain enough images"
                )  # shouldn't happen, when whole image net is used
                enough_images_in_class = False
            dot_product = mean_act[np.newaxis] @ tensor
            indices = torch.topk(dot_product, nr_images).indices.cpu().detach().numpy()

            image_full_paths = []

            for idx in indices[0]:
                if enough_images_in_class:
                    image = dictionary[str(idx)]
                else:
                    image = dictionary[str(indices[0][0])]
                full_path = path + str(class_id) + "/" + image
                image_full_paths.append(full_path)

        return image_full_paths

    def get_class_predictions(
        self, n_predictions: int, n_images: int
    ) -> List[ClassPrediction]:
        """
        Returns a list of the n_predictions most likely classes for the image.
        Each class contains the score (percentage), the label (name), and a list of n_images similar images.

        The images are returned in descending order of likelihood.
        """

        group_score_dict = copy.deepcopy(self.group_score_dict)

        # change group_score_dict to contain the maximum score of each group and the corresponding id
        for subclass_position in range(1, 1001):
            prediction = self.predictions[subclass_position].item()
            subclass = self.subclass_group_dict[subclass_position]
            if (
                prediction
                > group_score_dict[
                    (subclass.german_group, subclass.english_group)
                ].max_score
            ):
                group_score_dict[
                    (subclass.german_group, subclass.english_group)
                ].max_score = prediction
                group_score_dict[
                    (subclass.german_group, subclass.english_group)
                ].id = subclass.id

        # normalise group_score_dict max_scores such that they add up to ~100
        group_score_sum = sum(
            [group_score.max_score for group_score in group_score_dict.values()]
        )
        group_score_dict = {
            group_id: GroupScore(
                group_score.id, group_score.max_score / group_score_sum * 100
            )
            for group_id, group_score in group_score_dict.items()
        }

        # sort dictionary
        group_score_dict = dict(
            sorted(
                group_score_dict.items(),
                key=lambda item: item[1].max_score,
                reverse=True,
            )
        )

        # keep only n_predictions classes
        group_score_list = list(group_score_dict.items())[:n_predictions]

        # create the final ClassPrediction instances
        final_predictions = []
        for group_id, group_score in group_score_list:
            class_prediction = ClassPrediction(
                german_label=group_id.german_group,
                english_label=group_id.english_group,
                score=group_score.max_score,
                similar_images=self.get_similar_images(
                    imagenet_relative_path, group_score.id, n_images
                ),
            )
            final_predictions.append(class_prediction)

        return final_predictions
