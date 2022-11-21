import csv
import os
import random
import json
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


class Inceptionv1Analysis(NeuralNetworkAnalysis):
    """
    Performs image analysis using Inceptionv1.
    """

    def __init__(self, img: np.ndarray):
        """Performs the analysis on the given image."""
        self.model = inceptionv1(pretrained=True).eval()
        self.image = Image.fromarray(img)
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
        # calculate layer activations
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
        target_layers = [
            list(self.model.children())[136],
            list(self.model.children())[119],
            list(self.model.children())[128],
        ]  # lucent implementation

        # Construct the CAM object once, and then re-use it on many images:
        cam = GradCAMPlusPlus(
            model=self.model, target_layers=target_layers, use_cuda=False
        )
        targets = None  # [ClassifierOutputTarget(1000)]
        # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
        img_to_tensor = T.PILToTensor()(self.image)
        input_tensor = img_to_tensor[None, :] / 255.0
        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=targets,
            eigen_smooth=False,
            aug_smooth=False,
        )
        # In this example grayscale_cam has only one image in the batch:
        grayscale_cam = grayscale_cam[0, :]
        rgb_image = np.asarray(T.ToPILImage()(img_to_tensor)) / 255.0
        visualization = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)

        return visualization

    def get_similar_images(self, path, class_id, nr_images, feature_vector) -> List[str]:
        """
        Returns a list of size nr_images containing the full paths of most similar images
        """

        #class_id='n01440764' #for testing purposes
        image_full_paths = []
        enough_images_in_class =True

        if nr_images !=0:
            tensor = torch.load(f"{path}{class_id}/{class_id}_activation_tensor.pt").float()
            dictionary =json.load( open( f"{path}{class_id}/{class_id}_dictionary.json" ) )

            if len(dictionary)<nr_images:
                print(f'class {class_id} does not contain enough images') #shouldn't happen, when whole image net is used
                enough_images_in_class = False
            dot_product = feature_vector[np.newaxis] @ tensor
            indices = torch.topk(dot_product, nr_images).indices.cpu().detach().numpy()

            image_full_paths = []

            for idx in indices[0]:

                #if not enough images in class (shouldn't be the case), the first (and only?) image is shown multiple times to avoid crash of ui
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
        #calculate predictions and activation in feature space
        image = self.preprocess_image(self.image).unsqueeze(0)
        layer_string = "mixed5b"
        layer_number = filter_strings_to_numbers[layer_string]
        layer = list(self.model.children())[layer_number]
        activations = SaveFeatures(layer)
        predictions = self.model(image)[0][1:-7]
        mean_act = [
            activations.features[0, i].mean()
            for i in range(activations.features.shape[1])
        ]
        sum = torch.sum(torch.tensor(mean_act))
        mean_act= torch.tensor(mean_act) / sum


        translations = self.read_csv("translations.csv")
        folder_path = "../" + imagenet_relative_path

        # Create Empty Dict with all Categories in it:
        translated_classes = self.create_translation_class_dict()
        label_to_class_id ={}

        #print("adding percentages")
        # For every prediction: add the percentage to the translated class
        for idx in range(1, 1000):
            label, class_number = labelConverter()[
                idx + 1
            ]

            prediction = predictions[idx].item()
            translated_class = translations[label]
            label_to_class_id[translated_class]=class_number
            translated_classes[translated_class] += prediction


        ordered_predictions = {
            k: v
            for k, v in sorted(
                translated_classes.items(), key=lambda x: x[1], reverse=True
            )
        }
        
        # Calculate sum of all percentages (not exaclty 100)
        sum = 0
        for elem in ordered_predictions:
            sum = sum + ordered_predictions[elem]

        final_predictions = []
        i = 0

        for pred_label in ordered_predictions:
            if i <= n_predictions - 1:
                class_prediction = ClassPrediction(
                    label=pred_label,
                    score=ordered_predictions[pred_label] / sum * 100,
                    similar_images=self.get_similar_images(
                folder_path, label_to_class_id[pred_label], n_images,mean_act
            ),
                )
                final_predictions.append(class_prediction)
                i = i + 1
            else:
                break


        return final_predictions

    def read_csv(self, file_path) -> Dict:
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

    def __init__(self, img: np.ndarray, delay: int = 0):
        """
        Initialisation sleeps for delay seconds.
        """
        super().__init__(img)
        time.sleep(delay)
        self.image = img

    def get_full_path(self, relative_path):
        """
        Returns the full path based on a relative path.
        """
        return os.path.abspath(os.path.expanduser(os.path.expandvars(relative_path)))

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
        filter_ids = random.sample(range(0, nr_filters), k=n)
        for i in range(n):
            filter_id = filter_ids[i]
            image_path = self.get_full_path(
                f"{folder_path}{layer_string}/{filter_id}.jpg"
            )
            filter_activation = activations[i]
            filters.append((image_path, filter_id, filter_activation))
        return filters

    def get_saliency_map(self) -> np.ndarray:
        """
        Returns a saliency map of the image. The saliency map has the same dimensions
        as the input image and is a heatmap of the most important pixels in the image.
        Returns RGB image with values in [0, 255].
        """
        return np.clip(4 * (self.image - 128) + 128, 0, 255)

    def get_random_images_from_folder(self, path, nr_images) -> List[str]:
        """
        Returns a list of size nr_images containing the full paths of randomly
        selected images in the folder specified by path.
        """
        files = os.listdir(path)
        images = random.sample(files, k=nr_images)
        image_full_paths = []
        for image in images:
            relative_path = path + "/" + image
            full_path = self.get_full_path(relative_path)
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
        class_predictions = []
        folder_path = "../" + imagenet_relative_path
        possible_activations = np.linspace(0.08, 4.532).tolist()
        activations = random.choices(possible_activations, k=n_predictions)
        activations.sort(reverse=True)
        classes = open("../paprika/ml/LOC_synset_mapping.txt").read().split("\n")[:-1]
        for activation in activations:
            prediction_class = random.choice(classes)
            class_id, label_list = prediction_class.split(None, 1)
            if "," in label_list:
                label, label_list = label_list.split(",", 1)
            else:
                label = label_list
            images = self.get_random_images_from_folder(
                f"{folder_path}{class_id}", n_images
            )
            class_predictions.append(ClassPrediction(label, activation, images))
        # activations = [87.4, 3.1, 1.7, 1.6, 0.9]
        # predictions = [("Brezel", "Pretzel", "n07695742"), ("Schnalle", "Buckle", "n02910353"), ("Haken", "Hook", "n03532672"), ("Wurm", "Worm", "n01924916"), ("Lupe", "Loupe", "n03692522")]
        # for i in range(5):
        #     activation = activations[i]
        #     german_label, english_label, class_id = predictions[i]
        #     images = self.get_random_images_from_folder(f"{folder_path}{class_id}", n_images)
        #     class_predictions.append(ClassPrediction(f"{german_label}|{english_label}", activation, images))
        return class_predictions
