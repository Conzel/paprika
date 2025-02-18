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

from ._analysis import NeuralNetworkAnalysis, ClassPrediction, SaveFeatures
from paprika.ml._config import *
from ..ui._config import selected_layers

class Inceptionv1Analysis(NeuralNetworkAnalysis):
    """
    Performs image analysis using Inceptionv1.
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
    device = get_device()
    model = inceptionv1(pretrained=True).eval().to(device)

    def __init__(self, img: np.ndarray):
        """Performs the analysis on the given image."""
        # resize image to size 224x224
        img = img / 255
        img = resize(img, (224, 224), anti_aliasing=True)
        img = img * 255
        img = np.require(img, np.uint8, "C")

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
        image = self.preprocess_image(self.image).unsqueeze(0).to(self.device)
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
            model=self.model,
            target_layers=target_layers,
            use_cuda=torch.cuda.is_available(),
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

    def get_similar_images(
        self, path, class_id, nr_images, feature_vector
    ) -> List[str]:
        """
        Returns a list of size nr_images containing the full paths of most similar images
        """

        # class_id='n01440764' #for testing purposes
        image_full_paths = []
        enough_images_in_class = True

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
            dot_product = feature_vector[np.newaxis] @ tensor
            indices = torch.topk(dot_product, nr_images).indices.cpu().detach().numpy()

            image_full_paths = []

            for idx in indices[0]:

                # if not enough images in class (shouldn't be the case), the first (and only?) image is shown multiple times to avoid crash of ui
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
        # calculate predictions and activation in feature space
        image = self.preprocess_image(self.image).unsqueeze(0).to(self.device)
        # layer_string = "mixed5b"
        # layer_number = filter_strings_to_numbers[layer_string]
        # layer = list(self.model.children())[layer_number]
        # activations = SaveFeatures(layer)
        predictions = self.model(image)[0]
        # mean_act = [
        #     activations.features[0, i].mean()
        #     for i in range(activations.features.shape[1])
        # ]
        # act_sum = torch.sum(torch.tensor(mean_act))
        # mean_act = torch.tensor(mean_act) / act_sum

        # copy score dictionary
        group_score_dict = copy.deepcopy(self.group_score_dict)

        # change group_score_dict to contain the maximum score of each group and the corresponding id
        for subclass_position in range(1, 1001):
            prediction = predictions[subclass_position].item()
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
                    imagenet_relative_path, group_score.id, n_images, predictions
                ),
                # similar_images=self.get_similar_images(imagenet_relative_path, group_score.id, n_images, mean_act),
            )
            final_predictions.append(class_prediction)

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
        files = [file for file in files if ".JPEG" in file]
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
        # dummy data with longest labels
        class_predictions = []
        subclass_group_dict = construct_subclass_group_dict()
        subclass_group_dict = dict(
            sorted(
                subclass_group_dict.items(),
                key=lambda item: len(item[1].german_group),
                reverse=True,
            )
        )
        longest_subclass_group_list = list(subclass_group_dict.items())[:50]
        shortest_subclass_group_list = list(subclass_group_dict.items())[-50:]
        possible_activations = np.linspace(62.23, 137.88).tolist()
        activations = random.choices(possible_activations, k=n_predictions)
        for activation in activations:
            if random.randint(0, 10) % 3 != 0:
                subclass = random.choice(longest_subclass_group_list)[1]
            else:
                subclass = random.choice(shortest_subclass_group_list)[1]
            images = self.get_random_images_from_folder(
                f"{imagenet_relative_path}{subclass.id}", n_images
            )
            class_predictions.append(
                ClassPrediction(
                    subclass.german_group, subclass.english_group, activation, images
                )
            )

        # random labels
        # class_predictions = []
        # folder_path = "../" + imagenet_relative_path
        # possible_activations = np.linspace(0.08, 4.532).tolist()
        # activations = random.choices(possible_activations, k=n_predictions)
        # activations.sort(reverse=True)
        # classes = open("../paprika/ml/LOC_synset_mapping.txt").read().split("\n")[:-1]
        # for activation in activations:
        #     prediction_class = random.choice(classes)
        #     class_id, label_list = prediction_class.split(None, 1)
        #     if "," in label_list:
        #         label, label_list = label_list.split(",", 1)
        #     else:
        #         label = label_list
        #     images = self.get_random_images_from_folder(
        #         f"{folder_path}{class_id}", n_images
        #     )
        #     class_predictions.append(ClassPrediction(label, activation, images))

        # prezel dummy data
        # activations = [87.4, 3.1, 1.7, 1.6, 0.9]
        # predictions = [("Brezel", "Pretzel", "n07695742"), ("Schnalle", "Buckle", "n02910353"), ("Haken", "Hook", "n03532672"), ("Wurm", "Worm", "n01924916"), ("Lupe", "Loupe", "n03692522")]
        # for i in range(5):
        #     activation = activations[i]
        #     german_label, english_label, class_id = predictions[i]
        #     images = self.get_random_images_from_folder(f"{folder_path}{class_id}", n_images)
        #     class_predictions.append(ClassPrediction(f"{german_label}|{english_label}", activation, images))
        return class_predictions


class TestImagesAnalysis(NeuralNetworkAnalysis):
    """
    This class can be used to test that all images (both Imagenet and filter visualisations)
    can be opened in Qt and the app does not crash when doing so.
    It iterates through all Imagenet images and all filter visualisations,
    and it returns all of them at some point.
    Once all images have been iterated through, it defaults to returning the first image of the list.
    """

    def get_all_images_from_folder(path) -> List[str]:
        """
        Returns a list containing the full paths of all images in the folder.
        """
        files = os.listdir(path)
        files = [file for file in files if ".JPEG" in file]
        files = [
            os.path.abspath(os.path.expanduser(os.path.expandvars(f"{path}/{file}")))
            for file in files
        ]
        return files

    # construct a list containing all Imagenet images
    # version 1: add all image paths from the Imagenet folders
    # subclass_ids = [subclass.id for subclass in construct_subclass_group_dict().values()]
    # all_imagenet_images = []
    # for subclass_id in subclass_ids:
    #     all_imagenet_images.extend(get_all_images_from_folder(f"../imagenet/{subclass_id}"))
    # print(f"Number of Imagenet images: {len(all_imagenet_images)}")

    # version 2: similarly to how the paths are obtained in Inceptionv1Analysis
    all_imagenet_images = []
    subclass_group_dict = construct_subclass_group_dict()
    class_ids = [subclass.id for subclass in subclass_group_dict.values()]
    device = torch.device("cuda")
    model = inceptionv1(pretrained=True).eval().to(device)
    preprocess_image = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    image = (
        preprocess_image(np.random.random((224, 224, 3)).astype("float32"))
        .unsqueeze(0)
        .to(device)
    )
    feature_activations = SaveFeatures(list(model.children())[filter_strings_to_numbers["mixed5b"]])
    predictions = model(image)[0]
    mean_act = []
    for i in range(feature_activations.features.shape[1]):
        mean_act.append(feature_activations.features[0, i].mean())
    act_sum = torch.sum(torch.tensor(mean_act))
    mean_act = (torch.tensor(mean_act) / act_sum).to(device)

    for class_id in class_ids:
        tensor = (
            torch.load(
                f"{imagenet_relative_path}{class_id}/{class_id}_activation_tensor.pt"
            )
            .float()
            .to(device)
        )
        dictionary = json.load(
            open(f"{imagenet_relative_path}{class_id}/{class_id}_dictionary.json")
        )
        dot_product = mean_act[np.newaxis] @ tensor
        for idx in range(dot_product.shape[1]):
            image_name = dictionary[str(idx)]
            all_imagenet_images.append(
                imagenet_relative_path + str(class_id) + "/" + image_name
            )
    print(f"Number of Imagenet images: {len(all_imagenet_images)}")

    # construct a list containing all filter visualisations
    # the image paths are obtained similarly to how they are obtained in Inceptionv1Analysis
    all_visualisation_images = []
    model = inceptionv1(pretrained=True).eval().to(device)
    layer_names = filter_strings_to_numbers.keys()
    layer_activations = {}
    for layer_name in layer_names:
        layer_nr = filter_strings_to_numbers[layer_name]
        layer = list(model.children())[layer_nr]
        layer_activations[layer_name] = SaveFeatures(layer)
    predictions = model(image)[0]
    folder_path = "../" + visualisations_relative_path
    for layer_name in layer_names:
        activations = layer_activations[layer_name]
        filters = list(range(activations.features.shape[1]))
        for filter in filters:
            image_path = os.path.abspath(
                os.path.expanduser(
                    os.path.expandvars(f"{folder_path}{layer_name}/{filter}.jpg")
                )
            )
            all_visualisation_images.append(image_path)
    print(f"Number of visualisation images: {len(all_visualisation_images)}")

    def __init__(self, img: np.ndarray):
        self.image = img

    def get_most_activated_filters(
        self, layer_string: str, n: int
    ) -> List[Tuple[str, int, float]]:
        """
        Returns the filter visualisations that have not been returned yet and deletes them from the list.
        """
        filters = []
        for i in range(n):
            if len(self.all_visualisation_images) == 1:
                filters.append((self.all_visualisation_images[0], 0, 0))
            else:
                filters.append((self.all_visualisation_images.pop(), 0, 0))
        return filters

    def get_saliency_map(self) -> np.ndarray:
        """
        Returns the image itself.
        """
        return self.image

    def get_class_predictions(
        self, n_predictions: int, n_images: int
    ) -> List[ClassPrediction]:
        class_predictions = []
        for i in range(n_predictions):
            images = []
            for j in range(n_images):
                if len(self.all_imagenet_images) == 1:
                    images.append(self.all_imagenet_images[0])
                else:
                    images.append(self.all_imagenet_images.pop())
                if len(self.all_imagenet_images) % 10000 == 0:
                    print()
                    print()
                    print(len(self.all_imagenet_images))
                    print()
                    print()
            class_predictions.append(
                ClassPrediction(
                    "Geschirrspülmaschine", "Geschirrspülmaschine", 100.0, images
                )
            )
        return class_predictions
