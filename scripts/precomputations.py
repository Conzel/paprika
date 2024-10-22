"""
precompute activation in feature space and save it as tensor. Additionally, save dictionary (mapping images to numbers)
delete images that are too small to be passed through the model or with undesired ratio

"""
import os

# import imagesize
import json
import torch
from lucent.modelzoo import inceptionv1
import torchvision.transforms as T
import numpy as np
from PIL import Image

filter_strings_to_numbers = {
    "mixed3a": 70,
    "mixed3b": 78,
    "mixed4a": 87,
    "mixed4b": 95,
    "mixed4c": 103,
    "mixed4d": 111,
    "mixed4e": 119,
    "mixed5a": 128,
    "mixed5b": 136,
}


class SaveFeatures:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output

    def close(self):
        self.hook.remove()


if __name__ == "__main__":

    # image transforms:
    preprocess_image = T.Compose([T.Resize(256), T.CenterCrop(224)])
    to_tensor = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    source_folder = "../imagenet/"
    # choose layer
    layer_string = "mixed5b"
    layer_number = filter_strings_to_numbers[layer_string]
    device = torch.device("cuda")
    model = inceptionv1(pretrained=True).eval().to(device)

    all_classes = os.listdir(f"{source_folder}")
    counter = 0
    for class_id in all_classes:
        print(counter)
        print(class_id)
        counter += 1

        all_images = os.listdir(f"{source_folder}/{class_id}")

        # continue to next class if precomputations already done for this class
        if f"{class_id}_activation_tensor.pt" in all_images and f"{class_id}_dictionary.json" in all_images:
            continue

        activation_all_images = np.zeros(
            (len(all_images), 1024)
        )  # set second argument to length of layer activation
        dictionary_imagenames = {}

        i = -1
        for image in all_images:
            i += 1
            image_path = f"{source_folder}/{class_id}/{image}"
            img = np.asarray(Image.open(image_path))
            shape = img.shape
            width, height = shape[0], shape[1]
            min_side = min(width, height)

            dictionary_imagenames[i] = str(image)

            if min_side < 224:  # necessary condition (otherwise error is thrown)
                # os.remove(image_path)
                # print('deleted')
                # print(img.shape)
                continue


            # load and preprocess image
            pil_img = Image.fromarray(img.copy())
            # skip black and white images
            if len(pil_img.getbands()) != 3:
                # os.remove(image_path)
                # print('deleted')
                # print(img.shape)
                continue

            im_cropped = preprocess_image(pil_img)
            image = to_tensor(im_cropped).unsqueeze(0).to(device)

            # compute layer activation
            layer = list(model.children())[layer_number]
            activations = SaveFeatures(layer)
            predictions = model(image)[0]

            mean_act = [
                activations.features[0, i].mean()
                for i in range(activations.features.shape[1])
            ]
            activation_all_images[i] = torch.tensor(mean_act) / torch.sum(
                torch.tensor(mean_act)
            )  # save normalizes feature vector

        # save tensor
        torch.save(
            torch.tensor(activation_all_images.T),
            f"{source_folder}{class_id}/{class_id}_activation_tensor.pt",
        )

        # save dict
        json.dump(
            dictionary_imagenames,
            open(f"{source_folder}{class_id}/{class_id}_dictionary.json", "w"),
        )
