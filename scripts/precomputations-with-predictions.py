"""
Precompute the predictions for each ImageNet image and save it as a tensor.
"""
import os
# import imagesize
import json
import torch
from lucent.modelzoo import inceptionv1
import torchvision.transforms as T
import numpy as np
from PIL import Image

if __name__ == "__main__":

    # image transforms:
    preprocess_image = T.Compose(
        [
            T.Resize(224),
            T.CenterCrop(224),
        ]
    )
    to_tensor = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    source_folder = '../imagenet/'

    device = torch.device("cuda")
    model = inceptionv1(pretrained=True).eval().to(device)

    all_classes = os.listdir(f"{source_folder}")
    counter = 0
    for class_id in all_classes:
        print(counter)
        print(class_id)
        counter += 1

        if counter <= 30:
            continue

        all_images = os.listdir(f"{source_folder}/{class_id}")
        activation_all_images = np.zeros((len(all_images), 1008))  # set second argument to length of layer activation
        dictionary_imagenames = {}

        i = -1
        for image in all_images:
            i += 1
            image_path = f"{source_folder}/{class_id}/{image}"
            img = np.asarray(Image.open(image_path))
            shape = img.shape
            width, height = shape[0], shape[1]
            min_side = min(width, height)

            if min_side < 224:  # necessary condition (otherwise error is thrown)
                # os.remove(image_path)
                # print('deleted')
                # print(img.shape)
                continue

            dictionary_imagenames[i] = str(image)
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

            predictions = model(image)[0]
            activation_all_images[i] = predictions.detach().cpu().numpy()

        # save tensor
        torch.save(torch.tensor(activation_all_images.T), f"{source_folder}{class_id}/{class_id}_activation_tensor.pt")

        # save dict
        json.dump(dictionary_imagenames, open(f"{source_folder}{class_id}/{class_id}_dictionary.json", 'w'))
