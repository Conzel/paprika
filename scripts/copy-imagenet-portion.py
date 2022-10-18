"""
Copies a portion of imagenet to the /imagenet folder.

"""
import os
import random
import shutil
import imagesize

nr_images_per_class = 15
source_folder = ...
destination_folder = "../imagenet/"

mapping_lines = open("../paprika/ml/LOC_synset_mapping.txt").read().split("\n")[:-1]
for mapping_line in mapping_lines:
    class_id, class_labels = mapping_line.split(None, 1)
    all_images = os.listdir(f"{source_folder}{class_id}")
    selected_images = []
    for image in all_images:
        source_image = f"{source_folder}{class_id}/{image}"
        width, height = imagesize.get(source_image)
        ratio = width / height
        # select images that are not too wide or too tall
        if 0.85 <= ratio <= 1.35:
            selected_images.append(image)
    # copy a random sample of the selected images
    sample_images = random.sample(selected_images, k=nr_images_per_class)
    for sample_image in sample_images:
        source_image = f"{source_folder}{class_id}/{sample_image}"
        destination_images = f"{destination_folder}{class_id}/{sample_image}"
        if not os.path.exists(f"{destination_folder}{class_id}"):
            os.mkdir(f"{destination_folder}{class_id}")
        shutil.copyfile(source_image, destination_images)
