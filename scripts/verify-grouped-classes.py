import os
import random
import time

import pandas
from matplotlib import image as mpimg
from matplotlib import pyplot as plt

start_group_index = 0
nr_images_per_subclass = 3
sleep_seconds = 1


def get_random_images_from_folder(path, nr_images):
    files = os.listdir(path)
    imgs = []
    for file in files:
        if ".JPEG" in file:
            imgs.append(file)
    imgs = random.sample(imgs, k=nr_images)
    paths = []
    for img in imgs:
        relative_path = path + "/" + img
        paths.append(
            os.path.abspath(os.path.expanduser(os.path.expandvars(relative_path)))
        )
    return paths


translations_df = pandas.read_csv("grouped_classes.csv", delimiter="|")
grouped_classes_df = (
    translations_df.sort_values("english_group")
    .groupby(["english_group", "german_group"])
    .agg(list)
    .reset_index()
)

for index, row in grouped_classes_df.iterrows():
    if index < start_group_index:
        continue
    imagenet_id = row["id_1"]
    german_group = row["german_group"]
    english_group = row["english_group"]
    english_labels = row["english_labels"]
    nr_subgroups = len(english_labels)

    print()
    print(f"{index} / {len(grouped_classes_df) - 1}")
    print(f"{english_group}   {german_group}")
    print()
    for subgroup_index in range(nr_subgroups):
        print(f"{imagenet_id[subgroup_index]}   {english_labels[subgroup_index]}")
        images = get_random_images_from_folder(
            f"../imagenet/{imagenet_id[subgroup_index]}", nr_images_per_subclass
        )
        figure, axes = plt.subplots(1, nr_images_per_subclass)
        for image_index in range(nr_images_per_subclass):
            axes[image_index].imshow(mpimg.imread(images[image_index]))
            axes[image_index].get_xaxis().set_visible(False)
            axes[image_index].get_yaxis().set_visible(False)
        figure.set_dpi(200)
        figure.tight_layout()
        figure.show()
        time.sleep(sleep_seconds)

    print()
    input("press enter to go to next class\n")
