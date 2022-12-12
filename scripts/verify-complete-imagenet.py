"""
Verify that all ImageNet folders have at least 732 images.
"""

import os

from paprika.ml._label_converter import construct_subclass_group_dict

def count_images_in_folder(path):
    files = os.listdir(path)
    files = [file for file in files if ".JPEG" in file]
    return len(files)

subclass_group_dict = construct_subclass_group_dict()
assert(len(subclass_group_dict.values()) == 1000)

incomplete_classes = []

for subclass in subclass_group_dict.values():
    class_id = subclass.id
    folder_path = f"../imagenet/{class_id}"
    nr_images = count_images_in_folder(folder_path)
    if nr_images < 732:
        incomplete_classes.append([class_id, nr_images])

incomplete_classes.sort(key=lambda variable: variable[0])
for incomplete_class in incomplete_classes:
    print(f"{incomplete_class[0]}   {incomplete_class[1]}")