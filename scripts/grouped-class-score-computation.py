import math
from typing import NamedTuple

import cv2
import pandas
import torchvision.transforms as T
from lucent.modelzoo import inceptionv1

from paprika.ui._helper import image_for_analysis


class Subclass(NamedTuple):
    id_1: str
    german_group: str
    english_group: str


preprocess_image = T.Compose(
    [
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

for image_name in ["bird_drawing", "bullet_train", "carbonara", "carpet", "convertible", "liger", "mixed_dog", "table", "wall", "whiteboard"]:
    image = image_for_analysis(cv2.imread(f"test_images/{image_name}.jpg"))
    model = inceptionv1(pretrained=True).eval()

    classes_df = pandas.read_csv("grouped_classes.csv", delimiter="|")
    subclass_id_map = {}
    group_score_map = {}
    for _, row in classes_df.iterrows():
        subclass_id_map[row["id_2"]] = Subclass(row["id_1"], row["german_group"], row["english_group"])
        if (row["german_group"], row["english_group"]) not in group_score_map:
            group_score_map[(row["german_group"], row["english_group"])] = []

    img = preprocess_image(image).unsqueeze(0)
    predictions = model(img)[0]

    for subclass_id in range(1, 1001):
        prediction = predictions[subclass_id].item()
        subclass = subclass_id_map[subclass_id]
        group_score_map[(subclass.german_group, subclass.english_group)].append(prediction)

    group_max_score = {}
    group_mean_score = {}
    group_top10percent_mean_score = {}
    group_sum_score = {}

    for group_id, group_scores in group_score_map.items():
        group_max_score[group_id] = max(group_scores)
        group_mean_score[group_id] = sum(group_scores) / len(group_scores)
        sorted_top10percent_scores = sorted(group_scores, reverse=True)[: math.ceil(len(group_scores) / 10)]
        group_top10percent_mean_score[group_id] = sum(sorted_top10percent_scores) / len(sorted_top10percent_scores)
        group_sum_score[group_id] = sum(group_scores)

    group_max_score_sum = sum(group_max_score.values())
    group_max_score = {key: value / group_max_score_sum for key, value in group_max_score.items()}
    group_mean_score_sum = sum(group_mean_score.values())
    group_mean_score = {key: value / group_mean_score_sum for key, value in group_mean_score.items()}
    group_top10percent_mean_score_sum = sum(group_top10percent_mean_score.values())
    group_top10percent_mean_score = {key: value / group_top10percent_mean_score_sum for key, value in group_top10percent_mean_score.items()}
    group_sum_score_sum = sum(group_sum_score.values())
    group_sum_score = {key: value / group_sum_score_sum for key, value in group_sum_score.items()}

    group_max_score = dict(sorted(group_max_score.items(), key=lambda item: item[1], reverse=True))
    group_mean_score = dict(sorted(group_mean_score.items(), key=lambda item: item[1], reverse=True))
    group_top10percent_mean_score = dict(sorted(group_top10percent_mean_score.items(), key=lambda item: item[1], reverse=True))
    group_sum_score = dict(sorted(group_sum_score.items(), key=lambda item: item[1], reverse=True))

    print(image_name)

    print("\nMAX")
    for group_id, group_score in list(group_max_score.items())[: 5]:
        print(f"{round(group_score, 5)}   {group_id}")

    print("\nMEAN")
    for group_id, group_score in list(group_mean_score.items())[: 5]:
        print(f"{round(group_score, 5)}   {group_id}")

    print("\nTOP10PERCENT MEAN")
    for group_id, group_score in list(group_top10percent_mean_score.items())[: 5]:
        print(f"{round(group_score, 5)}   {group_id}")

    print("\nSUM")
    for group_id, group_score in list(group_sum_score.items())[: 5]:
        print(f"{round(group_score, 5)}   {group_id}")

    print()
    print()
    print()