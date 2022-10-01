from typing import List

import numpy as np
from PIL import Image

from paprika.cam import DummyCamera, BufferlessVideoCapture
from paprika.ml import DummyAnalysis, Inceptionv1Analysis
from paprika.ml._analysis import ClassPrediction
from paprika.ui._helper import image_for_analysis


def save_image(img: np.ndarray, name: str):
    image = Image.fromarray(img)
    image.save(f"analysis-without-gui/{name}.jpg", mode="RGB")


camera = BufferlessVideoCapture(0)
camera_image = camera.read()
# check image_for_analysis function for the specific formatting of the image
image_to_analyse = image_for_analysis(camera_image)
save_image(image_to_analyse, "image_to_analyse")

analysis = DummyAnalysis(image_to_analyse)
# analysis = Inceptionv1Analysis(image_to_analyse)

# saliency_map should be an RGB-formatted ndarray with values in [0, 255]
saliency_map = analysis.get_saliency_map()
assert type(saliency_map) == np.ndarray
save_image(saliency_map, "saliency_map")

# for one layer, obtain all filters
layer = "mixed3a"
filter_nr = 256
# get list of (image_path, filter_id, filter_activation)
all_filters_in_layer = analysis.get_most_activated_filters(layer, 256)

# check that all filter IDs are returned
all_filter_ids = []
for filter in all_filters_in_layer:
    all_filter_ids.append(filter[1])
assert sorted(all_filter_ids) == sorted(range(filter_nr))

# check that the filter activations add up to about 100
activation_sum = 0
for filter in all_filters_in_layer:
    activation_sum += filter[2]
print(f"100 ~ {activation_sum}")

# get prediction list with all classes
imagenet_class_nr = 1000
class_predictions = analysis.get_class_predictions(imagenet_class_nr, 0)

# check that prediction scores add up to about 100
prediction_score_sum = 0
for prediction in class_predictions:
    prediction_score_sum += prediction.score
print(f"100 ~ {prediction_score_sum}")
