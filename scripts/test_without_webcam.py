import cv2
import torch
import numpy as np
from PIL import Image
from lucent.modelzoo import inceptionv1
from paprika.ml import IMAGENET_CLASS_LIST, Inceptionv1Analysis, labelConverter
from torchvision import transforms
import matplotlib.pyplot as plt

model = inceptionv1(pretrained=True).eval()


preprocess_image = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
)

preprocess_tensor = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# test image
img = np.asarray(Image.open('test_images/remote_control.jpg'))
pil_img = Image.fromarray(img.copy())
im_cropped = preprocess_image(pil_img)
#im_tensor = preprocess_tensor(im_cropped).unsqueeze(0)

print('label',(Inceptionv1Analysis(im_cropped).get_class_predictions(3,1))[0].label)

print('score', (Inceptionv1Analysis(im_cropped).get_class_predictions(3,1))[0].score)
print('filters',Inceptionv1Analysis(im_cropped).get_most_activated_filters('mixed3a',3))
map = Inceptionv1Analysis(im_cropped).get_saliency_map()
plt.imshow(map)
plt.savefig("test_images/testNew.jpg")
while True:
    cv2.imshow("map",map)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Display the resulting frame
cv2.imshow("frame", np.asarray(im_cropped))

