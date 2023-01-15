import cv2
import torch
import numpy as np
from PIL import Image
from paprika.ml import IMAGENET_CLASS_LIST
from paprika.cam import BufferlessVideoCapture
from torchvision import transforms

model = torch.hub.load("pytorch/vision:v0.10.0", "googlenet", pretrained=True).eval()


preprocess_image = transforms.Compose(
    [transforms.Resize(256), transforms.CenterCrop(224)]
)

preprocess_tensor = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# define a video capture object
vid = BufferlessVideoCapture(0)

while True:

    # Capture the video frame
    # by frame
    frame = vid.read()

    pil_img = Image.fromarray(frame.copy())
    im_cropped = preprocess_image(pil_img)
    im_tensor = preprocess_tensor(im_cropped).unsqueeze(0)

    # Display the resulting frame
    cv2.imshow("frame", np.asarray(im_cropped))
    prediction = torch.nn.functional.softmax(model(im_tensor)[0], dim=0).argmax().item()
    print(IMAGENET_CLASS_LIST[prediction])

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Destroy all the windows
cv2.destroyAllWindows()
