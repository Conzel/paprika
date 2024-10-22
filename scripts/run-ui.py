import os

from paprika.cam import BufferlessVideoCapture, DummyCamera, CroppingVideoCapture
from paprika.ml._analysis import (
    Inceptionv1FasterAnalysis,
)
from paprika.ui._ui import UserInterface


print("starting camera...")
camera = BufferlessVideoCapture(0)
print("camera started")
ui = UserInterface(camera, Inceptionv1FasterAnalysis)
ui.run()
