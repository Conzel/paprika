import os

from paprika.cam import BufferlessVideoCapture, DummyCamera, CroppingVideoCapture
from paprika.ml._analysis import (
    DummyAnalysis,
    Inceptionv1Analysis,
    Inceptionv1FasterAnalysis,
    TestImagesAnalysis,
)
from paprika.ui._ui import UserInterface


print("starting camera...")
# camera = CroppingVideoCapture(0)
camera = BufferlessVideoCapture(0)
print("camera started")
ui = UserInterface(camera, Inceptionv1FasterAnalysis)
ui.run()
