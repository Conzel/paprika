import os

from paprika.cam import BufferlessVideoCapture, DummyCamera, CroppingVideoCapture
from paprika.ml._analysis import DummyAnalysis, Inceptionv1Analysis
from paprika.ui._ui import UserInterface


print("starting camera...")
camera = CroppingVideoCapture(0)
print("camera started")
ui = UserInterface(camera, Inceptionv1Analysis)
ui.run()
