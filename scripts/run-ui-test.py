import os

from paprika.cam import BufferlessVideoCapture, DummyCamera, CroppingVideoCapture
from paprika.ml._archived_analysis import TestImagesAnalysis
from paprika.ui._ui import UserInterface


print("starting camera...")
camera = BufferlessVideoCapture(0)
print("camera started")
ui = UserInterface(camera, TestImagesAnalysis)
ui.run()
