from paprika.cam import BufferlessVideoCapture, DummyCamera
from paprika.ml._analysis import DummyAnalysis
from paprika.ui._ui import UserInterface

print("starting camera...")
# camera = DummyCamera(0)
camera = BufferlessVideoCapture(-1)
print("camera started")
ui = UserInterface(camera)
ui.run()