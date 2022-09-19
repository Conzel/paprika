from paprika.cam import BufferlessVideoCapture
from paprika.ui._ui import UserInterface

print("starting camera...")
# camera = DummyCamera(0)
camera = BufferlessVideoCapture(0)
print("camera started")
ui = UserInterface(camera)
ui.run()
