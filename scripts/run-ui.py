import os

from paprika.cam import BufferlessVideoCapture, DummyCamera
from paprika.ml._analysis import DummyAnalysis
from paprika.ui._ui import UserInterface

# rotate all screens
os.system("xrandr --output HDMI-0 --rotate left")
os.system("xrandr --output DP-0 --rotate left")
os.system("xrandr --output DP-3 --rotate left")
os.system("xrandr --output DP-4 --rotate left")

print("starting camera...")
camera = DummyCamera(0)
# camera = BufferlessVideoCapture(-1)
print("camera started")
ui = UserInterface(camera)
ui.run()

# rotate all screens back
os.system("xrandr --output HDMI-0 --rotate normal")
os.system("xrandr --output DP-0 --rotate normal")
os.system("xrandr --output DP-3 --rotate normal")
os.system("xrandr --output DP-4 --rotate normal")
