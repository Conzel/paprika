# Setting up the paprika exhibit
## Staff instruction
### Plugging in the monitors
1. Each monitor is labelled 1-4 on the back. Put the monitors in the order that is indicated, from left to right. 
2. There are 4 cables. Letter (M or C) followed by a number (f.e. M1). For every monitor X, locate the corresponding cable end MX and plug it into the monitor (f.e. computer 1 needs cable end M1).
3. At the back of the computer, insert the cables ends marked C1-4 in the order 1-4, starting from the left. The first cable is HDMI, the rest are display port.

### Other hardware (mouse, keyboard, webcam)
One of the monitors has a USB cable, plug this into the monitor at one end and the computer on the other.
Then, plug the hardware into the monitor. Extra peripherals can be plugged into the computer directly.

### Software
The software should autostart when the PC is booted. 
Alternatively, the UI can be started via clicking the paprika icon in the application bar. 
A terminal will open and the UI will be shown in the four screens in a few seconds. The application bar is located on the left side on screen 1.

To quit the UI, press Ctrl + Q (wait a moment until all windows close).

# Technical information
## Monitors
### Plugging in
Ordering at the back:
1 4 3 2
(starting from HDMI, so right to left if the computer is facing you, and 
left to right if you do stuff in the back)

We want to have our computers ordered 1234.

This means you plug in the computers as follows 
(again, left to right if you are at the back of the PC):

  1      4     3     2
-----  ----- ----- -----
HDMI     DP    DP   DP

### Setting up
If it does not work immediately, press the windows key and enter `displays`. Set up the displays manually and rearrange them. Select `Portrait Right` in the orientation tab.

## Software
### Terminal
Some of the descriptions below require a terminal. To enter one, press the windows key and enter `Terminal`, then press enter or click on the terminal application.

To navigate the file tree in the terminal, use `cd` to switch your directory, and `ls` to list the current files.

### Configuration 
The software can be configured by browsing to `/home/paprika/paprika/ui` and editing the `_config.py` file.

Most setting should be left as-is. Lines 56-59 contain the monitor order,
which should be kept the same as long as the monitors are connected the correct way. Alternatively, one can swap the numbers around to make it fit to the desired monitor connection. 

In line 39, the analysis refresh rate is configured. This can take various values, depending on how fast the images should be analysed in the exhibition piece.

### Resetting the software
If some erroneous modifications have been made to the configuration, they can be undone by navigating to `/home/paprika/paprika/ui` and entering
```git reset --hard HEAD```
into the terminal.
