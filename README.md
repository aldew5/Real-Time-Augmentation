# Computer-Vision
Real-time video augmentation with ArUco markers. Markers with ids 1 and 2 will be augmented with the colors
blue and green respectively when either one or both of them are placed in the same frame. If a marker with
id three is also present in the frame it will be augmented with the combination of green and blue: green-blue.

This program was tested with a Raspberry Pi and a C270 HD webcam, but other compatible hardware should suffice.

# Requirements
- numpy
- opencv and opencv_contrib
- a web camera

# Instructions
Run *accumulator.py* with a webcam attached to your computer. You must print a page of ArUco markers to use 
for detection. Point the webcamera at the ArUco markers with ids 1 or 2 or 3 to see the real time video augmentation.
You must have the *_data* file placed in the same directory and you must place a clone of *opencv_contrib* in the *_data* file.

