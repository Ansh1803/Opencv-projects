Hand Position Tracking & Risk Zone Detection (OpenCV)

This project tracks a hand using basic HSV skin segmentation and detects whether the hand is entering or approaching a defined danger region on the screen.
What This Does

Detects your hand using a crude HSV mask

Finds the largest contour → assumes that’s the hand

Calculates hand center

Measures distance between hand center and a predefined "virtual box"

Classifies situations into 3 states: Safe, Warning, Danger 

Requirements
opencv
python
numpy
