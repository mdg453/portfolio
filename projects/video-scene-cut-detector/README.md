# Video Scene Cut Detector

## Overview
A minimal, pure-Python tool designed to automatically detect scene transitions and cuts in video streams, allowing for automated video segmentation and editing.

## Technical Details
- **Tech Stack:** Python, FFmpeg, OpenCV
- **Core Functionality:** Frame extraction, color histogram analysis, and thresholding algorithms.

## Interesting Concept
By analyzing the statistical distribution of colors rather than exact pixel values, the tool can understand the semantic similarity of frames, making it resilient to slight camera movements or noise.

## Key Challenge
**False Positives from Camera Motion:** Fast camera pans or large moving objects caused massive pixel differences, tricking the algorithm into detecting a scene cut where none existed.
*Solution:* Switched from direct pixel-wise difference calculations to comparing HSV color histograms across frames with an adaptive threshold, effectively ignoring motion while catching actual camera cuts.
