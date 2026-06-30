# Panorama Stitching

## Overview
A computer vision project implementing advanced video processing algorithms to create seamless high-resolution panoramas from a sequence of video frames or images.

## Technical Details
- **Tech Stack:** Python, OpenCV, NumPy
- **Core Functionality:** Feature detection, homography matrix calculation, and image warping.

## Interesting Concept
Instead of naively stitching images left-to-right, the algorithm calculates a global coordinate system, allowing it to project all frames onto a single cylindrical or spherical plane to maintain perspective correctness.

## Key Challenge
**Accumulated Geometric Distortion:** When stitching many frames sequentially, small errors in the homography estimation accumulate, causing the final panorama to severely warp or drift off-screen.
*Solution:* Implemented an anchor-frame strategy where transformations are computed relative to a central frame rather than just the previous frame, significantly reducing accumulated error.
