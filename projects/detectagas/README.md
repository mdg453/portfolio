# DetectaGas: Aerial CV Anomaly Detection

## Overview
A Computer Vision (CV) based Proof of Concept (POC) designed to detect potential gas leaks along pipelines using aerial and satellite imagery. By analyzing vegetation discoloration and structural anomalies around pipelines, the system can autonomously flag potential hazardous leaks.

## Technical Details
- **Tech Stack:** Python, OpenCV (cv2), NumPy
- **Core Functionality:** HSV color space transformation, morphological noise reduction, contour detection, and bounding box rendering.

## Interesting Concept
Instead of relying solely on expensive, on-the-ground hardware sensors scattered across thousands of miles of pipeline, this approach leverages overhead imagery. It identifies leaks indirectly by detecting the environmental impact of the gas (e.g., dead or yellowing vegetation in an otherwise green field).

## Key Challenge
**Color Thresholding & Noise:** Natural environments have massive color variance. Slight shadows or dirt patches can trigger false positives.
*Solution:* Transformed the images into the HSV color space, which is far more robust to lighting changes than BGR. Applied a combination of morphological operations (erosion followed by dilation) to filter out small noisy patches, ensuring only large, significant anomalies are flagged for review.
