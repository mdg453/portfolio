# Pyramid Blending & Hybrid Images

## Overview
A classic computer vision project focusing on seamless image blending and the construction of hybrid images using multi-scale frequency analysis.

## Technical Details
- **Tech Stack:** Python, NumPy, SciPy
- **Core Functionality:** Gaussian pyramids, Laplacian pyramids, and frequency domain manipulation.

## Interesting Concept
By exploiting how the human visual system processes spatial frequencies, hybrid images are created that look like one object up close (high frequencies) and a completely different object from a distance (low frequencies).

## Key Challenge
**Ghosting Artifacts in Blending:** Simple alpha blending causes noticeable seams and ghosting when combining images with different textures and colors.
*Solution:* Implemented a Laplacian pyramid blending technique that blends the images across multiple frequency bands independently, resulting in a mathematically seamless transition that preserves high-frequency details without harsh edges.
