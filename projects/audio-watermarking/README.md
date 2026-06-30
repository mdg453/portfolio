# Audio Watermarking Utility

## Overview
A steganographic tool for embedding inaudible, robust spread-spectrum watermarks into digital audio files for copyright protection and tracking.

## Technical Details
- **Tech Stack:** Python, SciPy, NumPy
- **Core Functionality:** Signal modulation, DCT (Discrete Cosine Transform) processing, and psychoacoustic masking.

## Interesting Concept
The watermark is treated like a pseudo-random noise signal spread across a wide frequency band. It relies on the listener's inability to perceive slight modifications in the audio, effectively hiding data in plain sound.

## Key Challenge
**Surviving Audio Compression:** Simple LSB (Least Significant Bit) watermarks are easily destroyed by MP3 compression or low-pass filtering.
*Solution:* Embedded the spread-spectrum signal in the mid-frequency range of the DCT domain. This ensured the watermark was imperceptible due to human hearing thresholds, yet survived lossy compression algorithms that typically discard high frequencies.
