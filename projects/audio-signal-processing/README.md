# Audio Signal Processing

## Overview
A digital signal processing (DSP) project focused on advanced noise reduction and spectral analysis algorithms designed to clean up and enhance degraded audio recordings.

## Technical Details
- **Tech Stack:** MATLAB, Signal Processing Toolbox
- **Core Functionality:** Fast Fourier Transforms (FFT), digital filter design (FIR/IIR), and spectral subtraction.

## Interesting Concept
By moving from the time domain into the frequency domain, the algorithm can surgically identify and remove specific frequencies (like a 60Hz electrical hum) without affecting the overall voice quality.

## Key Challenge
**Degrading Voice Quality:** Standard low-pass or band-pass filtering removed the noise but also muffled the human voice, making it sound underwater.
*Solution:* Implemented an adaptive Wiener filter based on spectral subtraction. By estimating the noise profile during silent periods, the algorithm adaptively subtracted only the noise frequencies from the speech segments, preserving the crispness of the vocal track.
