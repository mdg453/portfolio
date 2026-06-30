# FLL Robotics Controller

## Overview
Control software developed for competitive Lego Spike Prime robots in the FIRST Lego League (FLL). The codebase enables autonomous navigation, precise sensor-based alignment, and complex mechanical manipulation.

## Technical Details
- **Tech Stack:** Python, Pybricks (MicroPython)
- **Core Functionality:** PID control loops, sensor fusion, and asynchronous motor control.

## Interesting Concept
The robot uses gyroscopic feedback and light sensors to mathematically guarantee its position on the competition mat, removing the unpredictability of battery drain and wheel slippage.

## Key Challenge
**Erratic Line Following:** Raw data from the light sensors fluctuated wildly due to shadows and mat imperfections, causing the robot to violently oscillate while trying to follow a line.
*Solution:* Implemented a PID (Proportional-Integral-Derivative) controller coupled with a low-pass filter on the sensor readings. This smoothed out the noise and allowed the robot to aggressively, yet smoothly, track lines at high speeds.
