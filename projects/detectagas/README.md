# DetectaGas

## Overview
An IoT-based hardware-software integration project developed for the Google Developer Student Club (GDSC), designed to monitor environments and detect hazardous gas leaks in real-time.

## Technical Details
- **Tech Stack:** C++, Embedded Systems, IoT Sensors
- **Core Functionality:** Real-time sensor polling, threshold alerting, and data transmission.

## Interesting Concept
By pushing the detection logic directly onto the edge device (microcontroller), the system guarantees immediate, life-saving alerts without relying on a constant cloud connection or battling network latency.

## Key Challenge
**Sensor Calibration & False Positives:** Analog gas sensors are highly susceptible to temperature changes and environmental humidity, frequently triggering false alarms.
*Solution:* Developed an adaptive moving-average algorithm in C++ that dynamically calibrates the baseline threshold based on ambient environmental conditions, drastically reducing false positives while maintaining sensitivity.
