# Self Tuning Piano 🎹

An advanced embedded systems and machine learning project featuring real-time frequency analysis and a trained ML model for intelligent automated piano tuning.

## Project Overview
The Self Tuning Piano utilizes an Arduino Portenta to capture and process audio signals from the piano strings. The acoustic data is then fed into a Python-based controller that utilizes machine learning to determine the precise tuning adjustments needed.

## Key Features
- **Frequency Analysis**: High-precision Fast Fourier Transform (FFT) and frequency detection algorithms running in real-time to analyze the pitch of each string.
- **Sophisticated Noise Filtering**: Advanced DSP (Digital Signal Processing) techniques are implemented to clean the raw audio, stripping away environmental noise and isolating the fundamental harmonic frequencies of the piano.
- **Machine Learning Tuner Model**: A custom-trained machine learning model (located in the `python_controller/` directory) that learns the tuning curve of the specific piano and predicts the optimal tension adjustments based on historical tuning data and acoustic characteristics.

## Architecture
- `arduino_portenta_project.ino`: The embedded firmware responsible for high-rate audio sampling and on-device preliminary signal processing.
- `python_controller/`: Contains the ML models (`tuner_model.pkl`) and data processing pipelines (`ml_tuner.py`, `main.py`) that interface with the hardware to orchestrate the mechanical tuning.

## Technologies
- C++ / Arduino Portenta
- Python (Machine Learning, DSP)
- Fast Fourier Transform (FFT) algorithms
