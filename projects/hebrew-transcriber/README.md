# Hebrew Transcriber

## Overview
A high-accuracy, offline Hebrew speech-to-text transcription tool. It leverages state-of-the-art machine learning models to convert audio and video files into accurately punctuated Hebrew text.

## Technical Details
- **Tech Stack:** Python, OpenAI Whisper (Fine-tuned), FFmpeg
- **Core Functionality:** Audio extraction from video, audio normalization, and deep-learning-based transcription.

## Interesting Concept
The tool can be seamlessly integrated into existing workflows as it operates entirely locally, ensuring complete data privacy for sensitive recordings, which is a major advantage over cloud-based transcription APIs.

## Key Challenge
**Memory Constraints on Large Files:** Transcribing long audio files natively with the Whisper model led to out-of-memory (OOM) errors and degraded accuracy over time.
*Solution:* Leveraged FFmpeg to intelligently chunk the audio files into smaller, overlapping segments before passing them to the inference engine, stitching the text back together seamlessly.
