# SevenConverter

## Overview
A robust desktop application that serves as a multi-threaded wrapper for FFMPEG, allowing users to batch convert, compress, and manipulate video and audio files through an intuitive GUI.

## Technical Details
- **Tech Stack:** C# / .NET, FFMPEG, WPF/WinForms
- **Core Functionality:** Process management, batch execution, and UI asynchronous updates.

## Interesting Concept
The tool abstracts away the intense command-line complexity of FFMPEG, programmatically generating complex filter graphs based on user-friendly toggle switches and sliders.

## Key Challenge
**UI Freezing During Conversion:** Running intense video conversion processes caused the application's GUI thread to lock up, making it appear unresponsive to the user.
*Solution:* Architected the conversion pipeline using asynchronous C# `Task`s and `Process` wrappers. Safely dispatched standard-output progress updates back to the main UI thread using `Dispatcher.Invoke`, maintaining a fluid user experience while heavy processing ran in the background.
