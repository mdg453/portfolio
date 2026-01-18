# Hebrew Media Transcription using FFmpeg and a Fine-Tuned Whisper Model
#
# This script first processes a media file (e.g., MP4 video or M4A audio)
# into a temporary MP3 file using FFmpeg. It then transcribes the MP3 using
# a specialized, high-accuracy Whisper model fine-tuned for Hebrew.
# It is designed to process multiple files in a batch.
#
# --- Prerequisites ---
#
# 1. Python 3:
#    Ensure you have Python 3 installed on your system.
#
# 2. FFmpeg and ffprobe:
#    The script requires these command-line tools.
#    - On macOS (using Homebrew): `brew install ffmpeg`
#    - On Windows (using Chocolatey): `choco install ffmpeg`
#    - On Debian/Ubuntu: `sudo apt update && sudo apt install ffmpeg`
#
# 3. Required Python Libraries:
#    Install 'transformers', 'accelerate', and 'tqdm'.
#    `pip install transformers accelerate tqdm`
#
# 4. PyTorch with CUDA (for GPU acceleration):
#    For fast GPU processing, you must install PyTorch with CUDA support.
#    Visit the PyTorch website (pytorch.org) to get the correct installation command.
#    Example: `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`
#

import os

import faster_whisper
import torch
from transformers import pipeline
import subprocess
import shutil
import sys
import signal
import re
# import faster-whisper
from pathlib import Path
from typing import List, Optional

try:
    from tqdm import tqdm
except ImportError:
    print("Missing dependency: tqdm. Run 'pip install tqdm' and try again.")
    sys.exit(1)


# --- FFmpeg Helper Functions ---

def _require_ffmpeg():
    """Checks if ffmpeg and ffprobe are available in the system's PATH."""
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            sys.exit(f"Error: '{tool}' not found in PATH. Please install FFmpeg and ensure it's in your system's PATH.")

def _get_media_duration(path: Path) -> float:
    """Gets the duration of a media file in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=nokey=1:noprint_wrappers=1",
        str(path),
    ]
    try:
        duration_str = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
        return float(duration_str)
    except (subprocess.CalledProcessError, ValueError) as e:
        print(f"Error probing media duration: {e}")
        return 0.0

def convert_to_temp_mp3(src_path: Path) -> Optional[Path]:
    """Converts audio from a media file to a temporary MP3 file."""
    _require_ffmpeg()

    dst_path = src_path.with_suffix(".temp.mp3")
    total_duration = _get_media_duration(src_path)

    if total_duration == 0.0:
        print("Could not determine media duration. Aborting conversion.")
        return None

    # This command works for both video (extracts audio) and audio files (converts).
    cmd = [
        "ffmpeg", "-y", "-i", str(src_path),
        "-vn",  # Ignore video stream if it exists
        "-c:a", "libmp3lame",
        "-b:a", "192k",
        "-progress", "pipe:2", "-nostats",
        str(dst_path)
    ]

    print("Converting to temporary MP3 file...")
    rgx = re.compile(r"out_time_us=(\d+)")
    try:
        with subprocess.Popen(cmd, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace', bufsize=1) as p:
            with tqdm(total=total_duration, unit="s", dynamic_ncols=True, desc="Converting to MP3") as bar:
                for line in p.stderr:
                    if m := rgx.search(line):
                        elapsed_time = min(float(m.group(1)) / 1e6, total_duration)
                        bar.n = elapsed_time
                        bar.refresh()
            p.wait()
        if p.returncode != 0:
            print(f"\nFFmpeg failed with exit code {p.returncode}. Could not convert audio.")
            return None
    except KeyboardInterrupt:
        p.send_signal(signal.SIGINT)
        p.wait()
        print("\nAudio conversion cancelled by user.")
        return None

    print(f"\n[OK] Temporary audio saved to → {dst_path}")
    return dst_path



def transcribe_hebrew_audio(file_path: str, model: faster_whisper.WhisperModel) -> str:
    """Transcribes a Hebrew audio file using a pre-loaded faster-whisper model."""
    try:
        print(f"Starting transcription for '{file_path}'...")
        segments, _ = model.transcribe(file_path, language='he')
        # Use a generator expression with tqdm for progress bar
        texts = [s.text for s in tqdm(segments, unit=" segment", desc="Transcribing")]
        transcribed_text = ' '.join(texts).strip()
        print("Transcription complete.")
        return transcribed_text
    except Exception as e:
        return f"An error occurred during transcription: {e}"

# --- User Interaction and Main Execution Block ---

def _ask_for_files_interactive() -> List[Path]:
    """Interactively prompts the user to select one or more media files."""
    ACCEPTED_SUFFIXES = ['.mp4', '.mkv', '.mov', '.avi', '.m4a']
    if os.name == "nt":
        try:
            import tkinter as tk
            from tkinter import filedialog as fd
            print("Opening file dialog...")
            tk.Tk().withdraw()
            files = fd.askopenfilenames(
                title="Select Media File(s)",
                filetypes=[
                    ("Media Files", "*.mp4 *.mkv *.mov *.avi *.m4a"),
                    ("Video Files", "*.mp4 *.mkv *.mov *.avi"),
                    ("Audio Files", "*.m4a")
                ]
            )
            if files:
                return [Path(f) for f in files]
        except Exception as e:
            print(f"Could not open GUI file dialog ({e}), falling back to console input.")

    while True:
        p_str = input("Enter a path to a single media file OR a directory containing media files: ").strip().strip('"\'')
        p = Path(p_str).expanduser()
        if p.is_file() and p.suffix.lower() in ACCEPTED_SUFFIXES:
            return [p]
        if p.is_dir():
            media_files = [f for f in p.iterdir() if f.is_file() and f.suffix.lower() in ACCEPTED_SUFFIXES]
            if not media_files:
                print(f"No supported media files found in '{p}'. Please try again.")
                continue
            print(f"Found {len(media_files)} media file(s) to process.")
            return media_files
        else:
            print("Path is not a valid file or directory – please try again.")


def get_files_from_cli_paths(paths: List[str]) -> List[Path]:
    """Processes a list of command-line paths into a list of media files."""
    media_files = []
    ACCEPTED_SUFFIXES = ['.mp4', '.mkv', '.mov', '.avi', '.m4a']
    for p_str in paths:
        p = Path(p_str.strip().strip('"\''))
        if p.is_dir():
            print(f"Scanning directory: {p}")
            found = [f for f in p.iterdir() if f.is_file() and f.suffix.lower() in ACCEPTED_SUFFIXES]
            media_files.extend(found)
        elif p.is_file():
            if p.suffix.lower() in ACCEPTED_SUFFIXES:
                media_files.append(p)
            else:
                print(f"Warning: Skipping unsupported file type: {p}")
        else:
            print(f"Warning: Path not found or is not a file/directory: {p}")
    return media_files


if __name__ == "__main__":
    # --- Step 1: Get the list of files to process BEFORE loading the model ---
    if len(sys.argv) > 1:
        print("Processing files from command-line arguments...")
        media_paths = get_files_from_cli_paths(sys.argv[1:])
    else:
        print("No command-line arguments provided, starting interactive mode...")
        media_paths = _ask_for_files_interactive()

    if not media_paths:
        print("No media files found to process. Exiting.")
        sys.exit(0)

    # --- Step 2: Load the model ONCE ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if torch.cuda.is_available() else "float32"

    if device == "cuda":
        print("NVIDIA GPU detected. Model will run on the GPU for faster performance.")
    else:
        print("No NVIDIA GPU detected. Model will run on the CPU (this may be slow).")

    model_id = 'ivrit-ai/whisper-large-v3-turbo-ct2'
    print(f"Loading specialized Hebrew model '{model_id}'... (This may take a moment)")

    try:
        model = faster_whisper.WhisperModel(model_id, device=device, compute_type=compute_type)
        print("Model loaded successfully.")
    except Exception as e:
        sys.exit(f"Failed to load the model. Error: {e}")

    # --- Step 3: Loop through each file and process it ---
    total_files = len(media_paths)
    for i, media_path in enumerate(media_paths):
        print(f"\n--- Processing file {i+1}/{total_files}: {media_path.name} ---")
        temp_audio_path = None
        try:
            # Convert media file to a temporary MP3
            temp_audio_path = convert_to_temp_mp3(media_path)

            if temp_audio_path:
                # Transcribe the temp MP3 using the pre-loaded model
                transcription = transcribe_hebrew_audio(str(temp_audio_path), model)

                # Print the result
                print("\n--- Transcription Result ---")
                print(transcription)
                print("--------------------------")

                # Save transcription to a file
                if "Error:" not in transcription and "An error occurred" not in transcription:
                    output_filename = media_path.with_suffix(".txt")
                    with open(output_filename, "w", encoding="utf-8") as f:
                        f.write(transcription)
                    print(f"\nTranscription saved to '{output_filename}'")

        finally:
            # Clean up the temporary audio file
            if temp_audio_path and temp_audio_path.exists():
                print(f"Cleaning up temporary file: {temp_audio_path}")
                os.remove(temp_audio_path)

    print("\nAll files processed.")
