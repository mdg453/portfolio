ex1_minimal — minimal scene-cut detector

Contents
- ex1_minimal.py   (pure-Python detector, requires system ffmpeg)
- tiny_venv/       (optional minimal venv created with `python3 -m venv --without-pip`)
- run.sh           (convenience script to run detector using venv or system python)

Purpose
This package is tailored to be extremely small so you can upload it to Moodle: the detector
is pure-Python and uses the system `ffmpeg` command to read frames. The included `tiny_venv`
was created using `python3 -m venv --without-pip` and is tiny (~24K). No large Python wheels
are included.

Usage
1. Ensure `ffmpeg` (and `ffprobe`) are installed on the target machine and available on PATH.
   On Debian/Ubuntu: `sudo apt-get install ffmpeg`.

2. Run using system python:
   python3 ex1_minimal.py "Exercise Inputs-20251105/video3_category2.mp4"

3. Or run via the provided tiny venv (if present):
   ./run.sh "Exercise Inputs-20251105/video3_category2.mp4"

Notes
- This package does NOT include system-level ffmpeg; ffmpeg must be installed separately.
- The detector uses averaged CDF scoring (window size default 5). You can pass a custom
  window size as the second argument: `python3 ex1_minimal.py <video> <win_size>`.

Files created by packaging script
- ex1_minimal_package.tar.gz — tarball containing the files above (should be under 20 MB)

Contact
If you need me to also include a tiny README in Hebrew or tweak the packaging, tell me and I'll update it.
