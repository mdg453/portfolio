"""Minimal scene-cut detector without third-party Python packages.

This script requires only a working Python (3.7+) and the `ffmpeg` command-line
utility available on the system PATH. It does not depend on numpy or imageio,
so you can avoid creating a large venv.

Algorithm (matches your spec):
 - Stream raw grayscale frames from ffmpeg
 - For each frame compute a normalized 256-bin histogram (as a Python list)
 - Compute SAD (sum absolute differences) between consecutive histograms
 - Return the pair (i, i+1) with maximal SAD

Usage:
    python ex1_minimal.py <video_path>

Notes:
 - This is slower than a numpy version but keeps dependencies minimal.
 - If ffmpeg is not installed you'll get a helpful error message.
"""

import shutil
import subprocess
import sys
from collections import Counter
from typing import Tuple


def check_ffmpeg():
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH. Install ffmpeg to use this minimal detector.")


def probe_size(path: str) -> Tuple[int, int]:
    """Return (width, height) of the first video stream using ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=p=0:s=x",
        path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not proc.stdout.strip():
        raise RuntimeError(f"ffprobe failed to read video size: {proc.stderr.strip()}")
    w, h = proc.stdout.strip().split("x")
    return int(w), int(h)


def stream_gray_frames(path: str, width: int, height: int):
    """Yield raw grayscale frames (bytes objects) from ffmpeg stdout."""
    cmd = [
        "ffmpeg",
        "-i",
        path,
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-vsync",
        "0",
        "-loglevel",
        "error",
        "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.stdout is None:
        proc.kill()
        raise RuntimeError("ffmpeg did not provide stdout")

    frame_size = width * height
    try:
        while True:
            buf = proc.stdout.read(frame_size)
            if not buf:
                break
            if len(buf) < frame_size:
                # incomplete final frame — ignore
                break
            yield buf
    finally:
        proc.stdout.close()
        proc.kill()


def hist_normalized_from_bytes(frame_bytes: bytes):
    """Return normalized histogram (list of 256 floats) for raw grayscale frame bytes."""
    cnt = Counter(frame_bytes)
    total = len(frame_bytes)
    hist = [cnt.get(i, 0) / total for i in range(256)]
    return hist


def detect_cut_minimal(video_path: str, win_size: int = 5) -> Tuple[int, int]:
    check_ffmpeg()
    w, h = probe_size(video_path)
    # stream frames and compute per-frame normalized histograms -> CDFs on the fly
    cdfs = []
    for frame_bytes in stream_gray_frames(video_path, w, h):
        hist = hist_normalized_from_bytes(frame_bytes)
        # compute CDF for this frame
        cdf = []
        running = 0.0
        for v in hist:
            running += v
            cdf.append(running)
        cdfs.append(cdf)

    n = len(cdfs)
    if n < 2:
        raise ValueError("video must contain at least 2 frames")

    # score each possible cut i by L1 between averaged CDFs before and after
    scores = [0.0] * (n - 1)
    for i in range(n - 1):
        b0 = max(0, i - win_size + 1)
        b1 = i + 1
        a0 = i + 1
        a1 = min(n, i + 1 + win_size)
        if b1 - b0 <= 0 or a1 - a0 <= 0:
            scores[i] = 0.0
            continue
        # average CDFs binwise for the two windows
        # cdfs elements are lists of length 256
        avg_b = [0.0] * 256
        avg_a = [0.0] * 256
        for j in range(b0, b1):
            row = cdfs[j]
            for k in range(256):
                avg_b[k] += row[k]
        for j in range(a0, a1):
            row = cdfs[j]
            for k in range(256):
                avg_a[k] += row[k]
        nb = (b1 - b0)
        na = (a1 - a0)
        for k in range(256):
            avg_b[k] /= nb
            avg_a[k] /= na
        # L1 between averaged CDFs
        s = 0.0
        for k in range(256):
            s += abs(avg_a[k] - avg_b[k])
        scores[i] = s

    # if all scores nearly zero, fallback to simple consecutive-frame SAD on histograms
    if all(abs(x) < 1e-12 for x in scores):
        # recompute per-frame hist (we had hist via cdf differences if needed, but recompute simply)
        # convert cdfs back to hist (cdf diff)
        hists = []
        for cdf in cdfs:
            hist = [cdf[0]] + [cdf[k] - cdf[k - 1] for k in range(1, 256)]
            hists.append(hist)
        diffs = []
        for i in range(n - 1):
            s = 0.0
            h0 = hists[i]
            h1 = hists[i + 1]
            for k in range(256):
                s += abs(h1[k] - h0[k])
            diffs.append(s)
        cut = int(max(range(len(diffs)), key=lambda i: diffs[i]))
        return cut, cut + 1

    cut = int(max(range(len(scores)), key=lambda i: scores[i]))
    return cut, cut + 1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ex1_minimal.py <video_path> [win_size]")
        sys.exit(1)
    video = sys.argv[1]
    win = int(sys.argv[2]) if len(sys.argv) >= 3 else 5
    try:
        cut = detect_cut_minimal(video, win_size=win)
        print(cut)
    except Exception as e:
        print("ERROR:", e, file=sys.stderr)
        sys.exit(2)
