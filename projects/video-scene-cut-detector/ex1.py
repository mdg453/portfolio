import os
from typing import Tuple, Optional

import numpy as np

try:
    import imageio
except Exception as e:
    raise ImportError("imageio is required: pip install imageio") from e
WIN_SIZE_AVG = 5

def _to_gray(frame: np.ndarray) -> np.ndarray:
    # Convert RGB(A) or grayscale frame to 2D uint8 grayscale.
    a = np.asarray(frame)
    if a.ndim == 3 and a.shape[2] >= 3:
        # use standard luminance weights on first 3 channels
        r, g, b = a[..., 0].astype(np.float32), a[..., 1].astype(np.float32), a[..., 2].astype(np.float32)
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return np.clip(gray, 0, 255).astype(np.uint8)
    # already single channel
    return np.clip(a.astype(np.float32), 0, 255).astype(np.uint8)


def _hist_normalized(gray: np.ndarray, bins: int = 256) -> np.ndarray:
   # Return normalized histogram (sum to 1) for uint8 grayscale image.
    histogram, _ = np.histogram(gray.ravel(), bins=bins, range=(0, 256))
    histogram = histogram.astype(np.float32)
    s = histogram.sum()
    return histogram / s if s > 0 else histogram


def main(video_path: str, video_type: Optional[int] = None) -> Tuple[int, int]:
    # Detect a single hard cut using averaged histograms around each candidate cut.
    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)

    reader = imageio.get_reader(video_path)
    histograms = []
    try:
        for frame in reader:
            gray = _to_gray(frame)
            histograms.append(_hist_normalized(gray))
    finally:
        try:
            reader.close()
        except Exception:
            pass

    n = len(histograms)
    if n < 2:
        raise ValueError("need at least 2 frames")

    # compute averaged-hist L1 score for each possible cut i (0..n-2)
    scores = np.zeros(n - 1, dtype=np.float32)
    for i in range(n - 1):
        b0 = max(0, i - WIN_SIZE_AVG + 1)
        b1 = i + 1
        a0 = i + 1
        a1 = min(n, i + 1 + WIN_SIZE_AVG)
        if b1 - b0 <= 0 or a1 - a0 <= 0:
            scores[i] = 0.0
            continue
        hist_before = np.mean(histograms[b0:b1], axis=0)
        hist_after = np.mean(histograms[a0:a1], axis=0)
        # compute CDFs (cumulative histograms) and use L1 between CDFs (1D EMD)
        cdf_before = np.cumsum(hist_before)
        cdf_after = np.cumsum(hist_after)
        scores[i] = float(np.sum(np.abs(cdf_after - cdf_before)))

    # if all scores are zero (very small video or uniform), fallback to simple consecutive diffs
    if np.allclose(scores, 0.0):
        diffs = np.array([float(np.sum(np.abs(histograms[i + 1] - histograms[i]))) for i in range(n - 1)], dtype=np.float32)
        cut = int(np.argmax(diffs))
        return (cut, cut + 1)

    cut = int(np.argmax(scores))
    return (cut, cut + 1)


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python ex1.py <video_path>")
        sys.exit(1)
    video = sys.argv[1]
    print(main(video))
