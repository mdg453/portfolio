"""Visual report generator for ex1 scene-cut detection

Creates a multi-panel PNG showing:
- histogram-L1 diff curve with selected cut marker
- averaged color frames before/after the cut
- grayscale averaged frames before/after
- histograms and CDFs for before/after

Usage:
    python report.py <video_path> [--out-dir demo_output]

Saves a single PNG named report_<video_basename>.png in the out directory.
"""

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:
    raise ImportError("matplotlib is required to generate reports. Install with: pip install matplotlib")

try:
    import imageio
except Exception:
    raise ImportError("imageio is required. Install with: pip install imageio")

import ex1


def compute_diffs_and_hists(video_path: str):
    reader = imageio.get_reader(video_path)
    color_frames = []
    gray_hists = []
    for frame in reader:
        color_frames.append(np.asarray(frame))
        gray = ex1._to_gray(frame)
        gray_hists.append(ex1._hist_normalized(gray))
    reader.close()
    diffs = np.array([float(np.sum(np.abs(gray_hists[i + 1] - gray_hists[i]))) for i in range(len(gray_hists) - 1)])
    return color_frames, gray_hists, diffs


def mean_image(frames):
    arr = np.stack(frames).astype(np.float32)
    m = arr.mean(axis=0)
    return np.clip(m, 0, 255).astype(np.uint8)


def plot_report(video_path: str, out_dir: Optional[str] = "demo_output"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    basename = Path(video_path).stem
    out_png = Path(out_dir) / f"report_{basename}.png"

    print(f"Reading video and computing histograms: {video_path}")
    color_frames, gray_hists, diffs = compute_diffs_and_hists(video_path)

    print("Running detector to get cut index...")
    try:
        cut_pair = ex1.main(video_path)
    except Exception as e:
        print("Detector failed:", e)
        # fallback to argmax
        cut_idx = int(np.argmax(diffs))
        cut_pair = (cut_idx, cut_idx + 1)

    cut_idx = cut_pair[0]

    # windows for before/after averaging
    W = 5
    b0 = max(0, cut_idx - W + 1)
    b1 = cut_idx + 1
    a0 = cut_idx + 1
    a1 = min(len(color_frames), cut_idx + 1 + W)

    print(f"Selected cut: {cut_pair}, averaging windows before [{b0},{b1}) after [{a0},{a1})")

    # averaged color frames
    color_before = mean_image(color_frames[b0:b1])
    color_after = mean_image(color_frames[a0:a1])

    # averaged grayscale frames
    gray_before = ex1._to_gray(color_before)
    gray_after = ex1._to_gray(color_after)

    # averaged histograms (use precomputed gray_hists)
    hist_before = np.mean(np.stack(gray_hists[b0:b1]), axis=0)
    hist_after = np.mean(np.stack(gray_hists[a0:a1]), axis=0)

    # plot
    fig = plt.figure(figsize=(14, 8))

    gs = fig.add_gridspec(3, 4, wspace=0.6, hspace=0.6)

    ax_diff = fig.add_subplot(gs[0, :2])
    ax_diff.plot(diffs, color='gray')
    ax_diff.axvline(cut_idx, color='red', linestyle='--', label=f'cut {cut_pair}')
    ax_diff.set_title('Histogram L1 diffs per frame')
    ax_diff.set_xlabel('frame index')
    ax_diff.set_ylabel('L1 diff')
    ax_diff.legend()

    ax_color_before = fig.add_subplot(gs[0, 2])
    ax_color_before.imshow(color_before[:, :, :3])
    ax_color_before.set_title(f'Color before (mean {b0}-{b1-1})')
    ax_color_before.axis('off')

    ax_color_after = fig.add_subplot(gs[0, 3])
    ax_color_after.imshow(color_after[:, :, :3])
    ax_color_after.set_title(f'Color after (mean {a0}-{a1-1})')
    ax_color_after.axis('off')

    ax_gray_before = fig.add_subplot(gs[1, 0])
    ax_gray_before.imshow(gray_before, cmap='gray')
    ax_gray_before.set_title('Grayscale before (averaged)')
    ax_gray_before.axis('off')

    ax_gray_after = fig.add_subplot(gs[1, 1])
    ax_gray_after.imshow(gray_after, cmap='gray')
    ax_gray_after.set_title('Grayscale after (averaged)')
    ax_gray_after.axis('off')

    ax_hist = fig.add_subplot(gs[1, 2:4])
    ax_hist.plot(hist_before, color='blue', label='hist before')
    ax_hist.plot(hist_after, color='orange', label='hist after')
    ax_hist.set_title('Averaged histograms')
    ax_hist.set_xlabel('Intensity')
    ax_hist.set_ylabel('Probability')
    ax_hist.legend()

    ax_cdf = fig.add_subplot(gs[2, :2])
    cdf_before = np.cumsum(hist_before)
    cdf_after = np.cumsum(hist_after)
    ax_cdf.plot(cdf_before, color='blue', label='cdf before')
    ax_cdf.plot(cdf_after, color='orange', label='cdf after')
    ax_cdf.set_title('CDFs')
    ax_cdf.set_xlabel('Intensity')
    ax_cdf.set_ylabel('Cumulative probability')
    ax_cdf.legend()

    # zoomed diffs around cut
    ax_zoom = fig.add_subplot(gs[2, 2:4])
    L = 40
    lo = max(0, cut_idx - L)
    hi = min(len(diffs), cut_idx + L)
    ax_zoom.plot(range(lo, hi), diffs[lo:hi], color='gray')
    ax_zoom.axvline(cut_idx, color='red', linestyle='--')
    ax_zoom.set_title('Zoomed diffs around detected cut')
    ax_zoom.set_xlabel('frame index')

    plt.suptitle(f'Report: {basename}  (detected cut {cut_pair})')
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    print(f'Report saved to: {out_png}')
    return str(out_png)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python report.py <video_path> [--out-dir demo_output]')
        sys.exit(1)
    video = sys.argv[1]
    out = 'demo_output'
    if len(sys.argv) >= 3:
        out = sys.argv[2]
    plot_report(video, out)
