"""Generate visual reports for the top-2 histogram-diff candidates in a video.

Usage:
    python generate_top2_reports.py <video_path> [out_dir]

This will write two PNGs:
    report_<basename>_candidate_1_<idx>.png  (largest diff)
    report_<basename>_candidate_2_<idx>.png  (second largest diff)

The layout matches the report.py visual style.
"""
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:
    raise ImportError("matplotlib is required: pip install matplotlib")

import imageio
import ex1


def mean_image(frames):
    arr = np.stack(frames).astype(np.float32)
    m = arr.mean(axis=0)
    return np.clip(m, 0, 255).astype(np.uint8)


def compute_frames_hists(video_path: str):
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


def make_report_for_candidate(video_path: str, color_frames, gray_hists, diffs, cut_idx: int, rank: int, out_dir: str):
    Path = Path = __import__('pathlib').Path
    basename = Path(video_path).stem
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f"report_{basename}_candidate_{rank}_{cut_idx}.png"

    # windows
    W = 5
    b0 = max(0, cut_idx - W + 1)
    b1 = cut_idx + 1
    a0 = cut_idx + 1
    a1 = min(len(color_frames), cut_idx + 1 + W)

    color_before = mean_image(color_frames[b0:b1])
    color_after = mean_image(color_frames[a0:a1])
    gray_before = ex1._to_gray(color_before)
    gray_after = ex1._to_gray(color_after)
    hist_before = np.mean(np.stack(gray_hists[b0:b1]), axis=0)
    hist_after = np.mean(np.stack(gray_hists[a0:a1]), axis=0)

    # plotting
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(3, 4, wspace=0.6, hspace=0.6)

    ax_diff = fig.add_subplot(gs[0, :2])
    ax_diff.plot(diffs, color='gray')
    ax_diff.axvline(cut_idx, color='red', linestyle='--', label=f'candidate {rank} idx={cut_idx}')
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

    ax_zoom = fig.add_subplot(gs[2, 2:4])
    L = 40
    lo = max(0, cut_idx - L)
    hi = min(len(diffs), cut_idx + L)
    ax_zoom.plot(range(lo, hi), diffs[lo:hi], color='gray')
    ax_zoom.axvline(cut_idx, color='red', linestyle='--')
    ax_zoom.set_title('Zoomed diffs around candidate')
    ax_zoom.set_xlabel('frame index')

    plt.suptitle(f'Report: {basename}  (candidate {rank} idx={cut_idx})')
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return str(out_png)


def main(video_path: str, out_dir: str = 'demo_output'):
    print('Computing frames and histograms for', video_path)
    color_frames, gray_hists, diffs = compute_frames_hists(video_path)
    if len(diffs) == 0:
        raise ValueError('video has fewer than 2 frames')
    # get top-2 candidates
    inds = np.argsort(diffs)[::-1]
    top = inds[:2] if len(inds) >= 2 else np.append(inds, inds[0])
    print('Top candidate indices and diffs:')
    for rank, idx in enumerate(top, start=1):
        print(rank, idx, diffs[idx])

    outputs = []
    for rank, idx in enumerate(top, start=1):
        print(f'Generating report for candidate {rank} (index {idx})')
        out = make_report_for_candidate(video_path, color_frames, gray_hists, diffs, int(idx), rank, out_dir)
        print('Saved', out)
        outputs.append(out)
    return outputs


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python generate_top2_reports.py <video_path> [out_dir]')
        sys.exit(1)
    video = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) >= 3 else 'demo_output'
    main(video, out)

