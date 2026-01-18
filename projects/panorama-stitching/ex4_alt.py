import os
import cv2
import argparse
import numpy as np
from PIL import Image
from utils import video_to_frames


def get_translation(img1, img2):
    """
    Estimate global translation between img1 and img2 using phase correlation.

    Returns:
        dx, dy: sub-pixel shift (float, float) such that img2 is shifted by (dx, dy)
               relative to img1 (OpenCV returns (x_shift, y_shift)).
    Notes:
        phaseCorrelate also returns a response score; higher generally indicates a
        more reliable estimate.
    """
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float32)

    (dx, dy), response = cv2.phaseCorrelate(gray1, gray2)
    return dx, dy, response  # return response so caller can gate on reliability


def smooth_path(path, kernel_size=31):
    """Smooth a 1D signal with a moving average (kernel size forced to odd)."""
    if kernel_size % 2 == 0:
        kernel_size += 1

    pad = kernel_size // 2
    padded = np.pad(path, (pad, pad), mode="edge")
    return np.convolve(padded, np.ones(kernel_size) / kernel_size, mode="valid")


def median_filter(data, kernel_size=5):
    """Rolling median filter (kernel size forced to odd)."""
    if kernel_size % 2 == 0:
        kernel_size += 1

    pad = kernel_size // 2
    padded = np.pad(data, (pad, pad), mode="edge")

    out = np.empty(len(data), dtype=np.float64)
    for i in range(len(data)):
        out[i] = np.median(padded[i:i + kernel_size])
    return out


def generate_panorama(input_frames_path, n_out_frames):
    """
    Create multiple slit-scan panoramas (parallax views) from a sequence of frames.

    High-level steps:
      1) Load frames
      2) Choose keyframes based on translation magnitude
      3) Detect dominant motion direction (horizontal vs vertical)
      4) Stabilize the cross-axis (remove jitter)
      5) For each requested view, build a mosaic by stacking blended slits
    """
    files = sorted(
        f for f in os.listdir(input_frames_path)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    )

    frames = []
    print("Loading frames...")
    for f in files:
        img = cv2.imread(os.path.join(input_frames_path, f))
        if img is not None:
            frames.append(img)

    if not frames:
        return []

    # --- PASS 1: Keyframe selection ---
    print("Identifying keyframes...")
    key_frames = [frames[0]]
    last_kept = frames[0]

    raw_dx = []
    raw_dy = []

    for i in range(1, len(frames)):
        dx, dy, resp = get_translation(last_kept, frames[i])

        # Optional: reject unreliable shifts (tune threshold for your data)
        # phaseCorrelate returns (shift, response). Response closer to 1 is often better.
        # if resp < 0.2:
        #     continue

        # Keep keyframe if motion is significant
        if abs(dx) > 1.0 or abs(dy) > 1.0:
            key_frames.append(frames[i])
            raw_dx.append(dx)
            raw_dy.append(dy)
            last_kept = frames[i]

    frames = key_frames
    n_frames = len(frames)
    h, w = frames[0].shape[:2]
    print(f"Working with {n_frames} unique keyframes.")

    if n_frames < 2:
        # Not enough motion / frames to build a meaningful panorama
        return []

    # raw_dx/raw_dy represent motion between frames[i] -> frames[i+1]
    # Ensure they have length (n_frames - 1).
    if len(raw_dx) != n_frames - 1:
        # If the selection logic changes, this guards against indexing errors.
        raw_dx = raw_dx[: n_frames - 1]
        raw_dy = raw_dy[: n_frames - 1]

    # --- Direction detection ---
    total_abs_dx = float(np.sum(np.abs(raw_dx)))
    total_abs_dy = float(np.sum(np.abs(raw_dy)))

    if total_abs_dy > total_abs_dx:
        direction = "VERTICAL"
        main_motion = np.array(raw_dy, dtype=np.float64)
        cross_motion = np.array(raw_dx, dtype=np.float64)
        scan_axis_size = h
    else:
        direction = "HORIZONTAL"
        main_motion = np.array(raw_dx, dtype=np.float64)
        cross_motion = np.array(raw_dy, dtype=np.float64)
        scan_axis_size = w

    print(f"Detected {direction} motion (|dx|={total_abs_dx:.1f}, |dy|={total_abs_dy:.1f}).")

    # --- Cross-axis stabilization ---
    # We build a per-frame cross-axis shift with length n_frames:
    # cross_step[i] is the cross motion between frame i -> i+1 (length n_frames-1)
    cross_step = median_filter(cross_motion, kernel_size=3)
    cross_step = np.clip(cross_step, -3.0, 3.0)

    # cumulative_cross[i] = sum of cross_step up to i-1
    cumulative_cross = np.zeros(n_frames, dtype=np.float64)
    cumulative_cross[1:] = np.cumsum(cross_step)

    target_cross = smooth_path(cumulative_cross, kernel_size=min(101, 2 * n_frames - 1))

    print(f"Pre-stabilizing keyframes (removing jitter on {'X' if direction == 'VERTICAL' else 'Y'} axis)...")
    stab_frames = []
    for i, frame in enumerate(frames):
        shift = target_cross[i] - cumulative_cross[i]

        if direction == "HORIZONTAL":
            # Stabilize Y (vertical jitter): translate by (0, shift)
            M = np.float32([[1, 0, 0], [0, 1, shift]])
        else:
            # Stabilize X (horizontal jitter): translate by (shift, 0)
            M = np.float32([[1, 0, shift], [0, 1, 0]])

        stab = cv2.warpAffine(
            frame, M, (w, h),
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
        )
        stab_frames.append(stab)

    # --- Slit-scan panorama generation ---
    print(f"Generating dense {direction.lower()} panoramas...")

    margin = 50
    parallax_range = scan_axis_size - 2 * margin
    if parallax_range <= 0:
        raise ValueError("Margin too large for frame size; reduce margin.")

    slit_offsets = np.linspace(0, parallax_range, n_out_frames).astype(int)

    final_panoramas = []
    total_length = int(np.sum(np.maximum(1, np.abs(main_motion).astype(int))))
    print(f"Target Panorama Length: ~{total_length} pixels")

    for v in range(n_out_frames):
        base_slit = margin + slit_offsets[v]
        strips = []

        for i in range(n_frames - 1):
            d_main = float(main_motion[i])
            dist = max(1, int(abs(d_main)))

            img1 = stab_frames[i]
            img2 = stab_frames[i + 1]

            for k in range(dist):
                alpha = k / dist

                offset = int(k * (d_main / dist))
                pos1 = int(np.clip(base_slit - offset, 0, scan_axis_size - 1))
                pos2 = int(np.clip(base_slit - offset + int(d_main), 0, scan_axis_size - 1))

                if direction == "HORIZONTAL":
                    c1 = img1[:, pos1]
                    c2 = img2[:, pos2]
                    blended = cv2.addWeighted(c1, 1 - alpha, c2, alpha, 0)
                    strips.append(blended.reshape(h, 1, 3))
                else:
                    r1 = img1[pos1, :]
                    r2 = img2[pos2, :]
                    blended = cv2.addWeighted(r1, 1 - alpha, r2, alpha, 0)
                    strips.append(blended.reshape(1, w, 3))

        if direction == "HORIZONTAL":
            full_pano = np.hstack(strips)
        else:
            # If you need consistent top/bottom ordering, this heuristic may help,
            # but it is dataset-dependent.
            if np.sum(main_motion) > 0:
                strips.reverse()
            full_pano = np.vstack(strips)

        pano_rgb = cv2.cvtColor(full_pano, cv2.COLOR_BGR2RGB)
        final_panoramas.append(Image.fromarray(pano_rgb))

    print(f"Generated {len(final_panoramas)} panoramas.")
    return final_panoramas


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", type=str)
    parser.add_argument("--num_panoramas", type=int, default=120)
    args = parser.parse_args()

    input_video = os.path.abspath(args.video_path)
    video_name = os.path.splitext(os.path.basename(input_video))[0]
    frames_dir = os.path.join("input_frames", video_name)

    if not os.path.exists(frames_dir) or not os.listdir(frames_dir):
        video_to_frames(input_video, frames_dir)

    panos = generate_panorama(frames_dir, args.num_panoramas)

    if panos:
        output_subfolder = os.path.join("output_panoramas", video_name)
        os.makedirs(output_subfolder, exist_ok=True)

        print(f"Saving individual frames to {output_subfolder}...")
        for idx, pano in enumerate(panos):
            pano.save(os.path.join(output_subfolder, f"pano_{idx:03d}.jpg"))

        out_path = f"{video_name}_parallax.mp4"
        print(f"Saving video to {out_path}...")

        w, h = panos[0].size
        if w % 2 != 0: w -= 1
        if h % 2 != 0: h -= 1

        # IMPORTANT: VideoWriter requires all frames to match (w, h). [web:10]
        video = cv2.VideoWriter(
            out_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            24,
            (w, h),
        )

        sequence = panos + panos[::-1]
        for p in sequence:
            p_bgr = cv2.cvtColor(np.array(p), cv2.COLOR_RGB2BGR)
            p_resized = cv2.resize(p_bgr, (w, h))
            video.write(p_resized)

        video.release()
        print("Done.")
