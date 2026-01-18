# Demo script for visualizing the per-frame processing steps (original -> gray -> flattened -> histograms)

import os
import sys
from typing import Optional

import numpy as np

try:
    import imageio
except Exception:
    raise ImportError("imageio is required to run this demo. Install with: pip install imageio")

# Try to import matplotlib for plotting the histogram; it's optional but recommended.
try:
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except Exception:
    _HAS_MATPLOTLIB = False

# Import conversion helper from ex1 (same directory)
import ex1


def _print_matrix_snippet(name: str, arr: np.ndarray, rows: int = 5, cols: int = 5):
    print(f"{name} shape={arr.shape} dtype={arr.dtype}")
    if arr.ndim == 3:
        snippet = arr[:rows, :cols, :]
    else:
        snippet = arr[:rows, :cols]
    print(f"{name} top-left {rows}x{cols} snippet:\n{snippet}\n")


def demo_first_frame(video_path: str, out_dir: Optional[str] = "demo_output"):
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    reader = imageio.get_reader(video_path)
    try:
        # read the first frame
        try:
            frame = reader.get_data(0)
        except Exception:
            # fallback to iterator
            frame = next(iter(reader))

        # 1) Original frame
        print("\n=== STEP 1: ORIGINAL FRAME ===")
        _print_matrix_snippet("Original frame", frame)
        orig_path = os.path.join(out_dir, "original_frame.png")
        try:
            imageio.imwrite(orig_path, frame)
            print(f"Saved original frame image to: {orig_path}")
        except Exception as e:
            print(f"Could not save original frame image: {e}")
        input("Press Enter to continue to grayscale conversion...")

        # 2) Convert to grayscale using ex1 helper
        print("\n=== STEP 2: GRAYSCALE CONVERSION ===")
        gray = ex1._to_gray(frame)
        _print_matrix_snippet("Grayscale (uint8)", gray)
        gray_path = os.path.join(out_dir, "gray_frame.png")
        try:
            imageio.imwrite(gray_path, gray)
            print(f"Saved grayscale image to: {gray_path}")
        except Exception as e:
            print(f"Could not save grayscale image: {e}")
        input("Press Enter to continue to flattened view...")

        # 3) Flattened view (1D array)
        print("\n=== STEP 3: FLATTENED ARRAY ===")
        flat = gray.ravel()
        print(f"Flattened length: {flat.size}")
        # show the first 100 values (or fewer)
        to_show = min(100, flat.size)
        print(f"First {to_show} flattened values:\n{flat[:to_show]}\n")
        input("Press Enter to continue to histogram (unnormalized)...")

        # 4) Histogram (unnormalized counts)
        print("\n=== STEP 4: HISTOGRAM (UNNORMALIZED) ===")
        counts, edges = np.histogram(flat, bins=256, range=(0, 256))
        print(f"Histogram counts (first 30 bins):\n{counts[:30]}")
        print(f"Sum of counts (should equal number of pixels): {counts.sum()}\n")
        # save counts to a text file
        counts_path = os.path.join(out_dir, "hist_counts.npy")
        np.save(counts_path, counts)
        print(f"Saved raw histogram counts to: {counts_path}")
        input("Press Enter to continue to normalized histogram...")

        # 5) Normalized histogram (probabilities)
        print("\n=== STEP 5: HISTOGRAM (NORMALIZED) ===")
        counts_f = counts.astype(np.float32)
        total = counts_f.sum()
        if total <= 0:
            print("Warning: no pixel counts found (empty frame?)")
            normalized = counts_f
        else:
            normalized = counts_f / total
        print(f"Normalized histogram (first 30 bins):\n{normalized[:30]}")
        print(f"Sum normalized (should be 1.0): {normalized.sum()}\n")

        # save normalized
        norm_path = os.path.join(out_dir, "hist_normalized.npy")
        np.save(norm_path, normalized)
        print(f"Saved normalized histogram to: {norm_path}")

        # 6) Optional: plot histogram if matplotlib is available
        if _HAS_MATPLOTLIB:
            fig_path = os.path.join(out_dir, "histogram_plot.png")
            plt.figure(figsize=(8, 3))
            plt.bar(np.arange(256), normalized, width=1.0, color="gray")
            plt.title("Normalized intensity histogram (first frame)")
            plt.xlabel("Intensity")
            plt.ylabel("Probability")
            plt.xlim(0, 255)
            plt.tight_layout()
            plt.savefig(fig_path)
            plt.close()
            print(f"Saved histogram plot to: {fig_path}")
        else:
            print("matplotlib not installed; skipping histogram plot. Install with: pip install matplotlib")

        print("\nDemo complete. Files are in:", os.path.abspath(out_dir))

    finally:
        try:
            reader.close()
        except Exception:
            pass


if __name__ == "__main__":
    # Usage: python main.py <video_path> [--prompt]
    if len(sys.argv) >= 2:
        video = sys.argv[1]
    else:
        video = "Exercise Inputs-20251105/video1_category1.mp4"

    prompt_mode = ('--prompt' in sys.argv)

    # First, detect the cut using ex1.main
    print(f"Detecting scene cut in: {video}")
    try:
        cut_pair = ex1.main(video)
    except Exception as e:
        print(f"Error running ex1.main: {e}")
        # fall back to demo on first frame
        demo_first_frame(video)
        sys.exit(1)

    print(f"Detected cut between frames: {cut_pair}")
    i = cut_pair[0]

    # Frames to process: one before the cut, the two cut frames, and one after
    indices = [i - 1, i, i + 1, i + 2]
    # make unique and only non-negative
    indices = sorted({idx for idx in indices if idx >= 0})

    reader = imageio.get_reader(video)
    try:
        for idx in indices:
            try:
                frame = reader.get_data(idx)
            except Exception as e:
                print(f"Could not read frame {idx}: {e}")
                continue

            # prepare a per-frame output folder
            out_dir = os.path.join("demo_output", f"frame_{idx}")
            os.makedirs(out_dir, exist_ok=True)

            print(f"\n--- Processing frame {idx} ---")
            _print_matrix_snippet(f"Original frame {idx}", frame)
            orig_path = os.path.join(out_dir, f"original_frame_{idx}.png")
            try:
                imageio.imwrite(orig_path, frame)
                print(f"Saved original frame image to: {orig_path}")
            except Exception as e:
                print(f"Could not save original frame image: {e}")

            # grayscale conversion
            gray = ex1._to_gray(frame)
            _print_matrix_snippet(f"Grayscale frame {idx}", gray)
            gray_path = os.path.join(out_dir, f"gray_frame_{idx}.png")
            try:
                imageio.imwrite(gray_path, gray)
                print(f"Saved grayscale image to: {gray_path}")
            except Exception as e:
                print(f"Could not save grayscale image: {e}")

            # flattened
            flat = gray.ravel()
            print(f"Flattened length: {flat.size}")
            to_show = min(100, flat.size)
            print(f"First {to_show} flattened values:\n{flat[:to_show]}\n")
            np.save(os.path.join(out_dir, f"flat_{idx}.npy"), flat)

            # histogram counts (unnormalized)
            counts, edges = np.histogram(flat, bins=256, range=(0, 256))
            print(f"Histogram counts (first 30 bins):\n{counts[:30]}")
            print(f"Sum of counts (should equal number of pixels): {counts.sum()}\n")
            np.save(os.path.join(out_dir, f"hist_counts_{idx}.npy"), counts)

            # normalized histogram
            counts_f = counts.astype(np.float32)
            total = counts_f.sum()
            if total <= 0:
                normalized = counts_f
            else:
                normalized = counts_f / total
            print(f"Normalized histogram (first 30 bins):\n{normalized[:30]}")
            print(f"Sum normalized (should be 1.0): {normalized.sum()}\n")
            np.save(os.path.join(out_dir, f"hist_normalized_{idx}.npy"), normalized)

            # optional plot
            if _HAS_MATPLOTLIB:
                fig_path = os.path.join(out_dir, f"histogram_plot_{idx}.png")
                plt.figure(figsize=(8, 3))
                plt.bar(np.arange(256), normalized, width=1.0, color="gray")
                plt.title(f"Normalized intensity histogram (frame {idx})")
                plt.xlabel("Intensity")
                plt.ylabel("Probability")
                plt.xlim(0, 255)
                plt.tight_layout()
                plt.savefig(fig_path)
                plt.close()
                print(f"Saved histogram plot to: {fig_path}")

            if prompt_mode:
                input("Press Enter to continue to next frame...")

    finally:
        try:
            reader.close()
        except Exception:
            pass

    print("\nAll done. Check demo_output for per-frame folders and outputs.")
