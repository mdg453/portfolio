import os
import argparse
import numpy as np
from PIL import Image

# --- Backend Detection ---
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    cv2 = None
    HAS_CV2 = False

try:
    import imageio.v2 as iio
    from scipy.ndimage import shift as scipy_shift
    from skimage.registration import phase_cross_correlation
    from skimage.color import rgb2gray
    HAS_SCIPY_SKIMAGE = True
except ImportError:
    iio = None
    scipy_shift = None
    phase_cross_correlation = None
    rgb2gray = None
    HAS_SCIPY_SKIMAGE = False

# Helper Functions to Abstract Backend
def read_image(path):
    """
    Reads an image and returns it in RGB format.
    Prioritizes io (server) if available, else CV2 (local).
    """
    if HAS_SCIPY_SKIMAGE and iio is not None:
        try:
            return iio.imread(path)
        except Exception:
            pass # Fallback
            
    if HAS_CV2 and cv2 is not None:
        img = cv2.imread(path)
        if img is None: return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
    return None

def to_gray(img_rgb):
    """Returns grayscale float32 image."""
    if HAS_SCIPY_SKIMAGE and rgb2gray is not None:
        return rgb2gray(img_rgb)
    elif HAS_CV2 and cv2 is not None:
        return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    # Simple fallback
    return np.dot(img_rgb[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.float32)

def compute_translation(img1, img2):
    """
    Returns (dx, dy) where `img2` is shifted by (dx, dy) relative to `img1`.
    Subpixel accuracy.
    """
    gray1 = to_gray(img1)
    gray2 = to_gray(img2)
    
    # 1. Try Skimage (Server/Robust)
    if HAS_SCIPY_SKIMAGE and phase_cross_correlation is not None:
        # skimage returns shift to align moving(gray2) to ref(gray1).
        # if gray2 is shifted (10, 0) relative to gray1, 
        # phase_cross_correlation returns (-10, 0).
        # We want (dx, dy) = (10, 0).
        shift, error, diffphase = phase_cross_correlation(gray1, gray2, upsample_factor=10)
        # Shift is (dy, dx)
        return -shift[1], -shift[0]
        
    # 2. Try OpenCV
    if HAS_CV2 and cv2 is not None:
        # cv2.phaseCorrelate(src1, src2) -> shift of src2 relative to src1
        # returns (dx, dy)
        (dx, dy), response = cv2.phaseCorrelate(gray1.astype(np.float32), gray2.astype(np.float32))
        return dx, dy
        
    return 0.0, 0.0

def apply_shift(img, shift_tuple):
    """
    Applies (row_shift, col_shift) to the image.
    shift_tuple = (shift_y, shift_x) or (shift_row, shift_col, shift_ch)
    """
    # shift_tuple is expected to be (shift_y, shift_x) for 2D, or full tuple
    if HAS_SCIPY_SKIMAGE and scipy_shift is not None:
        # Scipy handles n-dim
        # If img is (H,W,3) we need (shift_y, shift_x, 0)
        real_shift = list(shift_tuple)
        while len(real_shift) < img.ndim:
            real_shift.append(0)
        return scipy_shift(img, tuple(real_shift), mode='constant', cval=0)
        
    if HAS_CV2 and cv2 is not None:
        # CV2 warpAffine
        # shift_tuple[0] is Y shift (rows), shift_tuple[1] is X shift (cols)
        # Translation matrix M = [[1, 0, tx], [0, 1, ty]]
        ty = shift_tuple[0]
        tx = shift_tuple[1]
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        h, w = img.shape[:2]
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
        
    return img

def video_to_frames(video_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if HAS_CV2:
        cap = cv2.VideoCapture(video_path)
        count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_path = os.path.join(output_dir, f"frame_{count:05d}.jpg")
            cv2.imwrite(frame_path, frame)
            count += 1
        cap.release()
        print(f"Extracted {count} frames from {video_path} to {output_dir}")
    else:
        print("Warning: OpenCV not available. Cannot extract frames.")

def get_translation(img1, img2):
    # Wrapper for compatibility with old calls if any
    return compute_translation(img1, img2)

def smooth_path(path, kernel_size=31):
    """Smooths the path using a moving average."""
    # Ensure kernel is odd
    if kernel_size % 2 == 0: kernel_size += 1
    # Pad to handle edges
    # Use 'edge' mode padding which is simple in numpy
    pad_size = kernel_size // 2
    path_padded = np.pad(path, (pad_size, pad_size), mode='edge')
    # Convolve
    path_smoothed = np.convolve(path_padded, np.ones(kernel_size)/kernel_size, mode='valid')
    return path_smoothed

def median_filter(data, kernel_size=5):
    """Simple rolling median filter to remove outliers."""
    # Ensure odd kernel
    if kernel_size % 2 == 0: kernel_size += 1
    pad_size = kernel_size // 2
    padded = np.pad(data, (pad_size, pad_size), mode='edge')
    result = []
    for i in range(len(data)):
        window = padded[i:i+kernel_size]
        result.append(np.median(window))
    return np.array(result)

def fill_zeros(data, threshold=0.1):
    """Interpolate zero (or near-zero) values using linear interpolation."""
    data = np.array(data)
    # Find indices of zeros/low values and non-zeros
    zeros = (np.abs(data) < threshold)
    non_zeros = ~zeros
    
    # If no non-zeros or no zeros, return as is
    if not np.any(non_zeros) or not np.any(zeros):
        return data
        
    # Interpolate
    x_indices = np.arange(len(data))
    data[zeros] = np.interp(x_indices[zeros], x_indices[non_zeros], data[non_zeros])
    return data

def generate_panorama(input_frames_path, n_out_frames):
    """
    Stereo Mosaicing: Creates n panoramas from different virtual viewpoints.
    """
    files = sorted([f for f in os.listdir(input_frames_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    frames = []
    
    print("Loading frames...")
    for f in files:
        # Use Abstract Helper
        img = read_image(os.path.join(input_frames_path, f))
        if img is not None:
            frames.append(img)
            
    if not frames: return []
    
    # --- PASS 1: Identify Keyframes & Direction ---
    print("Identifying keyframes...")
    key_frames = [frames[0]]
    last_kept = frames[0]
    
    raw_dx = []
    raw_dy = []
    
    for i in range(1, len(frames)):
        # Calculate Shift using robust helper (handles CV2 vs Skimage backend)
        dx, dy = compute_translation(last_kept, frames[i])
        
        # Check significance
        if abs(dx) > 1.0 or abs(dy) > 1.0:
            key_frames.append(frames[i])
            raw_dx.append(dx)
            raw_dy.append(dy)
            last_kept = frames[i]
            
    # Dummy motion for last frame
    if raw_dx:
        raw_dx.append(raw_dx[-1])
        raw_dy.append(raw_dy[-1])
    else:
        raw_dx.append(1.0)
        raw_dy.append(0)
        
    frames = key_frames
    n_frames = len(frames)
    h, w = frames[0].shape[:2]
    print(f"Working with {n_frames} unique keyframes.")
    
    # --- Auto-detect Direction ---
    total_abs_dx = sum([abs(x) for x in raw_dx])
    total_abs_dy = sum([abs(y) for y in raw_dy])
    
    if total_abs_dy > total_abs_dx:
        direction = 'VERTICAL'
        print(f"Detected VERTICAL motion (dy={total_abs_dy:.1f} > dx={total_abs_dx:.1f}).")
        main_motion = raw_dy
        cross_motion = raw_dx # We stabilize this axis
        scan_axis_size = h
        cross_axis_size = w
    else:
        direction = 'HORIZONTAL'
        print(f"Detected HORIZONTAL motion (dx={total_abs_dx:.1f} > dy={total_abs_dy:.1f}).")
        main_motion = raw_dx
        cross_motion = raw_dy # We stabilize this axis
        scan_axis_size = w
        cross_axis_size = h
        
    # --- 2. Stabilization Calculation (On Cross Axis) ---
    # Horizontal Mode: Stabilize Y (remove dy jitter)
    # Vertical Mode: Stabilize X (remove dx jitter)
    
    # --- 2. Stabilization Calculation (On Cross Axis) ---
    # Horizontal Mode: Stabilize Y (remove dy jitter)
    # Vertical Mode: Stabilize X (remove dx jitter)
    
    # Note: cross_motion is just a list of floats.
    cross_filtered = median_filter(cross_motion, kernel_size=3)
    cross_filtered = np.clip(cross_filtered, -3.0, 3.0)
    
    cumulative_cross = np.array([0.0] + list(np.cumsum(cross_filtered[:-1])))
    target_cross = smooth_path(cumulative_cross, kernel_size=101)
    
    print(f"Pre-stabilizing keyframes (removing jitter on {'X' if direction == 'VERTICAL' else 'Y'} axis)...")
    stab_frames = []
    
    for i, frame in enumerate(frames):
        shift_val = target_cross[i] - cumulative_cross[i]
        
        # apply_shift wrapper expects (shift_y, shift_x)
        if direction == 'HORIZONTAL':
             # stabilizing Y -> shift_y
             # shift_tuple = (shift_val, 0)
             stab = apply_shift(frame, (shift_val, 0))
        else:
             # stabilizing X -> shift_x
             # shift_tuple = (0, shift_val)
             stab = apply_shift(frame, (0, shift_val))
             
        stab_frames.append(stab)
        
    # --- 3. Spatial Strip Generation ---
    print(f"Generating dense {direction.lower()} panoramas...")
    
    margin = 50
    # Parallax range is along the scan axis
    parallax_range = scan_axis_size - 2 * margin
    slit_offsets = np.linspace(0, parallax_range, n_out_frames).astype(int)
    
    final_panoramas = []
    
    # Total expected length along main axis
    total_length = int(sum([abs(d) for d in main_motion[:-1]]))
    print(f"Target Panorama Length: ~{total_length} pixels")

    for v in range(n_out_frames):
        base_slit = margin + slit_offsets[v]
        strips = []
        
        for i in range(n_frames - 1):
            d_main = main_motion[i]
            dist = max(1, int(abs(d_main)))
            
            img1 = stab_frames[i].astype(np.float32)
            img2 = stab_frames[i+1].astype(np.float32)
            
            for k in range(dist):
                alpha = k / dist
                
                # Scan Logic:
                offset = int(k * (d_main / dist)) 
                
                pos1 = base_slit - offset
                pos2 = base_slit - offset + int(d_main)
                
                pos1 = np.clip(pos1, 0, scan_axis_size - 1)
                pos2 = np.clip(pos2, 0, scan_axis_size - 1)
                
                if direction == 'HORIZONTAL':
                    # Extract Columns
                    c1 = img1[:, pos1]
                    c2 = img2[:, pos2]
                    # Numpy Blend
                    blended = c1 * (1 - alpha) + c2 * alpha
                    strips.append(blended.reshape(h, 1, 3))
                else:
                    # Extract Rows
                    r1 = img1[pos1, :]
                    r2 = img2[pos2, :]
                    # Numpy Blend
                    blended = r1 * (1 - alpha) + r2 * alpha
                    strips.append(blended.reshape(1, w, 3))
        
        if direction == 'HORIZONTAL':
            full_pano = np.hstack(strips)
        else:
            if sum(main_motion) > 0:
                strips.reverse()
            full_pano = np.vstack(strips)
            
        # Clip to valid range and cast
        pano_uint8 = np.clip(full_pano, 0, 255).astype(np.uint8)
        final_panoramas.append(Image.fromarray(pano_uint8))
        
    print(f"Generated {len(final_panoramas)} panoramas.")
    return final_panoramas


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", type=str)
    # Default increased to 120 for smoother output
    parser.add_argument("--num_panoramas", type=int, default=120) 
    args = parser.parse_args()
    
    input_video = os.path.abspath(args.video_path)
    video_name = os.path.splitext(os.path.basename(input_video))[0]
    frames_dir = os.path.join("input_frames", video_name)
    
    if not os.path.exists(frames_dir) or not os.listdir(frames_dir):
        video_to_frames(input_video, frames_dir)
        
    panos = generate_panorama(frames_dir, args.num_panoramas)
    
    if panos:
        # --- NEW: Save individual frames ---
        output_subfolder = os.path.join("output_panoramas", video_name)
        if not os.path.exists(output_subfolder):
            os.makedirs(output_subfolder)
            
        print(f"Saving individual frames to {output_subfolder}...")
        for idx, pano in enumerate(panos):
            save_path = os.path.join(output_subfolder, f"pano_{idx:03d}.jpg")
            pano.save(save_path)

        # Save Video
        out_path = f"{video_name}_parallax.mp4"
        print(f"Saving video to {out_path}...")
        
        # Dimensions
        w, h = panos[0].size
        # Make divisble by 2 for H.264
        if w % 2 != 0: w -= 1
        if h % 2 != 0: h -= 1
        
        # 24 FPS is cinematic/smoother than 30 for this effect
        video = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 24, (w, h))
        
        # Ping Pong loop (Forward + Backward)
        sequence = panos + panos[::-1]
        
        for p in sequence:
            p_np = np.array(p)
            p_bgr = cv2.cvtColor(p_np, cv2.COLOR_RGB2BGR)
            p_resized = cv2.resize(p_bgr, (w, h))
            video.write(p_resized)
            
        video.release()
        print("Done.")
