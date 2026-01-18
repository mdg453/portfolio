"""
Image Combination Tool
======================

This script implements advanced image processing techniques for blending and hybrid image generation.
It consolidates logic for Pyramid Blending, Fourier Hybrid Images, and an interactive Mask Creator.

Features
--------
1. Pyramid Blending: Seamlessly blend two images using a mask and Laplacian Pyramids.
2. Fourier Hybrid Images: Create optical illusions by combining low frequencies of one image with high frequencies of another.
3. Grayscale Support: Process images in black and white.
4. Interactive Mask Creator: Built-in tool to create custom masks using GrabCut.

Usage Examples
--------------

1. Pyramid Blending (Default):
   python ex3.py --im1 inputs/spring.jpg --im2 inputs/menifa.jpg --mask inputs/mask.png --mode pyramid --out outputs/pyramid_result.png

2. Fourier Hybrid Image:
   python ex3.py --im1 inputs/hana.jpeg --im2 inputs/tzipi.jpeg --mode fourier --cutoff 20 --out outputs/hybrid_result.png

3. Grayscale Mode (add --grayscale):
   python ex3.py --im1 inputs/hana.jpeg --im2 inputs/tzipi.jpeg --mode fourier --grayscale --out outputs/hybrid_gray.png

4. Mask Creation (Interactive):
   python ex3.py --im1 inputs/spring.jpg --mode mask --out inputs/mask_custom.png

Arguments:
  --im1         Path to the first image (Close view / High freq for Hybrid, or Foreground for Pyramid, or Input for Mask)
  --im2         Path to the second image (Far view / Low freq for Hybrid, or Background for Pyramid)
  --mask        Path to the binary mask (Required for Pyramid mode if not split)
  --out         Path to save the output image
  --mode        'pyramid', 'fourier', or 'mask' (default: 'pyramid')
  --cutoff      Cutoff frequency for Fourier mode (default: 20.0)
  --grayscale   Process images in grayscale
"""
import cv2
import numpy as np
import os
import argparse
from scipy.ndimage import convolve
from numpy.fft import fft2, ifft2, fftshift, ifftshift

def relpath(filename):
    return os.path.join(os.path.dirname(__file__), filename)

# --- Pyramid Blending Logic ---

def _get_filter_vec(filter_size):
    # Binomial coefficients approximation for Gaussian
    if filter_size == 1:
        return np.array([[1]])
        
    f = np.array([1, 1])
    for _ in range(filter_size - 2):
        f = np.convolve(f, [1, 1])
        
    f = f / np.sum(f)
    return f.reshape(1, -1)

def _blur_and_downsample(im, filter_vec):
    # Blur with filter_vec (1D convolution in x and y)
    # Convolve x
    blurred = convolve(im, filter_vec, mode='mirror')
    # Convolve y (transpose filter)
    blurred = convolve(blurred, filter_vec.T, mode='mirror')
    
    # Subsample (take every 2nd pixel)
    return blurred[::2, ::2]

def _expand(im, filter_vec):
    # Upsample (insert zeros)
    upsampled = np.zeros((im.shape[0] * 2, im.shape[1] * 2), dtype=im.dtype)
    upsampled[::2, ::2] = im
    
    # Blur (interpolation)
    # We need to multiply the filter by 2 to maintain brightness after zero insertion
    
    # Convolve x
    expanded = convolve(upsampled, 2 * filter_vec, mode='mirror')
    # Convolve y
    expanded = convolve(expanded, 2 * filter_vec.T, mode='mirror')
    
    return expanded

def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    Builds a Gaussian pyramid for a given image.
    :param im: a grayscale image with double values in [0, 1]
    :param max_levels: the maximal number of levels in the resulting pyramid.
    :param filter_size: the size of the Gaussian filter (an odd scalar that represents a squared filter)
                        to be used in constructing the pyramid filter
    :return: pyr, filter_vec. Where pyr is the resulting pyramid as a standard python list with maximum max_levels elements,
             where each element is a grayscale image.
             filter_vec is a row vector of shape (1, filter_size) used for the pyramid construction.
    """
    # Create the 1D Gaussian filter
    filter_vec = _get_filter_vec(filter_size)
    
    pyr = [im]
    current_im = im
    
    for _ in range(max_levels - 1):
        # Blur and subsample
        downsampled_im = _blur_and_downsample(current_im, filter_vec)
        
        # Check if the image is too small
        if downsampled_im.shape[0] < 2 or downsampled_im.shape[1] < 2:
            break
            
        pyr.append(downsampled_im)
        current_im = downsampled_im
        
    return pyr, filter_vec

def build_laplacian_pyramid(im, max_levels, filter_size):
    """
    Builds a Laplacian pyramid for a given image.
    :param im: a grayscale image with double values in [0, 1]
    :param max_levels: the maximal number of levels in the resulting pyramid.
    :param filter_size: the size of the Gaussian filter (an odd scalar that represents a squared filter)
                        to be used in constructing the pyramid filter
    :return: pyr, filter_vec. Where pyr is the resulting pyramid as a standard python list with maximum max_levels elements,
             where each element is a grayscale image.
             filter_vec is a row vector of shape (1, filter_size) used for the pyramid construction.
    """
    g_pyr, filter_vec = build_gaussian_pyramid(im, max_levels, filter_size)
    
    l_pyr = []
    for i in range(len(g_pyr) - 1):
        # Expand the next level to match current level size
        expanded_next = _expand(g_pyr[i+1], filter_vec)
        
        # Crop if needed
        if expanded_next.shape != g_pyr[i].shape:
             expanded_next = expanded_next[:g_pyr[i].shape[0], :g_pyr[i].shape[1]]

        laplacian = g_pyr[i] - expanded_next
        l_pyr.append(laplacian)
        
    l_pyr.append(g_pyr[-1]) # The last level is the same as Gaussian
    
    return l_pyr, filter_vec

def laplacian_to_image(lpyr, filter_size, coeff):
    """
    Reconstructs an image from its Laplacian Pyramid.
    :param lpyr: Laplacian pyramid as a standard python list.
    :param filter_size: the size of the Gaussian filter (an odd scalar that represents a squared filter)
    :param coeff: a python list. The list length is the same as the number of levels in the pyramid lpyr.
                  Before reconstructing the image img, multiply each level i of the laplacian pyramid by its corresponding coefficient coeff[i].
    :return: img. The reconstructed image.
    """
    # Create the 1D Gaussian filter
    filter_vec = _get_filter_vec(filter_size)
    
    # Start with the last level (smallest)
    current_im = lpyr[-1] * coeff[-1]
    
    for i in range(len(lpyr) - 2, -1, -1):
        expanded_im = _expand(current_im, filter_vec)
        
        # Match size with the next level
        target_shape = lpyr[i].shape
        if expanded_im.shape != target_shape:
             expanded_im = expanded_im[:target_shape[0], :target_shape[1]]
             
        current_im = expanded_im + (lpyr[i] * coeff[i])
        
    return current_im

def blend_images(im1, im2, mask, max_levels, filter_size):
    """
    Blends two images using Laplacian pyramid blending.
    :param im1: a grayscale image with double values in [0, 1]
    :param im2: a grayscale image with double values in [0, 1]
    :param mask: a boolean mask (or float in [0,1]) representing the mask.
    :param max_levels: the maximal number of levels in the resulting pyramid.
    :param filter_size: the size of the Gaussian filter (an odd scalar that represents a squared filter)
    :return: blended_im. The blended image.
    """
    # Construct Laplacian pyramids for both images
    l1, filter_vec = build_laplacian_pyramid(im1, max_levels, filter_size)
    l2, _ = build_laplacian_pyramid(im2, max_levels, filter_size)
    
    # Construct Gaussian pyramid for the mask
    mask = mask.astype(np.float64)
    gm, _ = build_gaussian_pyramid(mask, max_levels, filter_size)
    
    # Blend the pyramids
    l_out = []
    for i in range(len(l1)):
        curr_mask = gm[i]
        if curr_mask.shape != l1[i].shape:
             curr_mask = curr_mask[:l1[i].shape[0], :l1[i].shape[1]]
             
        blended_level = curr_mask * l1[i] + (1 - curr_mask) * l2[i]
        l_out.append(blended_level)
        
    # Reconstruct the image
    coeff = [1] * len(l_out)
    blended_im = laplacian_to_image(l_out, filter_size, coeff)
    
    # Clip to [0, 1] just in case
    return np.clip(blended_im, 0, 1)

# --- Fourier Hybrid Logic ---

def create_hybrid_image(im1, im2, cutoff):
    """
    Creates a hybrid image from two images.
    :param im1: The image to be seen up close (High frequency).
    :param im2: The image to be seen from afar (Low frequency).
    :param cutoff: The cutoff frequency for the Gaussian filter.
    :return: The hybrid image.
    """
    # Ensure images are the same size
    assert im1.shape == im2.shape, "Images must be the same size"
    
    # Compute FFT
    im1_fft = fftshift(fft2(im1))
    im2_fft = fftshift(fft2(im2))
    
    # Create Gaussian filter
    rows, cols = im1.shape
    crow, ccol = rows // 2, cols // 2
    
    # Create a grid of coordinates
    x = np.arange(cols)
    y = np.arange(rows)
    xv, yv = np.meshgrid(x, y)
    
    # Calculate distance from center
    dist_sq = (xv - ccol)**2 + (yv - crow)**2
    
    # Gaussian Low Pass Filter
    sigma = cutoff
    low_pass = np.exp(-dist_sq / (2 * sigma**2))
    
    # High Pass Filter is 1 - Low Pass
    high_pass = 1 - low_pass
    
    # Apply filters
    filtered_fft1 = im1_fft * high_pass
    filtered_fft2 = im2_fft * low_pass
    
    # Combine
    hybrid_fft = filtered_fft1 + filtered_fft2
    
    # Inverse FFT
    hybrid_image = np.real(ifft2(ifftshift(hybrid_fft)))
    
    # Clip to valid range [0, 1]
    return np.clip(hybrid_image, 0, 1)

# --- Mask Creation Logic ---

def create_precise_mask(image_path, output_mask_path):
    """
    Opens an image, allows the user to select an ROI, and generates a binary mask using GrabCut.
    """
    # 1. Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return

    # Create initial mask for GrabCut
    mask = np.zeros(img.shape[:2], np.uint8)
    
    # Internal models required by GrabCut
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    print("Step 1: Draw a rectangle around the object using your mouse.")
    print("        Press ENTER or SPACE after drawing to start segmentation.")
    print("        Press 'c' to cancel selection.")
    
    # 2. Select Region of Interest (ROI)
    cv2.namedWindow("Select Object", cv2.WINDOW_NORMAL)
    rect = cv2.selectROI("Select Object", img, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Object")
    
    if rect == (0, 0, 0, 0):
        print("No selection made. Exiting.")
        return

    print("Segmenting... please wait.")

    # 3. Run GrabCut
    # iterCount=5, mode=GC_INIT_WITH_RECT
    try:
        cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    except Exception as e:
        print(f"Error during GrabCut: {e}")
        return

    # 4. Process the result
    # GrabCut modifies the mask in-place.
    # 0 = GC_BGD (Background), 1 = GC_FGD (Foreground), 2 = GC_PR_BGD (Probable Background), 3 = GC_PR_FGD (Probable Foreground)
    # We treat 1 and 3 as foreground (1), and 0 and 2 as background (0).
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    
    # Scale to 0-255 for saving
    final_mask = mask2 * 255

    # Optional: Morphological Opening to remove small noise
    kernel = np.ones((3,3), np.uint8)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # 5. Save the result
    cv2.imwrite(output_mask_path, final_mask)
    print(f"Precise mask saved to: {output_mask_path}")
    
    # Display result (optional, can be skipped if no-show is preferred, but interactive tool implies visuals)
    # create_precise_mask is interactive by nature.
    
    seg_img = img * mask2[:, :, np.newaxis]
    
    cv2.imshow("Segmented Object", seg_img)
    cv2.imshow("Final Mask", final_mask)
    print("Press any key to close windows...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# --- Main App Logic ---

def load_and_resize(path, size=(512, 512), grayscale=False):
    im = cv2.imread(path)
    if im is None:
        try:
            from PIL import Image
            pil_im = Image.open(path)
            im = np.array(pil_im)
            if im.ndim == 3 and im.shape[2] == 3:
                im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            elif im.ndim == 3 and im.shape[2] == 4:
                im = cv2.cvtColor(im, cv2.COLOR_RGBA2BGR)
        except Exception as e:
            pass

    if im is None:
        raise ValueError(f"Could not load image: {path}")
    
    im = cv2.resize(im, size)
    if grayscale:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    else:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    
    return im.astype(np.float64) / 255.0

def create_mask(size=(512, 512)):
    # Create a simple vertical split mask
    mask = np.zeros(size, dtype=np.float64)
    mask[:, :size[1]//2] = 1.0
    return mask

def blend_color_images(im1, im2, mask, max_levels, filter_size):
    # Blend each channel separately
    channels = []
    for i in range(3):
        blended_channel = blend_images(im1[:, :, i], im2[:, :, i], mask, max_levels, filter_size)
        channels.append(blended_channel)
    return np.dstack(channels)

def create_hybrid_color_image(im1, im2, cutoff):
    # Apply hybrid to each channel separately
    channels = []
    for i in range(3):
        hybrid_channel = create_hybrid_image(im1[:, :, i], im2[:, :, i], cutoff)
        channels.append(hybrid_channel)
    return np.dstack(channels)

def main():
    parser = argparse.ArgumentParser(description='Image Combination Tool: Pyramid Blending or Fourier Hybrid.')
    parser.add_argument('--im1', type=str, default=relpath('inputs/hana.jpeg'), help='Path to the first image (Close/Left/Foreground)')
    parser.add_argument('--im2', type=str, default=relpath('inputs/tzipi.jpeg'), help='Path to the second image (Far/Right/Background)')
    parser.add_argument('--mask', type=str, help='Path to the mask image (for pyramid blending)')
    parser.add_argument('--out', type=str, default='result.png', help='Path to save the result')
    parser.add_argument('--mode', type=str, choices=['pyramid', 'fourier', 'mask'], default='pyramid', help='Combination mode')
    parser.add_argument('--cutoff', type=float, default=20.0, help='Cutoff frequency for Fourier hybrid mode')
    parser.add_argument('--grayscale', action='store_true', help='Process images in grayscale')
    
    args = parser.parse_args()

    # Handle Mask Creation Mode
    if args.mode == 'mask':
        create_precise_mask(args.im1, relpath(args.out))
        return

    # Paths
    im1_path = args.im1
    im2_path = args.im2
    
    # Load and resize
    try:
        im1 = load_and_resize(im1_path, grayscale=args.grayscale)
        im2 = load_and_resize(im2_path, grayscale=args.grayscale)
    except Exception as e:
        print(e)
        return

    result = None
    mask = None

    if args.mode == 'pyramid':
        # Create mask
        if args.mask:
            try:
                mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, (im1.shape[1], im1.shape[0]))
                mask = mask.astype(np.float64) / 255.0
            except Exception as e:
                print(f"Error loading mask: {e}")
                return
        else:
            mask = create_mask(im1.shape[:2])
        
        # Parameters
        max_levels = 5
        filter_size = 5 # Gaussian filter size
        
        # Blend
        print(f"Blending {im1_path} and {im2_path} using Pyramid Blending...")
        if args.grayscale:
            result = blend_images(im1, im2, mask, max_levels, filter_size)
        else:
            result = blend_color_images(im1, im2, mask, max_levels, filter_size)
        
    elif args.mode == 'fourier':
        print(f"Creating Hybrid Image from {im1_path} (Close) and {im2_path} (Far) using Fourier...")
        if args.grayscale:
             result = create_hybrid_image(im1, im2, args.cutoff)
        else:
             result = create_hybrid_color_image(im1, im2, args.cutoff)

    # Save Result
    output_path = relpath(args.out)
    
    # Convert result to uint8 [0, 255]
    to_save = np.clip(result * 255.0, 0, 255).astype(np.uint8)
    
    # Convert to BGR if necessary for OpenCV
    if not args.grayscale and len(to_save.shape) == 3:
        to_save = cv2.cvtColor(to_save, cv2.COLOR_RGB2BGR)
        
    cv2.imwrite(output_path, to_save)
    print(f"Result saved to {output_path}")

if __name__ == '__main__':
    main()
