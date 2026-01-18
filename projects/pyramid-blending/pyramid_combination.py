import numpy as np
from scipy.ndimage import convolve
import os

def relpath(filename):
    return os.path.join(os.path.dirname(__file__), filename)

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
        
        # Check if the image is too small (less than 16x16 is a common heuristic, or just 1 pixel)
        # The exercise usually implies we stop if we can't downsample further properly
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
        
        # If sizes don't match exactly due to odd dimensions, we might need to crop or pad.
        # Usually in these exercises we assume power of 2 or handle it.
        # Let's crop the expanded image to match the current gaussian level if needed.
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
    # The mask should be float for blending
    mask = mask.astype(np.float64)
    gm, _ = build_gaussian_pyramid(mask, max_levels, filter_size)
    
    # Blend the pyramids
    l_out = []
    for i in range(len(l1)):
        # L_out[i] = Gm[i] * L1[i] + (1 - Gm[i]) * L2[i]
        # Ensure mask shape matches pyramid level shape
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

# --- Helper Functions ---

def _get_filter_vec(filter_size):
    # Binomial coefficients approximation for Gaussian
    # For filter_size=3: [1, 2, 1] / 4
    # For filter_size=5: [1, 4, 6, 4, 1] / 16
    # We can generate this by convolving [1, 1] with itself filter_size-1 times
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
import numpy as np
from scipy.ndimage import convolve
import os

def relpath(filename):
    return os.path.join(os.path.dirname(__file__), filename)

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
        
        # Check if the image is too small (less than 16x16 is a common heuristic, or just 1 pixel)
        # The exercise usually implies we stop if we can't downsample further properly
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
        
        # If sizes don't match exactly due to odd dimensions, we might need to crop or pad.
        # Usually in these exercises we assume power of 2 or handle it.
        # Let's crop the expanded image to match the current gaussian level if needed.
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
    # The mask should be float for blending
    mask = mask.astype(np.float64)
    gm, _ = build_gaussian_pyramid(mask, max_levels, filter_size)
    
    # Blend the pyramids
    l_out = []
    for i in range(len(l1)):
        # L_out[i] = Gm[i] * L1[i] + (1 - Gm[i]) * L2[i]
        # Ensure mask shape matches pyramid level shape
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

# --- Helper Functions ---

def _get_filter_vec(filter_size):
    # Binomial coefficients approximation for Gaussian
    # For filter_size=3: [1, 2, 1] / 4
    # For filter_size=5: [1, 4, 6, 4, 1] / 16
    # We can generate this by convolving [1, 1] with itself filter_size-1 times
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
    # (Since 3/4 of pixels are zero, average brightness drops by 4, but we only sum non-zeros... 
    # actually the standard is to multiply filter by 2)
    
    # Convolve x
    expanded = convolve(upsampled, 2 * filter_vec, mode='mirror')
    # Convolve x
    expanded = convolve(upsampled, 2 * filter_vec, mode='mirror')
    
    return expanded
