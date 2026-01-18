import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift

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
    # H(u,v) = exp(-D^2 / (2 * D0^2)) where D0 is cutoff
    # Note: cutoff in the prompt might refer to sigma or frequency radius. 
    # Standard interpretation for "cutoff" in this context often maps to sigma.
    sigma = cutoff
    low_pass = np.exp(-dist_sq / (2 * sigma**2))
    
    # High Pass Filter is 1 - Low Pass
    high_pass = 1 - low_pass
    
    # Apply filters
    # im1 (close) gets High Pass
    # im2 (far) gets Low Pass
    filtered_fft1 = im1_fft * high_pass
    filtered_fft2 = im2_fft * low_pass
    
    # Combine
    hybrid_fft = filtered_fft1 + filtered_fft2
    
    # Inverse FFT
    hybrid_image = np.real(ifft2(ifftshift(hybrid_fft)))
    
    # Clip to valid range [0, 1]
    return np.clip(hybrid_image, 0, 1)
