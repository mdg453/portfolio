import numpy as np
import matplotlib.pyplot as plt
from hybrid import create_hybrid_image

def test_hybrid_image():
    # Create synthetic images
    size = 256
    x = np.linspace(0, 4*np.pi, size)
    y = np.linspace(0, 4*np.pi, size)
    xv, yv = np.meshgrid(x, y)
    
    # Image 1: High frequency (Sine wave)
    im1 = 0.5 + 0.5 * np.sin(xv * 10)
    
    # Image 2: Low frequency (Gaussian blob)
    im2 = np.exp(-((xv - 2*np.pi)**2 + (yv - 2*np.pi)**2) / 10)
    
    # Create hybrid image
    cutoff = 20
    hybrid = create_hybrid_image(im1, im2, cutoff)
    
    # Visualize
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(im1, cmap='gray')
    plt.title('High Freq (Close)')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(im2, cmap='gray')
    plt.title('Low Freq (Far)')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(hybrid, cmap='gray')
    plt.title('Hybrid')
    plt.axis('off')
    
    plt.savefig('hybrid_test_result.png')
    print("Test completed. Result saved to hybrid_test_result.png")

if __name__ == '__main__':
    test_hybrid_image()
