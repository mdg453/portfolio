import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from pyramid_combination import blend_images, relpath
from fourier_combination import create_hybrid_image

def load_and_resize(path, size=(512, 512), grayscale=False):
    im = cv2.imread(path)
    if im is None:
        # Try loading with PIL (supports more formats like AVIF)
        try:
            from PIL import Image
            pil_im = Image.open(path)
            im = np.array(pil_im)
            # Convert RGB to BGR for consistency with cv2 if needed, 
            # but we convert to RGB right after anyway.
            # PIL is usually RGB. cv2.cvtColor expects BGR or BGRA.
            # Let's just handle it as RGB directly.
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
        # Add channel dimension for consistency if desired, or keep as 2D. 
        # The blend functions expect 2D for single channel.
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
    import argparse
    parser = argparse.ArgumentParser(description='Image Combination Tool: Pyramid Blending or Fourier Hybrid.')
    parser.add_argument('--im1', type=str, default=relpath('images/hana.jpeg'), help='Path to the first image (Close/Left)')
    parser.add_argument('--im2', type=str, default=relpath('images/tzipi.jpeg'), help='Path to the second image (Far/Right)')
    parser.add_argument('--mask', type=str, help='Path to the mask image (for pyramid blending)')
    parser.add_argument('--out', type=str, default='result.png', help='Path to save the result')
    parser.add_argument('--mode', type=str, choices=['pyramid', 'fourier'], default='pyramid', help='Combination mode')
    parser.add_argument('--cutoff', type=float, default=20.0, help='Cutoff frequency for Fourier hybrid mode')
    parser.add_argument('--grayscale', action='store_true', help='Process images in grayscale')
    parser.add_argument('--no-show', action='store_true', help='Do not display the result window')
    
    args = parser.parse_args()

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

    # Display results
    plt.figure(figsize=(12, 6))
    
    cmap = 'gray' if args.grayscale else None
    
    if args.mode == 'pyramid':
        plt.subplot(1, 4, 1)
        plt.imshow(im1, cmap=cmap)
        plt.title('Image 1')
        plt.axis('off')
        
        plt.subplot(1, 4, 2)
        plt.imshow(im2, cmap=cmap)
        plt.title('Image 2')
        plt.axis('off')
        
        plt.subplot(1, 4, 3)
        plt.imshow(mask, cmap='gray')
        plt.title('Mask')
        plt.axis('off')
        
        plt.subplot(1, 4, 4)
        plt.imshow(result, cmap=cmap)
        plt.title('Pyramid Result')
        plt.axis('off')
        
    else: # fourier
        plt.subplot(1, 3, 1)
        plt.imshow(im1, cmap=cmap)
        plt.title('High Freq (Close)')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(im2, cmap=cmap)
        plt.title('Low Freq (Far)')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(result, cmap=cmap)
        plt.title('Hybrid Result')
        plt.axis('off')
    
    plt.tight_layout()
    output_path = relpath(args.out)
    plt.savefig(output_path)
    print(f"Result saved to {output_path}")
    if not args.no_show:
        plt.show()

if __name__ == '__main__':
    main()
