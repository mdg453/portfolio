import cv2
import numpy as np
import os
import argparse

def relpath(filename):
    return os.path.join(os.path.dirname(__file__), filename)

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
    
    # Display result
    seg_img = img * mask2[:, :, np.newaxis]
    
    cv2.imshow("Segmented Object", seg_img)
    cv2.imshow("Final Mask", final_mask)
    print("Press any key to close windows...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Create a binary mask from an image using interactive GrabCut.')
    parser.add_argument('--image', type=str, required=True, help='Path to the input image')
    parser.add_argument('--out', type=str, default='mask.png', help='Path to save the output mask')
    
    args = parser.parse_args()
    
    # Handle relative paths properly if run from the script directory
    image_path = args.image if os.path.isabs(args.image) else relpath(args.image)
    output_path = args.out if os.path.isabs(args.out) else relpath(args.out)
    
    create_precise_mask(image_path, output_path)

if __name__ == '__main__':
    main()
