import cv2
import numpy as np

def detect_gas_leak_anomaly(image_path, output_path):
    print(f"Loading image from {image_path}...")
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not load image.")
        return

    # Convert to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define color range for dead/yellow/brown vegetation (the anomaly)
    # H: 10-45 (yellows and browns), S: 50-255, V: 50-255
    lower_brown = np.array([10, 50, 50])
    upper_brown = np.array([45, 255, 255])

    # Create a mask for the anomalous colors
    mask = cv2.inRange(hsv, lower_brown, upper_brown)

    # Morphological operations to remove noise
    kernel = np.ones((7,7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) # Remove small noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) # Fill holes

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    anomalies_detected = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Filter small noise contours
        if area > 20000:  # Adjust area threshold depending on image size
            anomalies_detected += 1
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Draw bounding box
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 4) # Red box
            
            # Add label with background for readability
            label = "WARNING: POTENTIAL GAS LEAK"
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
            
            # Draw black background rectangle for text
            cv2.rectangle(img, (x, y - text_height - 15), (x + text_width, y), (0, 0, 0), cv2.FILLED)
            cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    print(f"Detected {anomalies_detected} large anomalies.")
    
    # Save the output
    cv2.imwrite(output_path, img)
    print(f"Processed image saved to {output_path}")

if __name__ == "__main__":
    input_file = "input_aerial.jpg"
    output_file = "output_detection.jpg"
    detect_gas_leak_anomaly(input_file, output_file)
