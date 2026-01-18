try:
    import cv2
except ImportError:
    cv2 = None
import os

def video_to_frames(video_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
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

if __name__ == "__main__":
    # Example usage: extract boat.mp4
    # Adjust paths as needed based on where the script is run from
    input_video = os.path.abspath("Exercise Inputs-20260105/boat.mp4")
    output_folder = os.path.abspath("input_frames/boat")
    
    if not os.path.exists(input_video):
        print(f"Error: Video file not found at {input_video}")
    else:
        video_to_frames(input_video, output_folder)
