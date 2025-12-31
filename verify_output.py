import cv2
import os
import numpy as np

def verify_video(filepath):
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        print(f"Failed to open {filepath}")
        return False
    
    red_frames = 0
    green_frames = 0
    total_frames = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        total_frames += 1
        # Check pixel at (0, 0) or (10, 10). 
        # Using (10, 10) to be safely inside the 20px border
        # Frame is BGR
        pixel = frame[10, 10]
        
        # Check for Green (0, 255, 0)
        # Allow some compression artifact tolerance
        if pixel[1] > 200 and pixel[0] < 50 and pixel[2] < 50:
            green_frames += 1
            
        # Check for Red (0, 0, 255) -> BGR: (Blue, Green, Red) -> (0, 0, 255)
        # pixel[2] is Red channel
        if pixel[2] > 200 and pixel[0] < 50 and pixel[1] < 50:
            red_frames += 1

    cap.release()
    
    print(f"Video: {os.path.basename(filepath)}")
    print(f"  Total Frames: {total_frames}")
    print(f"  Green Frames (Accel): {green_frames}")
    print(f"  Red Frames (Decel): {red_frames}")
    
    if red_frames > 0 and green_frames > 0:
        print("  [PASS] Both acceleration and deceleration detected.")
        return True
    elif red_frames > 0 or green_frames > 0:
         print("  [WARN] Only one type of event detected (but something was detected).")
         return True
    else:
        print("  [FAIL] No events detected.")
        return False

def main():
    output_dir = "output"
    files = ["processed_case1.mp4", "processed_case2.mp4", "processed_case3.mp4"]
    
    results = []
    for f in files:
        path = os.path.join(output_dir, f)
        if os.path.exists(path):
            results.append(verify_video(path))
        else:
            print(f"File not found: {path}")
            results.append(False)
            
    if all(results):
        print("\nOVERALL STATUS: PASS")
    else:
        print("\nOVERALL STATUS: PARTIAL/FAIL")

if __name__ == "__main__":
    main()
