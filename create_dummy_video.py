
import cv2
import numpy as np
import os

def create_video():
    if not os.path.exists('input_video'):
        os.makedirs('input_video')
        
    width, height = 1280, 720
    fps = 20
    duration = 6 # seconds
    
    out_path = 'input_video/test_video.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    
    for i in range(duration * fps):
        # Create a black frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Draw a moving rectangle (car)
        x = int((i * 5) % width)
        y = 400
        cv2.rectangle(frame, (x, y), (x+100, y+50), (255, 0, 0), -1)
        
        # Add text
        cv2.putText(frame, f"Frame {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(frame)
        
    out.release()
    print(f"Created {out_path}")

if __name__ == "__main__":
    create_video()
