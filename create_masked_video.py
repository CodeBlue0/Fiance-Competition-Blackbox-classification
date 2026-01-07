import cv2
import argparse
import os

def mask_video(input_path, output_path, mask_ratio=0.15):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error opening video: {input_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    mask_height = int(height * mask_ratio)
    start_y = height - mask_height
    
    print(f"Processing {input_path} -> {output_path}")
    print(f"Masking bottom {mask_height} pixels (Ratio: {mask_ratio})")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Apply mask (black rectangle at bottom)
        cv2.rectangle(frame, (0, start_y), (width, height), (0, 0, 0), -1)

        out.write(frame)
        frame_count += 1
        if frame_count % 50 == 0:
            print(f"Processed {frame_count}/{total_frames} frames", end='\r')

    cap.release()
    out.release()
    print(f"\nDone. Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str)
    parser.add_argument('output_path', type=str)
    parser.add_argument('--ratio', type=float, default=0.15, help="Ratio of bottom height to mask (default 0.15)")
    args = parser.parse_args()
    
    mask_video(args.input_path, args.output_path, args.ratio)
