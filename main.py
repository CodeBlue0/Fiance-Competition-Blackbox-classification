import os
import cv2
from video_processor import VideoProcessor

def main():
    data_dir = "data"
    output_dir = "output"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    video_files = ["case1.mp4", "case2.mp4", "case3.mp4"]
    
    # Timestamps (Start, End) in seconds
    timestamps = {
        "case1.mp4": (6, 9),
        "case2.mp4": (13, 17),
        "case3.mp4": (7, 11)
    }
    
    processor = VideoProcessor()

    print("Starting processing...")
    
    for filename in video_files:
        input_path = os.path.join(data_dir, filename)
        output_path = os.path.join(output_dir, f"processed_{filename}")
        
        if not os.path.exists(input_path):
            print(f"File not found: {input_path}")
            continue
            
        print(f"Processing {filename}...")
        try:
            # Run without display for purely automated processing
            t_start, t_end = timestamps.get(filename, (0, 0))
            print(f"  > Validating Window: {t_start}s to {t_end}s")
            
            processor.process_video(input_path, t_start, t_end, output_path, show_display=False) 
            
            # Print Validation Stats
            valid_frames = processor.valid_braking_frames
            total_braking = processor.total_braking_frames
            
            print(f"Finished {filename}. Saved to {output_path}")
            if total_braking > 0:
                ratio = (valid_frames / total_braking) * 100
                print(f"  > Total Braking Frames: {total_braking}")
                print(f"  > Valid Braking Frames (Obstacle Detected): {valid_frames}")
                print(f"  > Validation Ratio: {ratio:.1f}%")
                
                # Identify Primary Obstacle
                if valid_frames > 0:
                    primary_obstacle, count = processor.obstacle_counts.most_common(1)[0]
                    print(f"  > Primary Obstacle: {primary_obstacle} ({count} detection frames)")
                
                if ratio > 20: # Threshold: >20% (Lowered for early cutoff logic + high strictness)
                    print("  > JUDGMENT: VALID SUDDEN BRAKING (Obstacle Confirmed)")
                else:
                    print("  > JUDGMENT: INVALID/EMPTY BRAKING (No significant obstacle in danger zone)")
            else:
                print("  > JUDGMENT: NO BRAKING DETECTED")

            # Reset stats for next video
            processor.valid_braking_frames = 0
            processor.total_braking_frames = 0
            processor.obstacle_counts.clear()
        except Exception as e:
            print(f"Failed to process {filename}: {e}")

    print("All processing complete.")

if __name__ == "__main__":
    main()
