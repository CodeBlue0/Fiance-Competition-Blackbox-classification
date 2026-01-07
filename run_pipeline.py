import os
import sys
import subprocess
import argparse

def run_command(command):
    print(f"\n[Pipeline] Running: {command}")
    try:
        subprocess.check_call(command, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"[Pipeline] Error: Command failed with exit code {e.returncode}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Run the full Driver Safety Assessment Pipeline")
    parser.add_argument("video_file", help="Name of the video file (e.g., test7.mp4) or path relative to input_video/")
    args = parser.parse_args()

    # Handle paths
    # If user provides full path or just filename, ensure we handle it
    if os.path.exists(args.video_file):
        video_path = args.video_file
        video_name = os.path.basename(video_path)
    elif os.path.exists(os.path.join("input_video", args.video_file)):
        video_path = os.path.join("input_video", args.video_file)
        video_name = args.video_file
    else:
        print(f"Error: Video file '{args.video_file}' not found.")
        sys.exit(1)
        
    video_name_no_ext = os.path.splitext(video_name)[0]
    
    # 1. Stage 1: Context Analysis (LLM + Tracking)
    print("="*60)
    print("STAGE 1: Context Analysis (LLM + Tracking)")
    print("="*60)
    run_command(f"python3 analyze_video_context.py {video_path} --debug")
    
    # 2. Stage 1-B: Risky Object Visualization (Optional but requested)
    # process_video.py expects just the filename inside input_video directory usually, 
    # checking process_video.py arguments: it takes 'video_name' and assumes input_video/{video_name}
    print("="*60)
    print("STAGE 1-B: Risky Object Visualization")
    print("="*60)
    # process_video.py logic: video_path = os.path.join(input_dir, target_video_name)
    # So we must pass just the filename
    run_command(f"python3 process_video.py {video_name}")
    
    # 3. Stage 2: Integrated Safety Assessment
    print("="*60)
    print("STAGE 2: Integrated Safety Assessment")
    print("="*60)
    json_path = os.path.join("output_llm", f"result_{video_name_no_ext}.json")
    run_command(f"python3 assess_driver_safety.py {video_path} {json_path}")
    
    print("="*60)
    print("PIPELINE COMPLETE")
    print(f"Final Assessment: result/{video_name_no_ext}.json")
    print("="*60)

if __name__ == "__main__":
    main()
