
import os
import sys
import argparse
import cv2
import os
import sys
import argparse
import cv2
import json
import base64
import requests
from PIL import Image
from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()

def encode_image(image):
    # Use PNG for lossless quality to help with small details like traffic lights
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode('utf-8')

def analyze_chunk(frames, detections_list, fps_val=1.0):
    api_key = os.environ.get("PERPLEXITY_API_KEY")
    if not api_key:
        return "Error: PERPLEXITY_API_KEY environment variable not set."

    # Prepare logic for Perplexity API
    # We will send the frames as a list of images.
    
    encoded_images = []
    for f in frames:
        encoded_images.append(encode_image(f))
        
    # Format detections for the prompt
    # detections_list is a list of lists: [[(id, class_name, box), ...], ...] corresponding to frames
    detection_text = "\n# Detected Objects (YOLOv8):\n"
    for i, dets in enumerate(detections_list):
        detection_text += f"Frame {i+1}: "
        if not dets:
            detection_text += "No objects detected.\n"
        else:
            det_strs = []
            for d in dets:
                tid, cls, box = d
                # box is [x1, y1, x2, y2]
                det_strs.append(f"{cls} (ID {tid}) at [{int(box[0])}, {int(box[1])}, {int(box[2])}, {int(box[3])}]")
            detection_text += ", ".join(det_strs) + "\n"
            
    print(f"--- sent to LLM ---\n{detection_text}-------------------")

    # Advanced Prompt for Traffic Analysis
    system_prompt = """# Role
You are an expert Traffic Scene Observer. Your goal is to provide a detailed, factual, and chronological description of the video frames.
**Do NOT assess risk or look for violations.** Just report what is physically happening.

# Task
Analyze the provided video frames frame-by-frame. **Each frame is labeled with a sequence number 'Frame X'.**
**Focus ONLY on dynamic elements (vehicles, pedestrians, traffic lights).**
**IGNORE background details** such as buildings, trees, sky, weather, or city architecture. Only describe what affects the traffic situation directly.

# Video Info
- **Frame Rate**: The images are sampled at **3 FPS** (Frames Per Second).
- **Timing**: Each frame represents approximately **0.33 seconds** of real time.
- **Duration**: The 15 frames cover a total duration of **5 seconds**.

# Additional Data
You are provided with a list of "Detected Objects" from a YOLO model for each frame, including Object IDs and Bounding Boxes. 
Use this date to identify specific vehicles or pedestrians by their ID (e.g., "Car (ID 5)").
Note that the bounding box format is [x1, y1, x2, y2] (top-left to bottom-right).

# Observation Checklist:
1. **Pedestrians**: Note exactly which frames a pedestrian is visible. Describe their location (e.g., crosswalk, sidewalk) and movement.
2. **Traffic Lights (CRITICAL)**: Differentiate between **Vehicle Signals** and **Pedestrian Signals**.
   - **Vehicle Signal**: Typically 3-4 circular lights, overhead or high on poles. (Red/Yellow/Green).
   - **Pedestrian Signal**: Typically lower, square housing, often with "Standing Man" (Red) or "Walking Man" (Green) icons.
   - **Status**: Report e.g., "Vehicle Light is Green", "Pedestrian Light is Red".
3. **Vehicles**: Describe the relative distance and behavior of other vehicles (e.g., "Vehicle ahead is far away", "Vehicle in left lane is passing").
4. **Ego-Vehicle**: Describe the ego-vehicle's motion based on the view (e.g., "moving forward constant speed", "slowing down").

# Output Format (JSON):
Return the result in valid JSON format.
{
    "situation_summary": "A detailed chronological summary of the scene. Example: 'At Frame 1, a pedestrian (ID 3) appears on the right sidewalk. The Vehicle Traffic Light is Green. The car (ID 8) ahead is maintaining a long distance.'",
    "pedestrian_events": "Specific details about pedestrians, e.g., 'pedestrian (ID 3) visible in Frames 3-10 on right sidewalk'",
    "traffic_light_status": "Status of BOTH signals if visible: e.g., 'Vehicle: Green, Pedestrian: Red'",
    "vehicle_context": "Details on other vehicles and distances"
}
"""

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": f"Analyze these frames. {detection_text}"
        }
    ]
    
    # NOTE: Perplexity API currently has limited support for direct image uploads in the 'chat/completions' style compared to GPT-4o.
    # However, if using a model that supports it, we might need a specific structure.
    # As of early 2025, Perplexity's 'sonar' models might expect text. 
    # If the user specifically requested Perplexity API for *video*, we assume the model handles it or we use a compatible one.
    # Standard Chat format for Multi-modal usually involves:
    # content: [{"type": "text", ...}, {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}]
    
    user_content = [{"type": "text", "text": "Here are the frames from the video chunk. Please analyze them according to the system instructions."}]
    
    for b64_img in encoded_images:
        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{b64_img}"
            }
        })
        
    messages[1]["content"] = user_content

    payload = {
        "model": "sonar", 
        "messages": messages,
        "temperature": 0.2,
        "top_p": 0.9,
        "return_citations": False,
        "search_domain_filter": ["perplexity.ai"],
        "return_images": False,
        "return_related_questions": False,
        "search_recency_filter": "month",
        "top_k": 0,
        "stream": False,
        "presence_penalty": 0,
        "frequency_penalty": 1
    }
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post("https://api.perplexity.ai/chat/completions", json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content']
    except Exception as e:
        return f"Error calling Perplexity API: {str(e)} Response: {response.text if 'response' in locals() else ''}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('video_path', type=str)
    parser.add_argument('--debug', action='store_true', help='Save debug frames to debug_frames/')
    args = parser.parse_args()
    
    video_path = args.video_path
    if not os.path.exists(video_path):
        print(f"File not found: {video_path}")
        sys.exit(1)
        
    if "PERPLEXITY_API_KEY" not in os.environ:
        print("Error: PERPLEXITY_API_KEY environment variable not set.")
        # We can try to proceed if the user plans to set it, but better warn.
        # sys.exit(1)

    print("Using Perplexity API for analysis...")
    print("Loading YOLOv8n for object tracking...")
    yolo = YOLO('yolov8n.pt')
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    
    # import json -> moved to top

    # Create output directory
    output_dir = "output_llm"
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare output filename: result_filename.json
    video_basename = os.path.basename(video_path)
    video_name_no_ext = os.path.splitext(video_basename)[0]
    output_filename = f"result_{video_name_no_ext}.json"
    output_path = os.path.join(output_dir, output_filename)
    
    analysis_results = []
    
    # Analyze 5-second chunks
    chunk_len = int(5 * fps)
    
    curr_frame = 0
    chunk_idx = 0
    
    # Create debug directory if debug mode is on
    if args.debug:
        debug_dir = "debug_frames"
        os.makedirs(debug_dir, exist_ok=True)
    
    while curr_frame < total:
        frames = []
        for _ in range(chunk_len):
            ret, frame = cap.read()
            if not ret: break
            frames.append(frame)
        
        if len(frames) < 10: break # Skip tiny chunks
        
        print(f"\nAnalyzing Chunk {chunk_idx+1} ({curr_frame} - {curr_frame+len(frames)} frames)...")
        
        # Run YOLO Tracking on the whole chunk to maintain ID consistency
        # We need to run it on all frames in the chunk, not just sampled ones, for better tracking?
        # Yes, `track` handles lists of frames.
        print("  Running YOLO tracking...")
        yolo_results = yolo.track(frames, persist=True, verbose=False, tracker="bytetrack.yaml")
        
        # Sample 15 frames (increase to 3 fps for better event detection)
        # 15 frames in 5 seconds = 3 fps effective for the model
        indices = [int(i) for i in list(range(0, len(frames), max(1, len(frames)//15)))[:15]]
        sampled_frames = [frames[i] for i in indices]
        
        # Extract detections for sampled frames
        chunk_detections = [] # list of lists for the sampled frames
        for idx in indices:
            res = yolo_results[idx]
            frame_dets = []
            if res.boxes and res.boxes.id is not None:
                boxes = res.boxes.xyxy.cpu().numpy()
                ids = res.boxes.id.cpu().numpy()
                clss = res.boxes.cls.cpu().numpy()
                names = res.names
                
                h, w, _ = frames[0].shape
                min_w = w / 15.0
                min_h = h / 15.0

                for b, tid, cls in zip(boxes, ids, clss):
                    # Filter Classes (Person, Bicycle, Car, Motorcycle, Bus, Truck)
                    if int(cls) not in [0, 1, 2, 3, 5, 7]:
                        continue
                        
                    x1, y1, x2, y2 = b
                    
                    # Filter Small Objects
                    if (x2 - x1) < min_w or (y2 - y1) < min_h:
                        continue

                    class_name = names[int(cls)]
                    frame_dets.append((int(tid), class_name, b))
            chunk_detections.append(frame_dets)

        # Annotate frames with sequence numbers for the VLM
        annotated_frames = []
        for i, f in enumerate(sampled_frames): # Iterate over sampled_frames for annotation
            f_copy = f.copy()
            # Draw "Frame {i+1}" in top-left corner
            # Yellow text with black outline for visibility
            text_frame = f"Frame {i+1}"
            font_scale = 1.5
            thickness = 3
            # Outline
            cv2.putText(f_copy, text_frame, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness+2)
            # Text
            cv2.putText(f_copy, text_frame, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness)
            
            # Draw YOLO Detections for this frame
            if i < len(chunk_detections):
                dets = chunk_detections[i]
                for d in dets:
                    tid, cls, box = d
                    x1, y1, x2, y2 = map(int, box)
                    # Draw Box
                    cv2.rectangle(f_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # Draw Label
                    label = f"{cls} {tid}"
                    cv2.putText(f_copy, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            annotated_frames.append(f_copy)
            
            # Save for user to see if debug is enabled
            if args.debug:
                cv2.imwrite(os.path.join(debug_dir, f"chunk_{chunk_idx+1}_frame_{i+1}.jpg"), f_copy)
            
        # Pass fps=2.0 to indicate density (not used in API call directly but good for context if needed)
        desc = analyze_chunk(annotated_frames, chunk_detections, fps_val=2.0)
        print(f"[Chunk {chunk_idx+1} Description]:\n{desc}")
        
        # Try to parse JSON from the description
        # Prepare full tracking data for Stage 2 (ALL frames)
        # yolo_results is a list of Results objects for each frame
        full_tracking_data = [] # List of lists of dets
        names = yolo.names
        
        h, w, _ = frames[0].shape
        min_w = w / 15.0
        min_h = h / 15.0

        for res in yolo_results:
            frame_dets = []
            if res.boxes and res.boxes.id is not None:
                boxes = res.boxes.xyxy.cpu().numpy()
                ids = res.boxes.id.cpu().numpy()
                clss = res.boxes.cls.cpu().numpy()
                
                for b, tid, cls in zip(boxes, ids, clss):
                    # Same filters as before
                    if int(cls) not in [0, 1, 2, 3, 5, 7]: continue
                    x1, y1, x2, y2 = b
                    # No bonnet filter
                    if (x2 - x1) < min_w or (y2 - y1) < min_h: continue
                    
                    # Save as lists for JSON serialization: [id, class_name, [x1,y1,x2,y2]]
                    frame_dets.append({
                        "id": int(tid),
                        "class": names[int(cls)],
                        "box": [float(x1), float(y1), float(x2), float(y2)]
                    })
            full_tracking_data.append(frame_dets)

        # ... (sampling logic for LLM remains the same) ...
        
        try:
            # removing code blocks if present
            clean_desc = desc.replace("```json", "").replace("```", "").strip()
            parsed_json = json.loads(clean_desc)
            # merging with chunk info
            chunk_data = {
                "chunk_id": chunk_idx + 1,
                "start_frame": curr_frame,
                "end_frame": curr_frame + len(frames),
                "tracking_data": full_tracking_data, # NEW: Save full tracking data
                **parsed_json # Expand the parsed fields
            }
        except json.JSONDecodeError:
            # Fallback if model fails to output valid JSON
            chunk_data = {
                "chunk_id": chunk_idx + 1,
                "start_frame": curr_frame,
                "end_frame": curr_frame + len(frames),
                "tracking_data": full_tracking_data, # NEW: Save here too
                "description": desc,
                "error": "JSON Parse Failed"
            }
            
        analysis_results.append(chunk_data)
        
        # Save incrementally (optional, but good if long video fails)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=4, ensure_ascii=False)
        
        curr_frame += len(frames)
        chunk_idx += 1
        
    print(f"\nAnalysis saved to: {output_path}")
    cap.release()

if __name__ == "__main__":
    main()
