import os
import sys
import glob
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from ultralytics import YOLO
import json
import argparse
import requests
from dotenv import load_dotenv

load_dotenv()

# Add repo to path for RiskyObject model
sys.path.append(os.path.join(os.getcwd(), 'risky_object'))

try:
    from models.model import RiskyObject
except ImportError:
    print("Could not import RiskyObject. Make sure you are in the workspace root and risky_object repo is cloned.")
    sys.exit(1)

# Configuration
weights_path = 'best_auc.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model Parameters
x_dim = 2048
h_dim = 256
n_frames = 100 
fps = 20

# Feature Extractor
class FeatureExtractor(torch.nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.net = models.resnet50(pretrained=True)
        self.net.fc = torch.nn.Identity()
        
    def forward(self, x):
        return self.net(x)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def extract_feature(img_pil, resnet):
    img_t = transform(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = resnet(img_t)
        feat = feat.view(1, 2048)
    return feat.cpu().numpy()

def get_risk_scores(chunk_frames, tracking_data, risky_model, resnet):
    # Logic adapted from process_video.py but using pre-computed tracking data
    
    # Pre-allocate feature arrays
    feat_tensor = np.zeros((n_frames, 31, 2048), dtype=np.float32)
    det_tensor = np.zeros((n_frames, 30, 6), dtype=np.float32)
    
    # Sample 100 frames for independent RNN model
    if len(chunk_frames) != len(tracking_data):
        print(f"Warning: Chunk frames ({len(chunk_frames)}) != Tracking data ({len(tracking_data)}). Truncating.")
        min_len = min(len(chunk_frames), len(tracking_data))
        chunk_frames = chunk_frames[:min_len]
        tracking_data = tracking_data[:min_len]
        
    indices = np.linspace(0, len(chunk_frames)-1, n_frames).astype(int)
    sampled_frames = [chunk_frames[i] for i in indices]
    sampled_tracking_data = [tracking_data[i] for i in indices]
    
    # To store max risk per ID for this chunk
    risk_summary = {} # {track_id: max_risk_score}
    
    # We also need to run YOLO tracking on the FULL chunk to maintain consistency?
    # process_video.py runs yolo.track on `frame` inside the loop over sampled_frames?
    # No, process_video.py: `results = yolo_model.track(frame, persist=True, verbose=False)[0]` inside the loop over sampled frames.
    # But `yolo.track` typically needs continuous frames for best results.
    # However, to save time we will follow process_video.py and just track on the sampled frames.
    # Reset tracker if needed? `persist=True` keeps it across calls. 
    # Since we are processing chunks independently in this function, we might want to reset or keep persist output.
    # `process_video.py` instantiates yolo once and loops.
    
    for t, (frame, frame_dets) in enumerate(zip(sampled_frames, sampled_tracking_data)):
        # Global Feature
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        feat_global = extract_feature(img_pil, resnet)
        feat_tensor[t, 0, :] = feat_global
        
        # Load YOLO Detections from JSON
        # Format: list of dicts {'id':..., 'class':..., 'box':...}
        count = 0
        h, w, _ = frame.shape
        
        for det in frame_dets:
            if count >= 30: break
            track_id = det['id']
            # box is [x1, y1, x2, y2]
            x1, y1, x2, y2 = det['box']
            x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
            
            # Extract Object Feature
            x1_c, y1_c = max(0, int(x1)), max(0, int(y1))
            x2_c, y2_c = min(w, int(x2)), min(h, int(y2))
            
            if x2_c <= x1_c or y2_c <= y1_c: continue
            
            obj_crop = frame[y1_c:y2_c, x1_c:x2_c]
            if obj_crop.size == 0: continue
            
            obj_pil = Image.fromarray(cv2.cvtColor(obj_crop, cv2.COLOR_BGR2RGB))
            obj_pil = obj_pil.resize((224, 224))
            feat_obj = extract_feature(obj_pil, resnet)
            
            feat_tensor[t, count+1, :] = feat_obj
            det_tensor[t, count, :] = [track_id, y1, x1, y2, x2, 0] # Label 0
            
            count += 1
            
    # Inference
    features = torch.Tensor(feat_tensor).to(device).unsqueeze(0)
    detection = torch.Tensor(det_tensor).to(device).unsqueeze(0)
    toa = torch.Tensor([n_frames]).to(device).unsqueeze(0)
    flow = features.clone()
    
    risky_model.eval()
    with torch.no_grad():
        _, all_outputs, _ = risky_model(features, detection, toa, flow)
        
    # Process outputs
    # all_outputs: list of 100 outputs (for each frame)
    # Each output corresponds to the objects present in that frame
    
    for t_idx in range(n_frames):
        outputs_t = all_outputs[t_idx] # list of numpy arrays for objects in this frame
        
        # We need to match with det_tensor
        obj_idx = 0
        for i in range(30):
            track_id = int(det_tensor[t_idx, i, 0])
            if track_id == 0: continue
            
            if obj_idx < len(outputs_t):
                logit = outputs_t[obj_idx]
                prob = np.exp(logit) / np.sum(np.exp(logit))
                risk_score = prob[0][1] # Probability of Risky (Class 1)
                
                # Update max risk for this ID
                if track_id not in risk_summary:
                    risk_summary[track_id] = 0.0
                if risk_score > risk_summary[track_id]:
                    risk_summary[track_id] = risk_score
                    
                obj_idx += 1
                
    return risk_summary

def judge_driver_behavior(chunk_info, risk_data):
    api_key = os.environ.get("PERPLEXITY_API_KEY")
    if not api_key:
        return "Error: PERPLEXITY_API_KEY not set."
        
    # Construct Prompt
    # risk_data is {id: score}
    # Filter for high risk?
    high_risk_objs = []
    for tid, score in risk_data.items():
        if score > 0.5: # Threshold for considering it relevant
            high_risk_objs.append(f"Object ID {tid}: Risk Score {score:.2f}")
            
    risk_text = "No high-risk objects detected by the Risk Model."
    if high_risk_objs:
        risk_text = "Risk Analysis Model Flags (Potential, Needs Verification):\n" + "\n".join(high_risk_objs)
        
    system_prompt = """# Role
You are an expert Autonomous Driving Safety Auditor.
Your goal is to detect safety-critical events and evaluate driver behavior.

# The Logic (Two-Step Process)
1. **Step 1: Event Detection**
   - Analyze the **Risk Model Flags** and **Situation Context**.
   - Did any **Safety-Critical Event** occur? (e.g., sudden braking, cut-in, near-miss, jaywalking, accident, violation).
   - If NO -> Output "event_detected": false. Stop there.
   - If YES -> Output "event_detected": true. Proceed to Step 2.

2. **Step 2: Analysis (Only if Event Detected)**
   - **Primary Cause**: Who caused the event?
     - **Ego-Fault**: Ego-vehicle was speeding, distracted, or aggressive.
     - **Other-Fault**: Another road user violated rules or created danger.
     - **Environmental**: Unavoidable external factor (weather, sudden object).
   - **Driver Rating**: How did the Ego-Vehicle driver react?
     - **Good**: Reacted safely and promptly (e.g., avoided collision, braked for pedestrian).
     - **Bad**: Failed to react, reacted late, or caused the issue.

# Output Format (JSON)
{
    "event_detected": true | false,
    "event_description": "Brief description of the event (e.g., 'Cut-in by white car') or null if none",
    "primary_cause": "Ego-Fault" | "Other-Fault" | "Environmental" | null,
    "driver_rating": "Good" | "Bad" | null,
    "reasoning": "Detailed explanation of the verdict.",
    "confirmed_risk_objects": [5]
}
"""

    user_msg = f"""
# Situation Context (TRUTH)
{chunk_info.get('situation_summary')}

# Risk Model Flags (HINTS)
{risk_text}

# Vehicle & Pedestrian Details
- Ego Vehicle: {chunk_info.get('vehicle_context')}
- Pedestrians: {chunk_info.get('pedestrian_events')}

Evaluate the driver's behavior based on the Verified Risks only.
"""

    payload = {
        "model": "sonar-pro",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg}
        ],
        "temperature": 0.2
    }
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post("https://api.perplexity.ai/chat/completions", json=payload, headers=headers)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"Error: {e}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('video_path', type=str)
    parser.add_argument('json_path', type=str, help="Path to Stage 1 JSON output")
    args = parser.parse_args()
    
    if "PERPLEXITY_API_KEY" not in os.environ:
        print("Set PERPLEXITY_API_KEY first.")
        sys.exit(1)

    # Load JSON
    with open(args.json_path, 'r') as f:
        stage1_data = json.load(f)
        
    # Load Models
    print("Loading ResNet...")
    resnet = models.resnet50(pretrained=True)
    modules = list(resnet.children())[:-1]
    resnet = torch.nn.Sequential(*modules).to(device)
    resnet.eval()
    
    print("Loading RiskyObject Model...")
    # x_dim, h_dim, n_frames, fps must match config
    risky_model = RiskyObject(2048, 256, 100, 20)
    risky_model = risky_model.to(device)
    if os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location=device)
        risky_model.load_state_dict(checkpoint['model'])
        print("Loaded RiskyObject weights.")
    else:
        print(f"Warning: {weights_path} not found.")
        sys.exit(1)
    risky_model.eval()
    
    # YOLO is now loaded from Stage 1 JSON, so no need to load model here.
    # yolo = YOLO('yolov8n.pt') 
    
    # Process Video
    cap = cv2.VideoCapture(args.video_path)
    fps_in = cap.get(cv2.CAP_PROP_FPS)
    chunk_len = int(5 * fps_in)
    
    final_assessments = []
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(frame)
    cap.release()
    
    print(f"Total Frames: {len(frames)}")
    
    # Iterate chunks based on Stage 1 data to align?
    # Or just iterate by index and match?
    # Stage 1 data is a list of dicts. We assume order is Chunk 1, Chunk 2...
    
    chunk_idx = 0
    for i in range(0, len(frames), chunk_len):
        if chunk_idx >= len(stage1_data): break
        
        chunk_frames = frames[i:i+chunk_len]
        if len(chunk_frames) < 10: continue
        
        print(f"\nScanning Chunk {chunk_idx+1} for Risk...")
        
        # 1. Get Risk Scores
        chunk_context = stage1_data[chunk_idx]
        tracking_data = chunk_context.get('tracking_data', [])
        
        if not tracking_data:
             print("  Warning: No tracking data found in Stage 1 JSON. Skipping Risk Analysis.")
             risk_data = {}
        else:
             risk_data = get_risk_scores(chunk_frames, tracking_data, risky_model, resnet)
             
        print(f"  Risk Data: {risk_data}")
        
        # 2. Judge
        print(f"  Context Summary: {chunk_context.get('situation_summary')[:100]}...")
        
        judgement = judge_driver_behavior(chunk_context, risk_data)
        print(f"  Judgement: {judgement}")
        
        # Parse JSON if possible
        try:
            clean_j = judgement.replace("```json", "").replace("```", "").strip()
            j_dict = json.loads(clean_j)
            chunk_context['safety_assessment'] = j_dict
        except:
            chunk_context['safety_assessment'] = {"raw_output": judgement}
            
        # Remove tracking_data to keep final output clean and small
        if 'tracking_data' in chunk_context:
            del chunk_context['tracking_data']
            
        final_assessments.append(chunk_context)
        chunk_idx += 1
        
    # Save Output
    output_dir = "result"
    os.makedirs(output_dir, exist_ok=True)
    
    video_basename = os.path.basename(args.video_path)
    video_name_no_ext = os.path.splitext(video_basename)[0]
    out_name = os.path.join(output_dir, f"{video_name_no_ext}.json")
    
    with open(out_name, 'w', encoding='utf-8') as f:
        json.dump(final_assessments, f, indent=4, ensure_ascii=False)
        
    print(f"\nAssessment Complete. Saved to {out_name}")

if __name__ == "__main__":
    main()
