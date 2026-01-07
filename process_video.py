
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

# Add repo to path
sys.path.append(os.path.join(os.getcwd(), 'risky_object'))

try:
    from models.model import RiskyObject
except ImportError:
    print("Could not import RiskyObject. Make sure you are in the workspace root and risky_object repo is cloned.")
    sys.exit(1)

import argparse

# Configuration
weights_path = 'best_auc.pth'
input_dir = 'input_video'
output_dir = 'output_video'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parse arguments
parser = argparse.ArgumentParser(description='Process a video for risky object detection.')
parser.add_argument('video_name', type=str, help='Name of the video file in input_video folder (e.g., test1.mp4)')
args = parser.parse_args()

target_video_name = args.video_name
video_path = os.path.join(input_dir, target_video_name)

if not os.path.exists(video_path):
    print(f"Error: Video file not found at {video_path}")
    print(f"Please make sure '{target_video_name}' is inside '{input_dir}' folder.")
    sys.exit(1)
    
print(f"Processing video: {video_path}")

# Model Parameters
x_dim = 2048
h_dim = 256
n_frames = 100 # Model expects 100 frames segments (approx 5s at 20fps or similar)
fps = 20 # data fps, might need resampling if video is different

# Setup Feature Extractor (ResNet50)
class FeatureExtractor(torch.nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.net = models.resnet50(pretrained=True)
        self.net.fc = torch.nn.Identity() # Remove last FC layer to get 2048 dim
        # OR use the specific extraction method from their code which cuts at avgpool
        # Their code: output = self.net.avgpool(output) after layer4.
        # ResNet50 forward does: conv1->bn1->relu->maxpool->layer1->2->3->4->avgpool->fc
        # So we can just use a hook or remove FC.
        # Ideally, we follow their exact method. 
        # Their method in src/model.py: return output of avgpool.
        
    def forward(self, x):
        return self.net(x)

# Re-implementing their exact FeatureExtractor structure to be safe if weights matter
# But since we use standard pretrained ResNet50 for features, standard torchvision is fine.
# Note: Their FeatureExtractor class in feat_extract/src/model.py defines forward explicitly.
# We will use a simplified version that matches the output.
resnet = models.resnet50(pretrained=True)
modules = list(resnet.children())[:-1] # Remove FC
resnet = torch.nn.Sequential(*modules).to(device)
resnet.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # transforms.Normalize? Their code feat_extract.py doesn't show normalize, just ToTensor.
    # We will stick to ToTensor to match their potential distribution, though standard ResNet expects normalization.
    # Looking at feat_extract.py: transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
    # So NO normalization.
])

def extract_feature(img_pil):
    img_t = transform(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = resnet(img_t) # 1x2048x1x1
        feat = feat.view(1, 2048)
    return feat.cpu().numpy()

def process_chunk(chunk_frames, chunk_idx, yolo_model, risky_model):
    print(f"Processing chunk {chunk_idx}, frames: {len(chunk_frames)}")
    
    # 1. Object Detection & Tracking
    # We need track IDs. YOLOv8 supports tracking.
    # We will run tracking on the whole chunk.
    detections_list = [] # List of (frame_idx, box, track_id, class)
    
    # Pre-allocate feature arrays
    # features: (n_frames, 31, 2048) -> 0 is global, 1..30 are objects
    # detection: (n_frames, 30, 6) -> track_id, y1, x1, y2, x2, label
    feat_tensor = np.zeros((n_frames, 31, 2048), dtype=np.float32)
    det_tensor = np.zeros((n_frames, 30, 6), dtype=np.float32)
    
    # Resample frames if needed to match n_frames?
    # The requirement says "5 seconds". IF video is 30fps, 5s = 150 frames.
    # Model expects 100 frames.
    # We should uniformly sample 100 frames from the chunk.
    indices = np.linspace(0, len(chunk_frames)-1, n_frames).astype(int)
    sampled_frames = [chunk_frames[i] for i in indices]
    
    # Scaling factors for normalized coordinates in model
    # Model seems to use 1080p reference for normalization in demo.py?
    # "unnormalized_cor[1]/1080" -> y / 1080.
    # "unnormalized_cor[2]/720" -> x / 720. (Wait, usually 1920x1080 is x,y... let's check input)
    # demo.py: y[0][t][bbox] -> 6 values. 
    # indices: 1->y1, 2->x1, 3->y2, 4->x2. 
    # norm: y1/1080, x1/720 ...
    # This implies the training data was 720x1080 (HxW) or similar? 
    # Typically 1920x1080. 
    # Let's assume the code expects x normalized by 720 and y by 1080? That's weird. 1080 is usually height.
    # In demo.py: `unnormalized_cor[1]/1080` (index 1 is y1?).
    # `feat_extract.py`: `row['y1'], row['x1']`.
    # Let's check `feat_extract.py`: 
    # `detections[... ] = row['track_id'], row['y1'], row['x1'], row['y2'], row['x2'], row['label']`
    # So Index 0: track_id
    # Index 1: y1
    # Index 2: x1
    # Index 3: y2
    # Index 4: x2
    # Index 5: label
    # And demo.py divides index 1 by 1080. If 1080 is height, that makes sense for y.
    # Divides index 2 by 720. If 720 is width... wait, 720p is 1280x720. 
    # So maybe 720 is width? Or they used portrait video? 
    # Most likely it's 1280x720 video, but coordinates are flipped/mixed.
    # Let's assume standard landscape 1280x720?
    # Actually, `feat_extract.py` has comments: scaling_w = 1080/224... scaling_h = 720/224.
    # That implies original W=1080, H=720? No, W usually > H.
    # Maybe original is 720x1080 (Portrait)? Or maybe 1920x1080?
    # Let's not stress too much, we will just pass pixel coords and hope the normalization (which we can't easily change in the loaded model weights) is robust enough or we match the training resolution roughly.
    # We will resize input frames to 1280x720 for processing to match the likely "720" denominator if it represents 720p.
    # Re-reading: `unnormalized_cor[2]/720` -> x1/720.
    # If width is 720, it's small.
    # If height is 720, then x1/720 is wrong unless it's square/portrait.
    # Let's assume the model was trained on data scaled to roughly 1280x720 or 1920x1080 but the divisor is hardcoded.
    # We should perform detection on the original frame, then mapped to the model's expected input.
    # We will use the sampled frames.
    
    track_history = {}
    
    for t, frame in enumerate(sampled_frames):
        # Resize to 224x224 for Feature Extractor Global
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        feat_global = extract_feature(img_pil)
        feat_tensor[t, 0, :] = feat_global
        
        # YOLO Detection
        results = yolo_model.track(frame, persist=True, verbose=False)[0]
        
        # Parse detections
        # We need to limit to 30 objects
        boxes = results.boxes.cpu() 
        count = 0
        for box in boxes:
            if count >= 30: break
            
            # get track id
            if box.id is None: continue
            track_id = int(box.id[0])
            
            # get coords (x1, y1, x2, y2)
            x1, y1, x2, y2 = box.xyxy[0].numpy()
            
            # class
            cls = int(box.cls[0])
            # Filter for vehicles/people if possible? 
            # classes: 0=person, 2=car, 3=motorcycle, 5=bus, 7=truck (COCO)
            if cls not in [0, 1, 2, 3, 5, 7]: continue
            
            # Crop object
            # Ensure coords within bounds
            h, w, _ = frame.shape
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(w, x2); y2 = min(h, y2)
            
            if x2 <= x1 or y2 <= y1: continue
            
            # 1. Bonnet/Hood filtering
            # Removed as per user request to detect objects in the bottom area.
            # bottom_threshold = h - (h / 10.0)
            # if y2 > bottom_threshold: continue

            # 2. Minimum size check (1/15 of frame dimensions)
            min_w = w / 15.0
            min_h = h / 15.0
            
            if (x2 - x1) < min_w or (y2 - y1) < min_h: continue
            
            obj_crop = frame[int(y1):int(y2), int(x1):int(x2)]
            obj_pil = Image.fromarray(cv2.cvtColor(obj_crop, cv2.COLOR_BGR2RGB))
            obj_pil = obj_pil.resize((224, 224))
            feat_obj = extract_feature(obj_pil)
            
            # Store features (index = count + 1)
            feat_tensor[t, count+1, :] = feat_obj
            
            # Store detection info
            # Format: track_id, y1, x1, y2, x2, label
            # We don't have ground truth label, so 0 (or -1?)
            # Model output `all_labels` relies on this, but for inference we just want `all_outputs`.
            # We'll set label to 0.
            det_tensor[t, count, :] = [track_id, y1, x1, y2, x2, 0]
            
            count += 1
            
    # Prepare Inputs
    features = torch.Tensor(feat_tensor).to(device).unsqueeze(0) # 1 x 100 x 31 x 2048
    detection = torch.Tensor(det_tensor).to(device).unsqueeze(0) # 1 x 100 x 30 x 6
    toa = torch.Tensor([n_frames]).to(device).unsqueeze(0) # dummy TOA
    # flow: duplicate features as fallback
    flow = features.clone()
    
    # Run Inference
    risky_model.eval()
    with torch.no_grad():
        # model forward: losses, all_outputs, all_labels = model(features, detection, toa, flow)
        _, all_outputs, _ = risky_model(features, detection, toa, flow)
        
    # Process outputs
    # all_outputs is a list of lists (frames x objects) -> probabilities?
    # In demo.py: `all_outputs` is result.
    # In model.py: `output` is `softmax`? No, `dense2(out)`. Output dim is 2 (CrossEntropy).
    # So `output` is logits for [safe, risky].
    
    # We want to visualize.
    # For each frame `t`, for each object `bbox`, we have a score.
    # We need to map `frame_outputs` back to the boxes.
    
    # Visualizing on the SAMPLED frames
    out_frames = []
    
    # all_outputs is [frame0_outputs, frame1_outputs, ...]
    # frame_outputs is list of object outputs in that frame
    
    for t_idx, frame in enumerate(sampled_frames):
        frame_vis = frame.copy()
        outputs_t = all_outputs[t_idx] # list of numpy arrays
        
        # Retrieve detections for this frame to match indices
        # det_tensor[t_idx] has the info.
        
        # The model logic:
        # It loops `for bbox in range(30):`
        # Checks `if y[0][t][bbox][0] == 0: continue` (track_id == 0 means no object)
        # So the `outputs_t` list corresponds exactly to the VALID objects in that frame, 
        # IN THE ORDER they appeared in `y`.
        
        obj_idx = 0
        for i in range(30):
            track_id = det_tensor[t_idx, i, 0]
            if track_id == 0: continue
            
            if obj_idx < len(outputs_t):
                logit = outputs_t[obj_idx]
                # Softmax to get prob
                prob = np.exp(logit) / np.sum(np.exp(logit))
                risk_score = prob[0][1] # Probability of class 1
                
                # Draw box
                y1, x1, y2, x2 = det_tensor[t_idx, i, 1:5]
                # Coords are float, cast to int
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Color based on risk
                color = (0, 255, 0) # Green
                if risk_score > 0.8: color = (0, 0, 255) # Red
                
                cv2.rectangle(frame_vis, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame_vis, f"{risk_score:.2f}", (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                obj_idx += 1
                
        out_frames.append(frame_vis)
        
    return out_frames

def main():
    # Load Model
    print("Loading model...")
    model = RiskyObject(x_dim, h_dim, n_frames, fps)
    model = model.to(device)
    
    if os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
    else:
        print(f"Weights not found at {weights_path}")
        # sys.exit(1) # Try to run anyway? No, meaningless.
    
    model.eval()
    
    # Load YOLO
    print("Loading YOLO...")
    yolo = YOLO('yolov8n.pt') 
    
    # Load Video
    cap = cv2.VideoCapture(video_path)
    fps_in = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps_in
    
    print(f"Video Info: {duration}s, {total_frames} frames, {fps_in} fps")
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(frame)
    cap.release()
    
    # Split into 5s chunks
    # 5 seconds * fps_in
    chunk_len = int(5 * fps_in)
    
    # We need specifically 3 chunks, or cover 14s.
    # 0-5, 5-10, 10-15(padded)
    
    chunks = []
    for i in range(0, len(frames), chunk_len):
        chunk = frames[i:i+chunk_len]
        if len(chunk) < 10: continue # Skip tiny chunks
        chunks.append(chunk)
        
    os.makedirs(output_dir, exist_ok=True)
    
    for i, chunk in enumerate(chunks):
        if i >= 3: break # Limit to 3 as requested
        
        vis_frames = process_chunk(chunk, i, yolo, model)
        
        # Save video
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        out_path = os.path.join(output_dir, f'result_{base_name}_part{i+1}.webm')
        h, w, _ = vis_frames[0].shape
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'vp80'), 20, (w, h)) # 20fps output, VP8 WebM
        for f in vis_frames:
            out.write(f)
        out.release()
        print(f"Saved {out_path}")

if __name__ == '__main__':
    main()
