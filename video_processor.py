import cv2
import numpy as np
from collections import Counter
from ultralytics import YOLO

class VideoProcessor:
    def __init__(self):
        # AI Model
        self.model = YOLO("yolov8n.pt") 
        self.valid_braking_frames = 0
        self.total_braking_frames = 0
        self.obstacle_counts = Counter()

    def process_video(self, input_path, braking_start, braking_end, output_path=None, show_display=True):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error opening video file {input_path}")
            return

        # Video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Output writer
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Define Danger Zone Focus
        dz_x1 = int(width * 0.2)
        dz_x2 = int(width * 0.8)
        dz_y1 = int(height * 0.4)
        dz_y2 = int(height * 0.8) # Exclude bottom 20% (Ego vehicle)
        
        # Define Strict Center for Red Box (25% - 75%)
        strict_x1 = int(width * 0.25)
        strict_x2 = int(width * 0.75)
        
        # Track History: {track_id: {'centroid': (cx, cy), 'dist_to_center': float, 'ratio': float, 'frame_last_seen': int}}
        track_history = {}
        next_track_id = 0
        
        frame_count = 0 
        
        # Adjust Validation Window (Start 1 second earlier)
        validation_start = max(0, braking_start - 1.0)
        validation_end = braking_end

        print(f"  > Validating Window: {validation_start}s to {validation_end}s (Includes 1s Lookahead)")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Calculate current time
            current_time = frame_count / fps
            
            color = None
            validation_text = ""
            validation_color = (255, 255, 255)
            
            # Reset obstacle flags for this frame
            frame_is_valid = False
            primary_obstacle_name = ""
            max_threat_level = 0 # 0: Green, 1: Orange, 2: Red

            # Time Window Logic (Includes 1.0s Pre-Brake Lookahead)
            # Excludes last 1.0s of braking (as requested) to avoid static/settled frames
            validation_window_start = braking_start - 1.0
            validation_window_end = braking_end - 1.0 # Excludes last 1s (As requested) 
            
            # DEBUG
            if frame_count == 0:
                print(f"DEBUG: FPS={fps}, Window=[{validation_window_start:.2f}, {validation_window_end:.2f}]")
            
            is_braking_event = False

            # Only validate and draw RED borders within this specific window
            if validation_window_start <= current_time <= validation_window_end:
                is_braking_event = True
                color = (0, 0, 255) # Red Border
                self.total_braking_frames += 1  
                    
                # --- INTELLIGENT VALIDATION (Manual Tracking) ---
                # 1. Detect Objects
                results = self.model(frame, verbose=False)
                
                current_detections = [] # List of dicts

                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        detected_cls_name = self.model.names[cls_id]
                        
                        # Filter Classes
                        if cls_id in [0, 1, 2, 3, 5, 7]:
                            x1, y1, x2, y2 = box.xyxy[0]
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            
                            cx = (x1 + x2) // 2
                            cy = (y1 + y2) // 2
                            
                            # Danger Zone Filter
                            if dz_x1 < cx < dz_x2 and dz_y1 < cy < dz_y2:
                                area = (x2 - x1) * (y2 - y1)
                                ratio = area / (width * height)
                                current_detections.append({
                                    'centroid': (cx, cy),
                                    'area': area,
                                    'ratio': ratio,
                                    'cls_name': detected_cls_name,
                                    'box': (x1, y1, x2, y2)
                                })

                # 2. Match to Existing Tracks (Greedy Euclidean Match)
                active_tracks = {} # {track_id: detection_data}
                
                # Filter out stale tracks (not seen for > 5 frames)
                track_history = {k: v for k, v in track_history.items() if frame_count - v['frame_last_seen'] < 5}

                # Matches list
                matches = [] # (track_id, det_idx, dist)
                
                if current_detections and track_history:
                    for tid, tdata in track_history.items():
                        px, py = tdata['centroid']
                        for idx, det in enumerate(current_detections):
                             cx, cy = det['centroid']
                             dist = np.sqrt((px-cx)**2 + (py-cy)**2)
                             if dist < 100: # Max Match Distance
                                 matches.append((tid, idx, dist))
                    
                    # Sort by distance (Best matches first)
                    matches.sort(key=lambda x: x[2])
                
                used_tracks = set()
                used_det_indices = set()
                
                # Apply Matches
                for tid, idx, dist in matches:
                    if tid not in used_tracks and idx not in used_det_indices:
                        used_tracks.add(tid)
                        used_det_indices.add(idx)
                        
                        # Update Track
                        det = current_detections[idx]
                        curr_dist_to_center = abs(det['centroid'][0] - (width / 2))
                        prev_data = track_history[tid]
                        
                        # Logic: Centering (Lateral only check?)
                        # Technically dist_to_center < prev is centering.
                        # Strict: Must move closer by at least 1.0px (Very significant motion)
                        is_centering = curr_dist_to_center < (prev_data['dist_to_center'] - 1.0)
                        
                        # Strict Growth: Must grow by at least 3% to filter noise
                        is_growing = det['ratio'] > (prev_data['ratio'] * 1.03)
                        
                        track_history[tid] = {
                            'centroid': det['centroid'],
                            'dist_to_center': curr_dist_to_center,
                            'ratio': det['ratio'],
                            'frame_last_seen': frame_count,
                            'is_centering': is_centering,
                            'is_growing': is_growing
                        }
                        active_tracks[tid] = det

                # Create New Tracks for Unmatched Detections
                for idx, det in enumerate(current_detections):
                    if idx not in used_det_indices:
                        tid = next_track_id
                        next_track_id += 1
                        
                        curr_dist_to_center = abs(det['centroid'][0] - (width / 2))
                        track_history[tid] = {
                            'centroid': det['centroid'],
                            'dist_to_center': curr_dist_to_center,
                            'ratio': det['ratio'],
                            'frame_last_seen': frame_count,
                            'is_centering': False, 
                            'is_growing': False
                        }
                        active_tracks[tid] = det

                # 3. Process Active Tracks for Visuals
                for tid, det in active_tracks.items():
                    data = track_history[tid]
                    box = det['box']
                    x1, y1, x2, y2 = box
                    cx, cy = det['centroid']
                    detected_cls_name = det['cls_name']
                    ratio = det['ratio']
                    
                    is_central = strict_x1 < cx < strict_x2
                    is_centering = data['is_centering']
                    is_growing = data['is_growing']
                    
                    # --- COLOR LOGIC (Priority: Orange > Red) ---
                    box_color = (0, 255, 0) # Default Green
                    threat_level = 0
                    
                    # 1. Check Red Condition (Growing + Central)
                    if is_central and is_growing:
                        box_color = (0, 0, 255) # Red
                        threat_level = 2
                    
                    # 2. Check Orange Condition (Centering) - OVERRIDES Red Color
                    if is_centering:
                        box_color = (0, 165, 255) # Orange
                        threat_level = 1 # Still valid threat
                    
                    # Draw Box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)
                    
                    # Update Frame Status
                    if threat_level > 0 or (is_central and is_growing): # Valid if either condition met
                        frame_is_valid = True
                        # For display text, we stick to the color's meaning
                        if threat_level > max_threat_level: 
                            max_threat_level = threat_level
                            primary_obstacle_name = detected_cls_name
                            
                        self.obstacle_counts[detected_cls_name] += 1

                if frame_is_valid:
                    validation_text = f"BRAKING: VALID ({primary_obstacle_name})"
                    validation_color = (0, 255, 0) # Green text
                    self.valid_braking_frames += 1
                else:
                    validation_text = "BRAKING: INVALID (CLEAR)"
                    validation_color = (0, 0, 255) # Red text
                
                # self.total_braking_frames += 1 # REMOVED DUPLICATE
                
                # Draw Red Border (Braking Indicator)
                cv2.rectangle(frame, (0, 0), (width, height), color, 20)
                
                # Draw Text
                if validation_text:
                     cv2.putText(frame, validation_text, (width//2 - 200, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, validation_color, 3)

            # Show Timestamp (Debug)
            cv2.putText(frame, f"Time: {current_time:.2f}s", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Write frame to output video
            if out:
                out.write(frame)
            
            # Display frame (optional)
            if show_display:
                cv2.imshow('Braking Validation', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            frame_count += 1

        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
