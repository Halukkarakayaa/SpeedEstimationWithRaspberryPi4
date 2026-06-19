import cv2
import time
from ultralytics import YOLO
import math
import numpy as np
from collections import deque
import json
from datetime import datetime

# CONFIGURATION 

# --- CAMERA I/O SETTINGS ---
CAMERA_INDEX = 0       # Input camera index
SAVE_OUTPUT_VIDEO = True                 # Flag to save the processed output video
HEADLESS_MODE = False                   # Flag to run without GUI display
TARGET_VIDEO_PATH = "pc_camera_record.mp4" # Output video file path
INFER_SIZE     = 640      # YOLOv8 input size
CONF_THRESHOLD = 0.4     # Minimum confidence score for detection
DEVICE         = "cpu"    # Target device for inference

VEHICLE_CLASSES = [2, 3, 5, 7]
WINDOW_TITLE = "Vehicle Detection — Video Mode"
DETECT_EVERY_N = 1   


# ───────────────────────────────
# SPEED ESTIMATION CONFIGURATION 
# ────────────────────────────────
# Pixel coordinates of the reference area in the video frame
SOURCE_PTS = np.array([
    [163,552],   # Sol üst
    [279,355],   # Sağ üst
    [1166,400],   # Sağ alt 
    [1256,599]     # Sol alt
], dtype=np.float32)

# Real-world metric coordinates of the reference area
DEST_PTS = np.array([
    [0, 0],       # Sol üst 
    [12, 0],      # Sağ üst 
    [12, 40],     # Sağ alt
    [0, 40]       # Sol alt 
], dtype=np.float32)


# Calculate the Homography matrix for perspective transformation
MATRIX = cv2.getPerspectiveTransform(SOURCE_PTS, DEST_PTS)

# ─────────────────────────────────────────────
# MEMORY & TRACKING VARIABLES
# ─────────────────────────────────────────────

vehicle_history = {} # Stores timestamp and metric coordinates for speed calc
vehicle_speeds = {}  # Stores the latest calculated speed per vehicle ID

# ─────────────────────────────────────────────
# FILTERING & TRACKING CONFIGURATION
# ─────────────────────────────────────────────
EMA_ALPHA = 0.6  # Exponential Moving Average weight for bounding boxes
ema_states = {}  # Stores smoothed bounding box coordinates
smoothed_boxes_to_draw = [] # Temporary list for rendering the current frame

lost_tracks = {}          # Counts how many frames a track ID has been missing
MAX_LOST_FRAMES = 15      # Maximum allowed missing frames before deleting ID

# Analytics and JSON Logging Variables
traffic_data_log = []
counted_vehicle_ids = set() # Unique set to count cumulative vehicles
last_log_time = time.time()

# Smart 3-State Machine Variables (Moving, Stopped, Parked)
stationary_frames = {} # Counts consecutive frames a vehicle's speed is near zero
locked_classes = {} # Locks the vehicle class (prevents Car <-> Truck switch)
frozen_boxes = {} # Stores locked bounding box coordinates for parked vehicles

CLASS_COLORS = {
    2: (255, 200, 0),     # (Car) -> Mavi
    3: (0, 255, 0),     # (Motorcycle) -> Yeşil
    5: (255, 0, 255),   # (Bus) -> Pembe / Macenta
    7: (0, 255, 255)    # (Truck) -> Sarı
}
# ─────────────
# LOAD MODEL 
# ─────────────
print("[INFO] Loading YOLOv8n model...")
model = YOLO("yolov8n.pt")
print("[INFO] Model loaded. Opening video file...")

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    raise RuntimeError(f"[ERROR] Could not open video file: {CAMERA_INDEX}")
# Extract video properties
video_fps = cap.get(cv2.CAP_PROP_FPS)
if video_fps == 0 or math.isnan(video_fps):
     video_fps = 2 # Default fallback FPS

video_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


scale_factor = video_height / 720.0

print(f"[INFO] Video loaded: {video_width}x{video_height} @ {video_fps} FPS")

# Initialize VideoWriter if saving is enabled
out = None
if SAVE_OUTPUT_VIDEO:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(TARGET_VIDEO_PATH, fourcc, video_fps, (video_width, video_height))
    print(f"[INFO] Output will be saved to: {TARGET_VIDEO_PATH}")

# Performance metrics
processing_fps = 0.0 
processed_frame_count = 0
fps_update_interval = 0.5 
last_fps_time = time.time()
frame_counter = 0

# COLOUR & FONT CONSTANTS
COLOR_BOX   = (0, 255, 0)
COLOR_LABEL = (0, 255, 0)
COLOR_FPS   = (0, 200, 255)
COLOR_BG    = (0, 0, 0)
FONT        = cv2.FONT_HERSHEY_SIMPLEX
BOX_THICKNESS = max(2, int(2 * scale_factor))

# Helper function for drawing formatted UI labels
def draw_label(frame, text, x, y, base_scale=0.55, color=COLOR_LABEL, base_thickness=1):
    dynamic_scale = base_scale * scale_factor
    dynamic_thickness = max(1, int(base_thickness * scale_factor))
    
    (tw, th), baseline = cv2.getTextSize(text, FONT, dynamic_scale, dynamic_thickness)
    cv2.rectangle(frame, (x, y - th - baseline - int(2 * scale_factor)), (x + tw + int(4 * scale_factor), y + baseline), COLOR_BG, cv2.FILLED)
    cv2.putText(frame, text, (x + int(2 * scale_factor), y - int(2 * scale_factor)), FONT, dynamic_scale, color, dynamic_thickness, cv2.LINE_AA)

if not HEADLESS_MODE:
    print("[INFO] Processing started. Press 'q' to quit.")
    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
else:
    print("[INFO] Headless Mode Active. GUI is disabled for performance.")
    print("[INFO] Processing video in background. Press CTRL+C in terminal to stop.")

# ─────────────────────────────────────────────
# MAIN PROCESSING LOOP
# ─────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        print("[INFO]  Camera stream ended or disconnected.")
        break 

    frame_counter += 1

    # Virtual time within the video.
    current_video_time = time.time()

    if frame_counter % DETECT_EVERY_N == 0:
        
        # Run YOLO detection and ByteTrack tracking
        results = model.track(
            source=frame, imgsz=INFER_SIZE, conf=CONF_THRESHOLD,
            classes=VEHICLE_CLASSES, verbose=False, device=DEVICE,
            tracker="bytetrack.yaml", persist=True
        )
        last_result = results[0]
        smoothed_boxes_to_draw = [] 

        if last_result.boxes is not None and last_result.boxes.id is not None:
            boxes = last_result.boxes.xyxy.cpu().numpy()
            confs = last_result.boxes.conf.cpu().numpy()
            clss = last_result.boxes.cls.cpu().numpy()
            track_ids = last_result.boxes.id.int().cpu().numpy()

            current_ids_in_frame = set()

            for box, conf, cls, track_id in zip(boxes, confs, clss, track_ids):
                current_ids_in_frame.add(track_id)
                x1, y1, x2, y2 = box

                # CLASS LOCKING
                if track_id not in locked_classes:
                    locked_classes[track_id] = int(cls)
                final_cls_id = locked_classes[track_id] 

                # EMA BOUNDING BOX SMOOTHING
                if track_id in ema_states:
                    ex1, ey1, ex2, ey2 = ema_states[track_id]
                    nx1 = EMA_ALPHA * x1 + (1 - EMA_ALPHA) * ex1
                    ny1 = EMA_ALPHA * y1 + (1 - EMA_ALPHA) * ey1
                    nx2 = EMA_ALPHA * x2 + (1 - EMA_ALPHA) * ex2
                    ny2 = EMA_ALPHA * y2 + (1 - EMA_ALPHA) * ey2
                    ema_states[track_id] = [nx1, ny1, nx2, ny2]
                else:
                    ema_states[track_id] = [x1, y1, x2, y2]

                smooth_x1, smooth_y1, smooth_x2, smooth_y2 = ema_states[track_id]
                bottom_center_x = (smooth_x1 + smooth_x2) / 2
                bottom_center_y = smooth_y2

                # Reset missing frame counter since object is currently visible
                lost_tracks[track_id] = 0
                
                # ROI FILTERING
                # Process speed only if the vehicle's bottom center is inside the ROI polygon
                is_inside = cv2.pointPolygonTest(SOURCE_PTS, (float(bottom_center_x), float(bottom_center_y)), False)

                if is_inside >= 0:
                    # PERSPECTIVE TRANSFORM
                    # Convert 2D pixel coordinates to real-world metric coordinates
                    pt = np.array([[[bottom_center_x, bottom_center_y]]], dtype=np.float32)
                    transformed_pt = cv2.perspectiveTransform(pt, MATRIX)
                    real_x, real_y = transformed_pt[0][0] 

                    
                    if track_id not in vehicle_history:
                        # Expand history buffer to 45 frames for stable EDS filtering
                        vehicle_history[track_id] = deque(maxlen=45) 
                        vehicle_speeds[track_id] = 0.0

                    # Append current metric position and timestamp to history
                    vehicle_history[track_id].append((current_video_time, real_x, real_y))
                    history = vehicle_history[track_id]

                    # Default calculated speed to prevent NameError on initial frames
                    calculated_speed = 0.0

                    # SPEED ESTIMATION 
                    # Wait at least 3 frames before calculating initial speed
                    if len(history) > 3:
                        old_video_time, old_x, old_y = history[0]
                        time_diff = current_video_time - old_video_time 
                        
                        # Prevent division by zero
                        if time_diff > 0.1:
                            dist_meters = math.sqrt((real_x - old_x)**2 + (real_y - old_y)**2)
                            
                            speed_ms = dist_meters / time_diff
                            calculated_speed = speed_ms * 3.6
                            
                            # EDS Filter Rule: Speeds under 5 km/h are considered jitter/noise
                            if calculated_speed < 5.0:
                                vehicle_speeds[track_id] = 0.0
                            else:
                                prev_speed = vehicle_speeds[track_id]
                                if prev_speed == 0.0:
                                    vehicle_speeds[track_id] = calculated_speed
                                else:
                                    # Apply 50% EMA to smooth speed transitions
                                    vehicle_speeds[track_id] = (prev_speed * 0.5) + (calculated_speed * 0.5)
                    
                    current_speed = vehicle_speeds[track_id]
                    # --- SMART STATIONARY DETECTION (3-STATE MACHINE) ---
                    # Detect micro-movements (< 2.0 km/h) to instantly wake up parked vehicles
                    if calculated_speed < 2.0: 
                        stationary_frames[track_id] = stationary_frames.get(track_id, 0) + 1
                    else:
                        stationary_frames[track_id] = 0 # Reset counter if vehicle moves
                        
            
                    stationary_count = stationary_frames.get(track_id, 0)
                    is_stopped = stationary_count > 30  # Stopped in traffic
                    is_parked = stationary_count > 120 # Completely parked
                    counted_vehicle_ids.add(track_id) # Add to unique cumulative counter
                    
                    # BOX FREEZING LOGIC
                    if is_parked:
                        # Lock the bounding box coordinates to eliminate visual jitter
                        if track_id not in frozen_boxes:
                            frozen_boxes[track_id] = ema_states[track_id] 
                        final_box = frozen_boxes[track_id] 
                    else:
                        # Release the lock if the vehicle starts moving again
                        if track_id in frozen_boxes:
                            del frozen_boxes[track_id] 
                        final_box = ema_states[track_id] 
                    
                    # Append final processed data for the drawing function
                    smoothed_boxes_to_draw.append((final_box, track_id, final_cls_id, float(conf), current_speed, is_stopped, is_parked))
                    
                
                else:
                    # --- OUTSIDE ROI ---
                    # Clear history to avoid sudden speed spikes upon re-entry
                    if track_id in vehicle_history:
                        del vehicle_history[track_id]
                        if track_id in vehicle_speeds:
                            del vehicle_speeds[track_id]

            # --- MEMORY CLEANUP ---
            # Remove stale tracking IDs and sub-dictionaries to prevent RAM memory leaks
            keys_to_update = [tid for tid in ema_states if tid not in current_ids_in_frame]
            for tid in keys_to_update:
                lost_tracks[tid] = lost_tracks.get(tid, 0) + 1
                
                if lost_tracks[tid] > MAX_LOST_FRAMES:
                    del ema_states[tid]
                    if tid in vehicle_history: del vehicle_history[tid]
                    if tid in vehicle_speeds: del vehicle_speeds[tid]
                    if tid in stationary_frames: del stationary_frames[tid]
                    if tid in locked_classes: del locked_classes[tid]
                    if tid in frozen_boxes: del frozen_boxes[tid]
                    del lost_tracks[tid]
        else:
            ema_states.clear()

    # ─────────────────────────────────────────────
    # VISUALIZATION & UI DASHBOARD
    # ─────────────────────────────────────────────
    
    # Draw the Region of Interest (ROI) Polygon
    pts = SOURCE_PTS.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

    # Render bounding boxes and status labels based on the 3-State Machine
    for s_box, track_id, cls_id, confidence, speed, is_stopped, is_parked in smoothed_boxes_to_draw:
        x1, y1, x2, y2 = map(int, s_box)
        
        box_color = CLASS_COLORS.get(cls_id, (255, 255, 255))
        
        if is_parked:
            # State 3: Parked
            gray_color = (150, 150, 150)
            cv2.rectangle(frame, (x1, y1), (x2, y2), gray_color, BOX_THICKNESS)
            draw_label(frame, f"#{track_id} | Parked", x1, y2 + int(20 * scale_factor), base_scale=0.5, color=gray_color)
        elif is_stopped:
            # State 2: Stopped
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, BOX_THICKNESS)
            # Trafikte durduğunu belli etmek için yazıyı turuncu/kırmızımsı verebiliriz
            draw_label(frame, f"#{track_id} | Stopped", x1, y2 + int(20 * scale_factor), base_scale=0.5, color=(0, 100, 255))
        else:
            # State 1: Moving
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, BOX_THICKNESS)
            draw_label(frame, f"#{track_id} | {speed:.1f} km/h", x1, y2 + int(20 * scale_factor), base_scale=0.5, color=box_color)

    # Calculate and render Processing FPS
    processed_frame_count += 1
    now = time.time()
    elapsed = now - last_fps_time

    if elapsed >= fps_update_interval:
        processing_fps = processed_frame_count / elapsed
        processed_frame_count = 0
        last_fps_time = now

    draw_label(frame, f"Processing FPS: {processing_fps:.1f}", x=int(10*scale_factor), y=int(30*scale_factor), base_scale=0.8, color=COLOR_FPS, base_thickness=2)
    
   # TRAFFIC ANALYTICS & CONGESTION LOGIC
    active_vehicles = len(smoothed_boxes_to_draw) # All visible vehicles in ROI
    
    # Calculate average speed ONLY using active traffic (Moving + Stopped, excluding Parked)
    active_traffic_speeds = [s for _, _, _, _, s, is_stopped, is_parked in smoothed_boxes_to_draw if not is_parked]
    active_traffic_count = len(active_traffic_speeds)
    current_avg_speed = sum(active_traffic_speeds) / active_traffic_count if active_traffic_count > 0 else 0.0

    # UI Dashboard Rendering
    draw_label(frame, f"Total Vehicles in area: {active_vehicles}", x=int(10*scale_factor), y=int(65*scale_factor), base_scale=0.6, color=(255, 220, 0), base_thickness=1)
    draw_label(frame, f"Vehicles in traffic: {active_traffic_count}", x=int(10*scale_factor), y=int(95*scale_factor), base_scale=0.6, color=(0, 255, 0), base_thickness=1)
    draw_label(frame, f"Total Passed Vehicles: {len(counted_vehicle_ids)}", x=int(10*scale_factor), y=int(125*scale_factor), base_scale=0.6, color=(0, 255, 255), base_thickness=1)
    draw_label(frame, f"Average Speed: {current_avg_speed:.1f} km/h", x=int(10*scale_factor), y=int(155*scale_factor), base_scale=0.6, color=(255, 255, 255), base_thickness=1)
    
    # Trigger Congestion Alert if average speed drops below 10 km/h
    congestion_status = False
    if current_avg_speed < 10.0 and active_traffic_count > 2:
        congestion_status = True
        draw_label(frame, "!!! TRAFFIC CONGESTION !!!", x=int(10*scale_factor), y=int(185*scale_factor), base_scale=0.7, color=(0, 0, 255), base_thickness=2)

    # JSON LOGGING (Big Data Node Generation)
    current_time = time.time()
    if current_time - last_log_time >= 1.0:
        log_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_cumulative_vehicles": len(counted_vehicle_ids),
            "current_density": active_vehicles,
            "average_corridor_speed_kmh": round(current_avg_speed, 2),
            "congestion_alert": congestion_status
        }
        
        # Print to terminal for live monitoring
        print(f"[JSON LOG] {log_entry}")
        
        # Append to array and save to JSON file
        traffic_data_log.append(log_entry)
        with open("traffic_log.json", "w", encoding="utf-8") as f:
            json.dump(traffic_data_log, f, indent=4)
            
        last_log_time = current_time

    # Save output frame and display UI
    if SAVE_OUTPUT_VIDEO and out is not None:
        out.write(frame)

    # Show GUI only if Headless Mode is disabled
    if not HEADLESS_MODE:
        cv2.imshow(WINDOW_TITLE, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[INFO] Processing stopped by user (Q pressed).")
            break

# ─────────────────────────────────────────────
# CLEANUP
# ─────────────────────────────────────────────
cap.release()

if SAVE_OUTPUT_VIDEO and out is not None:
    out.release()
cv2.destroyAllWindows()
print("[INFO] Processing completed successfully. Files closed.")