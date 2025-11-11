from flask import Flask, render_template, Response, jsonify, request, redirect, url_for, session as flask_session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
import cv2
import numpy as np
from ultralytics import YOLO
import time
from collections import defaultdict, deque
import json
import os
import atexit

# ============================================================================
# Flask App Configuration
# ============================================================================
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:Mass2712@localhost:5432/engagetrack'
app.config['SECRET_KEY'] = 'your-secret-key-change-this' 
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# ============================================================================
# Database Models
# ============================================================================
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Session(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    start_time = db.Column(db.DateTime, default=datetime.utcnow)
    end_time = db.Column(db.DateTime)
    duration = db.Column(db.Integer)
    total_students = db.Column(db.Integer, default=0)
    engaged_percentage = db.Column(db.Float, default=0.0)
    neutral_percentage = db.Column(db.Float, default=0.0)
    bored_percentage = db.Column(db.Float, default=0.0)

class Detection(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('session.id'), nullable=False)
    student_id = db.Column(db.Integer, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    engagement_state = db.Column(db.String(50))
    confidence = db.Column(db.Float)
    image_path = db.Column(db.String(255))

# ============================================================================
# Global Variables
# ============================================================================
camera = None
model = None
pose_model = None  # YOLO11-pose model
detection_active = False
pose_detection_active = True  # Toggle for pose detection
current_session_id = None
student_trackers = {}
next_student_id = 1
engagement_history = defaultdict(lambda: deque(maxlen=30))
# Alert tracking: stores previous state per student to detect transitions
previous_states = {}
active_alerts = []  # List of current alerts with timestamp
alert_cooldown = {}  # Prevent alert spam for same student

# ============================================================================
# Model Loading
# ============================================================================
def load_model():
    """Load YOLO engagement detection model"""
    global model
    try:
        model_files = ['best.onnx', 'engagement_detection_model.onnx', 'engagement_detection_model.pt']
        
        for model_file in model_files:
            if os.path.exists(model_file):
                model = YOLO(model_file, task='detect')
                print(f"✓ Loaded {model_file} successfully")
                print(f"Model classes: {model.names}")
                return True
        
        print("✗ No model file found!")
        return False
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return False

def load_pose_model():
    """Load YOLO11-pose model for keypoint detection"""
    global pose_model
    try:
        if os.path.exists('yolo11n-pose.pt'):
            pose_model = YOLO('yolo11n-pose.pt', task='pose')
            print(f"✓ Loaded yolo11n-pose.pt successfully")
            return True
        else:
            print("⚠ yolo11n-pose.pt not found, downloading...")
            pose_model = YOLO('yolo11n-pose.pt')  # Will auto-download
            print(f"✓ Downloaded and loaded yolo11n-pose.pt")
            return True
    except Exception as e:
        print(f"✗ Error loading pose model: {e}")
        return False

# Initialize models on startup
load_model()
load_pose_model()

# ============================================================================
# Helper Functions
# ============================================================================
def get_engagement_state(class_name):
    """Map YOLO class names to engagement states"""
    engagement_map = {
        'engaged': 'engaged',
        'neutral': 'neutral',
        'bored': 'bored',
        'sleepy': 'sleepy',
        'yawning': 'sleepy',
        'confused': 'neutral',
        'frustrated': 'bored',
        'looking_away': 'bored',
        'focused': 'engaged',
        'distracted': 'bored'
    }
    return engagement_map.get(class_name.lower(), 'neutral')

def calculate_iou(box1, box2):
    """Calculate Intersection over Union for tracking"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    
    iou = inter_area / (box1_area + box2_area - inter_area)
    return iou

def assign_student_id(box, current_detections):
    """Assign consistent student IDs using simple tracking"""
    global student_trackers, next_student_id
    
    best_match_id = None
    best_iou = 0.3
    
    for student_id, tracker_info in student_trackers.items():
        iou = calculate_iou(box, tracker_info['box'])
        if iou > best_iou:
            best_iou = iou
            best_match_id = student_id
    
    if best_match_id is not None:
        student_trackers[best_match_id]['box'] = box
        student_trackers[best_match_id]['last_seen'] = time.time()
        return best_match_id
    else:
        new_id = next_student_id
        next_student_id += 1
        student_trackers[new_id] = {'box': box, 'last_seen': time.time()}
        return new_id

def cleanup_trackers():
    """Remove trackers for students not seen in last 2 seconds"""
    global student_trackers, previous_states, alert_cooldown
    current_time = time.time()
    to_remove = [sid for sid, info in student_trackers.items() 
                 if current_time - info['last_seen'] > 2.0]
    for sid in to_remove:
        del student_trackers[sid]
        # Clean up alert tracking for removed students
        if sid in previous_states:
            del previous_states[sid]
        if sid in alert_cooldown:
            del alert_cooldown[sid]

def check_alert_trigger(student_id, current_state):
    """
    Check if an alert should be triggered for state change.
    Only alerts when transitioning from engaged/neutral to disengaged states.
    Returns: (should_alert: bool, alert_message: str or None)
    """
    global previous_states, alert_cooldown
    
    current_time = time.time()
    
    # Define disengaged states that should trigger alerts
    disengaged_states = ['sleepy', 'bored', 'yawning', 'frustrated']
    engaged_states = ['engaged', 'neutral', 'confused']
    
    # Check cooldown (don't alert same student within 10 seconds)
    if student_id in alert_cooldown:
        if current_time - alert_cooldown[student_id] < 10.0:
            return False, None
    
    # Get previous state
    prev_state = previous_states.get(student_id, 'engaged')
    
    # Update previous state
    previous_states[student_id] = current_state
    
    # Alert only on transition from engaged/neutral to disengaged
    if prev_state in engaged_states and current_state in disengaged_states:
        alert_cooldown[student_id] = current_time
        
        # Create specific alert message
        alert_messages = {
            'sleepy': f'Student #{student_id} appears SLEEPY',
            'bored': f'Student #{student_id} appears BORED',
            'yawning': f'Student #{student_id} is YAWNING',
            'frustrated': f'Student #{student_id} appears FRUSTRATED'
        }
        
        return True, alert_messages.get(current_state, f'Student #{student_id} disengaged')
    
    return False, None

def update_active_alerts():
    """Remove alerts older than 5 seconds"""
    global active_alerts
    current_time = time.time()
    active_alerts = [(msg, ts) for msg, ts in active_alerts if current_time - ts < 5.0]

def mid(a, b):
    """Calculate midpoint between two points"""
    return (a + b) / 2

def get_pose_features(keypoints):
    """
    Extract geometric features from YOLO11-pose keypoints (17 COCO keypoints)
    Keypoint indices: 0=nose, 1=left_eye, 2=right_eye, 5=left_shoulder, 6=right_shoulder,
                      11=left_hip, 12=right_hip, 13=left_knee, 14=right_knee
    Returns: dict of features or None if insufficient keypoints
    """
    if keypoints is None or len(keypoints) == 0:
        return None
    
    try:
        # keypoints shape: (17, 3) where each row is [x, y, confidence]
        kp = keypoints[0] if len(keypoints.shape) == 3 else keypoints
        
        # Extract key points (x, y)
        nose = kp[0, :2]
        left_eye = kp[1, :2]
        right_eye = kp[2, :2]
        left_shoulder = kp[5, :2]
        right_shoulder = kp[6, :2]
        left_hip = kp[11, :2]
        right_hip = kp[12, :2]
        
        # Confidence scores
        nose_conf = kp[0, 2]
        eye_conf = (kp[1, 2] + kp[2, 2]) / 2
        shoulder_conf = (kp[5, 2] + kp[6, 2]) / 2
        hip_conf = (kp[11, 2] + kp[12, 2]) / 2
        
        # Check if we have sufficient confidence in key points
        if shoulder_conf < 0.3 or hip_conf < 0.3:
            return None
        
        # Calculate midpoints
        shoulder_mid = mid(left_shoulder, right_shoulder)
        hip_mid = mid(left_hip, right_hip)
        eye_mid = mid(left_eye, right_eye)
        
        # Torso vector (shoulder to hip)
        torso_vec = shoulder_mid - hip_mid
        torso_length = np.linalg.norm(torso_vec)
        
        # Torso angle (forward/backward lean)
        # Positive = leaning forward, Negative = leaning back
        torso_angle = np.degrees(np.arctan2(torso_vec[0], torso_vec[1]))
        
        # Head vector (nose to shoulder midpoint)
        head_vec = nose - shoulder_mid
        head_angle = np.degrees(np.arctan2(head_vec[0], head_vec[1]))
        
        # Head droop (nose below shoulders = positive value)
        head_droop = nose[1] - shoulder_mid[1]
        head_droop_normalized = head_droop / torso_length if torso_length > 0 else 0
        
        # Shoulder-hip distance (normalized by frame height for scale invariance)
        shoulder_hip_dist = np.linalg.norm(shoulder_mid - hip_mid)
        
        # Head tilt (lateral)
        head_tilt = abs(nose[0] - shoulder_mid[0]) / torso_length if torso_length > 0 else 0
        
        return {
            'torso_angle': torso_angle,
            'head_angle': head_angle,
            'head_droop_normalized': head_droop_normalized,
            'head_tilt': head_tilt,
            'eye_conf': eye_conf,
            'nose_conf': nose_conf,
            'torso_length': torso_length,
            'shoulder_hip_dist': shoulder_hip_dist
        }
        
    except Exception as e:
        print(f"Error extracting pose features: {e}")
        return None

def classify_posture(features):
    """
    Classify engagement state based on geometric features
    Returns: 'engaged', 'bored', 'sleepy', 'yawning', 'frustrated', or 'confused'
    
    Tunable thresholds (adjust based on your camera angle and data):
    """
    if features is None:
        return None
    
    torso_angle = features['torso_angle']
    head_droop = features['head_droop_normalized']
    head_tilt = features['head_tilt']
    eye_conf = features['eye_conf']
    
    # SLEEPY: Head significantly drooped, low eye confidence, hunched forward
    if head_droop > 0.15 and (eye_conf < 0.4 or torso_angle > 15):
        return 'sleepy'
    
    # YAWNING: Head tilted back (negative droop), mouth open (low nose confidence can indicate)
    elif head_droop < -0.1 and torso_angle < -10:
        return 'yawning'
    
    # BORED/SLOUCHED: Forward lean, moderate head droop
    elif torso_angle > 20 or (head_droop > 0.08 and torso_angle > 10):
        return 'bored'
    
    # FRUSTRATED: Extreme head tilt or unusual posture
    elif head_tilt > 0.15:
        return 'frustrated'
    
    # CONFUSED: Moderate head tilt
    elif head_tilt > 0.08:
        return 'confused'
    
    # ENGAGED: Upright posture, head aligned, good eye confidence
    elif abs(torso_angle) < 15 and head_droop < 0.05 and eye_conf > 0.5:
        return 'engaged'
    
    # Default: neutral/engaged
    else:
        return 'engaged'

def combine_engagement_signals(face_state, posture_state):
    """
    Combine facial expression and body posture
    Priority: Face detection > Posture (face is more accurate)
    """
    if not posture_state:
        return face_state
    
    # Normalize to lowercase for comparison
    face_lower = face_state.lower()
    posture_lower = posture_state.lower()
    
    # If both agree, use that state
    if face_lower == posture_lower:
        return face_state
    
    # Priority rules: face detection is more reliable
    # But if posture shows extreme disengagement, consider it
    
    if face_lower == 'sleepy' or posture_lower == 'sleepy':
        return 'sleepy'
    
    if face_lower == 'yawning' or posture_lower == 'yawning':
        return 'yawning'
    
    if face_lower == 'bored' and posture_lower in ['bored', 'frustrated']:
        return 'bored'
    
    if face_lower == 'frustrated' or posture_lower == 'frustrated':
        return 'frustrated'
    
    if face_lower == 'confused' or posture_lower == 'confused':
        return 'confused'
    
    if face_lower == 'engaged' and posture_lower == 'engaged':
        return 'engaged'
    
    # Default: trust face detection
    return face_state



# ============================================================================
# Video Processing
# ============================================================================
def generate_frames():
    """Generate video frames with detection and pose analysis"""
    global camera, detection_active, pose_detection_active, current_session_id
    
    # Initialize camera
    if camera is None or not camera.isOpened():
        if camera is not None:
            camera.release()
            time.sleep(0.5)  # Give time for camera to release
        
        # Try DirectShow backend for Windows
        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not camera.isOpened():
            print("Error: Cannot open camera with DirectShow, trying default...")
            camera = cv2.VideoCapture(0)
            if not camera.isOpened():
                print("Error: Cannot open camera")
                return
        
        # Set camera properties
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 30)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Warm up camera
        for _ in range(5):
            camera.read()
        
        print("✓ Camera initialized successfully")
    
    try:
        while True:
            # Read frame
            success, frame = camera.read()
            if not success or frame is None:
                print("Failed to read frame")
                time.sleep(0.1)
                continue
            
            # Validate frame
            if frame.size == 0 or len(frame.shape) != 3:
                print("Invalid frame dimensions")
                time.sleep(0.1)
                continue
            
            frame_height, frame_width = frame.shape[:2]
            
            # Ensure frame is valid before processing
            if frame_height == 0 or frame_width == 0:
                print("Invalid frame size")
                continue
            posture_state = None
            current_detections = []
            detections_to_save = []
            pose_results = None
            pose_keypoints = None
            
            # Process with YOLO11-pose (only if enabled)
            if pose_detection_active and pose_model is not None:
                pose_results = pose_model(frame, conf=0.3, verbose=False)
                
                if len(pose_results) > 0 and pose_results[0].keypoints is not None:
                    # Get keypoints from first detected person
                    kpts = pose_results[0].keypoints.data.cpu().numpy()
                    if len(kpts) > 0:
                        pose_keypoints = kpts[0]  # First person's keypoints (17, 3)
                        features = get_pose_features(pose_keypoints)
                        if features:
                            posture_state = classify_posture(features)
                            print(f"Posture: {posture_state} | Torso angle: {features['torso_angle']:.1f}° | Head droop: {features['head_droop_normalized']:.2f}")
            
            # YOLO Detection
            if detection_active and model is not None:
                results = model(frame, conf=0.4, iou=0.5, verbose=False)
                
                if len(results) > 0:
                    r = results[0]
                    frame = r.plot()  # YOLO's built-in visualization
                    
                    if r.boxes is not None and len(r.boxes) > 0:
                        print(f"YOLO detected {len(r.boxes)} objects")
                        
                        for box in r.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            confidence = float(box.conf[0])
                            class_id = int(box.cls[0])
                            class_name = model.names[class_id]
                            
                            student_id = assign_student_id([x1, y1, x2, y2], current_detections)
                            current_detections.append(student_id)
                            
                            face_state = get_engagement_state(class_name)
                            engagement_state = combine_engagement_signals(face_state, posture_state)
                            engagement_history[student_id].append(engagement_state)
                            
                            # Check for alert trigger (real-time state change detection)
                            should_alert, alert_message = check_alert_trigger(student_id, engagement_state)
                            if should_alert and alert_message:
                                active_alerts.append((alert_message, time.time()))
                                print(f"🚨 ALERT: {alert_message}")
                            
                            # Add overlay text
                            if posture_state:
                                cv2.putText(frame, f"Posture: {posture_state}", (x1, y2 + 20), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                            
                            text_color = (0, 0, 255) if engagement_state in ['sleepy', 'bored'] else \
                                        (0, 255, 0) if engagement_state == 'engaged' else (255, 255, 255)
                            
                            y_offset = 40 if posture_state else 20
                            cv2.putText(frame, f"State: {engagement_state.upper()}", (x1, y2 + y_offset), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
                            
                            # Save to database (without images for privacy)
                            if current_session_id:
                                detections_to_save.append({
                                    'session_id': current_session_id,
                                    'student_id': student_id,
                                    'engagement_state': engagement_state,
                                    'confidence': confidence,
                                    'image_path': None  # No images saved for privacy
                                })
                    else:
                        print("No detections in this frame")
                
                # Save detections to database
                if current_session_id and detections_to_save:
                    with app.app_context():
                        try:
                            for det_data in detections_to_save:
                                detection = Detection(**det_data)
                                db.session.add(detection)
                            db.session.commit()
                            print(f"Saved {len(detections_to_save)} detections")
                        except Exception as e:
                            print(f"Database error: {e}")
                            db.session.rollback()
                
                cleanup_trackers()
                update_active_alerts()
                
                # Display stats
                if len(current_detections) > 0:
                    cv2.putText(frame, f"Students: {len(current_detections)}", (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (250, 165, 96), 2)
                else:
                    cv2.putText(frame, "Scanning for faces...", (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                # Display active alerts (real-time, only when state changes)
                if active_alerts:
                    y_pos = 60
                    for alert_msg, _ in active_alerts:
                        # Draw alert background for better visibility
                        text_size = cv2.getTextSize(alert_msg, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                        cv2.rectangle(frame, (5, y_pos - 25), (text_size[0] + 15, y_pos + 5), 
                                    (0, 0, 255), -1)
                        cv2.putText(frame, f"⚠ {alert_msg}", (10, y_pos), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                        y_pos += 35
            
            # Draw skeleton (only if pose detection is active)
            if pose_detection_active and pose_keypoints is not None:
                # Draw keypoints manually to avoid overwriting frame
                try:
                    for i, (x, y, conf) in enumerate(pose_keypoints):
                        if conf > 0.5:  # Only draw confident keypoints
                            cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
                    
                    # Draw skeleton connections (simplified)
                    connections = [
                        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
                        (5, 11), (6, 12), (11, 12),  # Torso
                        (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
                    ]
                    for start_idx, end_idx in connections:
                        if pose_keypoints[start_idx][2] > 0.5 and pose_keypoints[end_idx][2] > 0.5:
                            start_point = (int(pose_keypoints[start_idx][0]), int(pose_keypoints[start_idx][1]))
                            end_point = (int(pose_keypoints[end_idx][0]), int(pose_keypoints[end_idx][1]))
                            cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
                except Exception as e:
                    print(f"Error drawing skeleton: {e}")
                
                if posture_state:
                    # Color mapping for all model classes
                    posture_colors = {
                        'sleepy': (0, 0, 255),      # Red
                        'bored': (0, 165, 255),     # Orange
                        'confused': (0, 255, 255),  # Yellow
                        'engaged': (0, 255, 0),     # Green
                        'frustrated': (128, 0, 128), # Purple
                        'yawning': (255, 0, 0)      # Blue
                    }
                    color = posture_colors.get(posture_state.lower(), (255, 255, 255))
                    cv2.putText(frame, f"Posture: {posture_state.upper()}", (10, frame_height - 20), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Encode and yield frame
            try:
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not ret:
                    continue
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            except Exception as e:
                print(f"Error encoding frame: {e}")
                continue
                
    except Exception as e:
        print(f"Error in generate_frames: {e}")

# ============================================================================
# Flask Routes
# ============================================================================
@app.route('/')
def index():
    if 'user_id' not in flask_session:
        return redirect(url_for('login'))
    return render_template('home.html', user_id=flask_session.get('user_id'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.get_json()
        user = User.query.filter_by(email=data.get('email')).first()
        
        if user and check_password_hash(user.password_hash, data.get('password')):
            flask_session['user_id'] = user.id
            return jsonify({'success': True})
        return jsonify({'success': False, 'message': 'Invalid credentials'}), 401
    
    return render_template('login.html')

@app.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()
    
    if User.query.filter_by(email=data.get('email')).first():
        return jsonify({'success': False, 'message': 'Email already exists'}), 400
    
    user = User(email=data.get('email'), password_hash=generate_password_hash(data.get('password')))
    db.session.add(user)
    db.session.commit()
    return jsonify({'success': True})

@app.route('/logout')
def logout():
    flask_session.pop('user_id', None)
    return redirect(url_for('login'))

@app.route('/live')
def live():
    if 'user_id' not in flask_session:
        return redirect(url_for('login'))
    return render_template('live.html')

@app.route('/reports')
def reports():
    if 'user_id' not in flask_session:
        return redirect(url_for('login'))
    return render_template('reports.html')

@app.route('/settings')
def settings():
    if 'user_id' not in flask_session:
        return redirect(url_for('login'))
    return render_template('settings.html')

@app.route('/about')
def about():
    if 'user_id' not in flask_session:
        return redirect(url_for('login'))
    return render_template('about.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_detection', methods=['POST'])
def start_detection():
    global detection_active, current_session_id, student_trackers, next_student_id
    global previous_states, active_alerts, alert_cooldown
    
    if 'user_id' not in flask_session:
        return jsonify({'success': False, 'message': 'Not authenticated'}), 401
    
    detection_active = True
    student_trackers = {}
    next_student_id = 1
    # Reset alert tracking
    previous_states = {}
    active_alerts = []
    alert_cooldown = {}
    
    new_session = Session(user_id=flask_session['user_id'])
    db.session.add(new_session)
    db.session.commit()
    current_session_id = new_session.id
    
    return jsonify({'success': True, 'session_id': current_session_id})

@app.route('/toggle_pose', methods=['POST'])
def toggle_pose():
    """Toggle pose detection on/off"""
    global pose_detection_active
    
    pose_detection_active = not pose_detection_active
    return jsonify({
        'success': True, 
        'pose_active': pose_detection_active,
        'message': f"Pose detection {'enabled' if pose_detection_active else 'disabled'}"
    })

@app.route('/stop_detection', methods=['POST'])
def stop_detection():
    global detection_active, current_session_id
    
    detection_active = False
    
    if current_session_id:
        session_obj = db.session.get(Session, current_session_id)
        if session_obj:
            session_obj.end_time = datetime.utcnow()
            session_obj.duration = int((session_obj.end_time - session_obj.start_time).total_seconds())
            
            detections = Detection.query.filter_by(session_id=current_session_id).all()
            if detections:
                total = len(detections)
                engaged = sum(1 for d in detections if d.engagement_state == 'engaged')
                neutral = sum(1 for d in detections if d.engagement_state == 'neutral')
                bored = sum(1 for d in detections if d.engagement_state in ['bored', 'sleepy'])
                
                session_obj.engaged_percentage = (engaged / total) * 100
                session_obj.neutral_percentage = (neutral / total) * 100
                session_obj.bored_percentage = (bored / total) * 100
                session_obj.total_students = len(set(d.student_id for d in detections))
            
            db.session.commit()
        current_session_id = None
    
    return jsonify({'success': True})

@app.route('/session_stats')
def session_stats():
    global current_session_id, student_trackers
    
    if not current_session_id:
        return jsonify({'active': False, 'duration': 0, 'students': 0, 'engaged': 0, 'neutral': 0, 'bored': 0})
    
    session_obj = db.session.get(Session, current_session_id)
    if not session_obj:
        return jsonify({'active': False})
    
    duration = int((datetime.utcnow() - session_obj.start_time).total_seconds())
    recent_detections = Detection.query.filter_by(session_id=current_session_id)\
        .filter(Detection.timestamp >= datetime.utcnow() - timedelta(seconds=5)).all()
    
    if recent_detections:
        total = len(recent_detections)
        engaged_pct = (sum(1 for d in recent_detections if d.engagement_state == 'engaged') / total) * 100
        neutral_pct = (sum(1 for d in recent_detections if d.engagement_state == 'neutral') / total) * 100
        bored_pct = (sum(1 for d in recent_detections if d.engagement_state in ['bored', 'sleepy']) / total) * 100
    else:
        engaged_pct = neutral_pct = bored_pct = 0
    
    return jsonify({
        'active': True,
        'duration': duration,
        'students': len(student_trackers),
        'engaged': round(engaged_pct, 1),
        'neutral': round(neutral_pct, 1),
        'bored': round(bored_pct, 1)
    })

@app.route('/past_sessions')
def past_sessions():
    if 'user_id' not in flask_session:
        return jsonify({'success': False}), 401
    
    sessions = Session.query.filter_by(user_id=flask_session['user_id'])\
        .filter(Session.end_time.isnot(None))\
        .order_by(Session.start_time.desc()).all()
    
    return jsonify({'sessions': [{
        'id': s.id,
        'start_time': s.start_time.strftime('%Y-%m-%d %H:%M:%S'),
        'duration': s.duration,
        'students': s.total_students,
        'engaged': round(s.engaged_percentage, 1),
        'neutral': round(s.neutral_percentage, 1),
        'bored': round(s.bored_percentage, 1)
    } for s in sessions]})

@app.route('/session_details/<int:session_id>')
def session_details(session_id):
    if 'user_id' not in flask_session:
        return jsonify({'success': False}), 401
    
    session_obj = db.session.get(Session, session_id)
    if not session_obj or session_obj.user_id != flask_session['user_id']:
        return jsonify({'success': False}), 404
    
    detections = Detection.query.filter_by(session_id=session_id).order_by(Detection.timestamp).all()
    
    student_data = defaultdict(list)
    
    for d in detections:
        student_data[d.student_id].append({
            'timestamp': d.timestamp.strftime('%H:%M:%S'),
            'state': d.engagement_state,
            'confidence': round(d.confidence, 2)
        })
    
    return jsonify({
        'session': {
            'id': session_obj.id,
            'start_time': session_obj.start_time.strftime('%Y-%m-%d %H:%M:%S'),
            'duration': session_obj.duration,
            'students': session_obj.total_students,
            'engaged': round(session_obj.engaged_percentage, 1),
            'neutral': round(session_obj.neutral_percentage, 1),
            'bored': round(session_obj.bored_percentage, 1)
        },
        'student_data': dict(student_data)
    })

# ============================================================================
# Cleanup
# ============================================================================
def cleanup_camera():
    """Release camera resources on shutdown"""
    global camera
    if camera is not None:
        try:
            camera.release()
            print("Camera released successfully")
        except Exception as e:
            print(f"Error releasing camera: {e}")

# ============================================================================
# Main
# ============================================================================
if __name__ == '__main__':
    atexit.register(cleanup_camera)
    
    with app.app_context():
        db.create_all()
    
    try:
        app.run(host='0.0.0.0', port=5002, debug=True)
    except KeyboardInterrupt:
        print("\nShutting down...")
        cleanup_camera()
