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
import mediapipe as mp

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:Mass2712@localhost:5432/engagetrack'
app.config['SECRET_KEY'] = 'your-secret-key-change-this' 
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Database Models
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
    duration = db.Column(db.Integer)  # in seconds
    total_students = db.Column(db.Integer, default=0)
    engaged_percentage = db.Column(db.Float, default=0.0)
    neutral_percentage = db.Column(db.Float, default=0.0)
    bored_percentage = db.Column(db.Float, default=0.0)

class Detection(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('session.id'), nullable=False)
    student_id = db.Column(db.Integer, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    engagement_state = db.Column(db.String(50))  # engaged, neutral, bored, sleepy
    confidence = db.Column(db.Float)
    image_path = db.Column(db.String(255))  # Path to saved snapshot image

# Global variables for video processing
camera = None
model = None
detection_active = False
current_session_id = None
student_trackers = {}
next_student_id = 1
engagement_history = defaultdict(lambda: deque(maxlen=30))  # 30 frames history per student
student_snapshots = {}  # Track saved snapshots: {student_id: image_path}

# Initialize MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Load YOLO model
def load_model():
    global model
    try:
        # Try loading .onnx model first (as shown in user's files), then .pt
        if os.path.exists('engagement_detection_model.onnx'):
            model = YOLO('engagement_detection_model.onnx', task='detect')
            print("Loaded ONNX model successfully")
        elif os.path.exists('engagement_detection_model.pt'):
            model = YOLO('engagement_detection_model.pt', task='detect')
            print("Loaded PT model successfully")
        elif os.path.exists('best.onnx'):
            model = YOLO('best.onnx', task='detect')
            print("Loaded best.onnx model successfully")
        else:
            print("Model file not found! Please add engagement_detection_model.onnx or .pt to the root directory")
            return False
        
        # Print model classes for debugging
        if model:
            print(f"Model classes: {model.names}")
        
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

# Initialize model on startup
load_model()

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
    best_iou = 0.3  # Minimum IOU threshold
    
    # Try to match with existing students
    for student_id, tracker_info in student_trackers.items():
        iou = calculate_iou(box, tracker_info['box'])
        if iou > best_iou:
            best_iou = iou
            best_match_id = student_id
    
    if best_match_id is not None:
        # Update existing tracker
        student_trackers[best_match_id]['box'] = box
        student_trackers[best_match_id]['last_seen'] = time.time()
        return best_match_id
    else:
        # Create new student ID
        new_id = next_student_id
        next_student_id += 1
        student_trackers[new_id] = {
            'box': box,
            'last_seen': time.time()
        }
        return new_id

def cleanup_trackers():
    """Remove trackers for students not seen in last 2 seconds"""
    global student_trackers
    current_time = time.time()
    to_remove = [sid for sid, info in student_trackers.items() 
                 if current_time - info['last_seen'] > 2.0]
    for sid in to_remove:
        del student_trackers[sid]

def analyze_posture(pose_landmarks, frame_height):
    """
    Analyze body posture from MediaPipe pose landmarks to determine engagement.
    Returns: posture state ('attentive', 'bored', 'tired', 'active')
    """
    if not pose_landmarks:
        return None
    
    try:
        # Extract key landmarks
        landmarks = pose_landmarks.landmark
        
        # Get coordinates (normalized 0-1, multiply by frame height for pixel values)
        nose = landmarks[mp_pose.PoseLandmark.NOSE]
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
        
        # Calculate average positions
        shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
        hip_y = (left_hip.y + right_hip.y) / 2
        head_y = nose.y
        
        # Calculate posture metrics
        torso_length = abs(shoulder_y - hip_y)
        head_to_shoulder = abs(head_y - shoulder_y)
        
        # Rule-based posture classification
        # Slouched: small torso length (compressed posture)
        if torso_length < 0.15:
            return 'bored'
        
        # Head down: head significantly below shoulders
        elif head_y > shoulder_y + 0.05:
            return 'tired'
        
        # Head tilted or resting: head very close to shoulders
        elif head_to_shoulder < 0.08:
            return 'tired'
        
        # Leaning back: shoulders significantly above hips (relaxed/disengaged)
        elif shoulder_y < hip_y - 0.25:
            return 'bored'
        
        # Upright and alert posture
        elif torso_length > 0.18 and head_y < shoulder_y:
            return 'attentive'
        
        # Default: active/neutral
        else:
            return 'active'
            
    except Exception as e:
        print(f"Error analyzing posture: {e}")
        return None

def combine_engagement_signals(face_state, posture_state):
    """
    Combine facial expression (YOLO) and body posture (MediaPipe) 
    to determine final engagement state.
    """
    if not posture_state:
        # If no posture detected, rely on face only
        return face_state
    
    # Engagement matrix: face x posture
    # Priority: if either signal shows disengagement, mark as disengaged
    
    if face_state == 'sleepy' or posture_state == 'tired':
        return 'sleepy'
    
    if face_state == 'bored' or posture_state == 'bored':
        return 'bored'
    
    if face_state == 'engaged' and posture_state == 'attentive':
        return 'engaged'
    
    if face_state == 'engaged' or posture_state == 'attentive':
        return 'engaged'
    
    # Both neutral/active
    return 'neutral'

def save_student_snapshot(frame, student_id, session_id, bbox):
    """Save a snapshot image of a detected student with bounding box"""
    try:
        # Create session directory if it doesn't exist
        session_dir = os.path.join('static', 'session_snapshots', str(session_id))
        os.makedirs(session_dir, exist_ok=True)
        
        # Create a copy of the frame for the snapshot
        snapshot = frame.copy()
        
        # Define filename
        filename = f'student_{student_id}.jpg'
        filepath = os.path.join(session_dir, filename)
        
        # Save the snapshot
        cv2.imwrite(filepath, snapshot)
        
        # Return relative path for database storage
        return f'session_snapshots/{session_id}/{filename}'
    except Exception as e:
        print(f"Error saving snapshot: {e}")
        return None

def generate_frames():
    global camera, detection_active, current_session_id
    
    if camera is None or not camera.isOpened():
        # Release if exists but not opened
        if camera is not None:
            camera.release()
        
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            print("Error: Cannot open camera")
            return
        
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 30)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    try:
        while True:
            success, frame = camera.read()
            if not success or frame is None:
                print("Failed to read frame from camera")
                break
            
            # Validate frame
            if frame.size == 0:
                print("Empty frame received")
                continue
            
            frame_height, frame_width = frame.shape[:2]
            posture_state = None
            
            # Process frame with MediaPipe Pose first (for all frames, not just when detection is active)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results_pose = pose.process(rgb_frame)
            
            # Analyze posture if landmarks detected
            if results_pose.pose_landmarks:
                posture_state = analyze_posture(results_pose.pose_landmarks, frame_height)
            
            if detection_active and model is not None:
                # Run YOLO inference with higher confidence threshold
                results = model(frame, conf=0.4, iou=0.5, verbose=False)
                
                current_detections = []
                detections_to_save = []
                
                # Get the first result (single frame)
                if len(results) > 0:
                    r = results[0]
                    
                    # Use YOLO's built-in plot() to draw boxes, labels, and confidence
                    frame = r.plot()
                    
                    # Process detections for tracking and database
                    if r.boxes is not None and len(r.boxes) > 0:
                        print(f"YOLO detected {len(r.boxes)} objects")
                        
                        for box in r.boxes:
                            # Get box coordinates
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            confidence = float(box.conf[0])
                            class_id = int(box.cls[0])
                            class_name = model.names[class_id]
                            
                            # Assign student ID for tracking
                            student_id = assign_student_id([x1, y1, x2, y2], current_detections)
                            current_detections.append(student_id)
                            
                            # Get facial engagement state from YOLO
                            face_state = get_engagement_state(class_name)
                            
                            # Combine facial expression with body posture
                            engagement_state = combine_engagement_signals(face_state, posture_state)
                            
                            # Update engagement history
                            engagement_history[student_id].append(engagement_state)
                            
                            # Add posture and engagement info overlay
                            if posture_state:
                                posture_text = f"Posture: {posture_state}"
                                cv2.putText(frame, posture_text, (x1, y2 + 20), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                            
                            engagement_text = f"State: {engagement_state.upper()}"
                            y_offset = 40 if posture_state else 20
                            
                            # Color based on engagement
                            if engagement_state in ['sleepy', 'bored']:
                                text_color = (0, 0, 255)  # Red
                            elif engagement_state == 'engaged':
                                text_color = (0, 255, 0)  # Green
                            else:
                                text_color = (255, 255, 255)  # White
                            
                            cv2.putText(frame, engagement_text, (x1, y2 + y_offset), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
                            
                            # Prepare detection for database save
                            if current_session_id:
                                snapshot_key = f"{current_session_id}_{student_id}"
                                image_path = None
                                
                                if snapshot_key not in student_snapshots:
                                    image_path = save_student_snapshot(frame, student_id, current_session_id, [x1, y1, x2, y2])
                                    if image_path:
                                        student_snapshots[snapshot_key] = image_path
                                else:
                                    image_path = student_snapshots[snapshot_key]
                                
                                detections_to_save.append({
                                    'session_id': current_session_id,
                                    'student_id': student_id,
                                    'engagement_state': engagement_state,
                                    'confidence': confidence,
                                    'image_path': image_path
                                })
                    else:
                        print("No detections in this frame")
            
            # Save all detections to database in a single context
            if current_session_id and detections_to_save:
                with app.app_context():
                    try:
                        for det_data in detections_to_save:
                            detection = Detection(
                                session_id=det_data['session_id'],
                                student_id=det_data['student_id'],
                                engagement_state=det_data['engagement_state'],
                                confidence=det_data['confidence'],
                                image_path=det_data.get('image_path')
                            )
                            db.session.add(detection)
                        db.session.commit()
                        print(f"Saved {len(detections_to_save)} detections to database")
                    except Exception as e:
                        print(f"Database commit error: {e}")
                        try:
                            db.session.rollback()
                        except:
                            pass  # Silently fail if rollback fails
            
            # Cleanup old trackers
            cleanup_trackers()
            
            # Check for sleepy/bored students directly from YOLO results and show alert
            sleepy_count = 0
            bored_count = 0
            yawning_count = 0
            
            if len(results) > 0 and results[0].boxes is not None:
                for box in results[0].boxes:
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id].lower()
                    
                    if class_name == 'sleepy':
                        sleepy_count += 1
                    elif class_name == 'bored':
                        bored_count += 1
                    elif class_name == 'yawning':
                        yawning_count += 1
            
            # Display student count
            if len(current_detections) > 0:
                fps_text = f"Students: {len(current_detections)}"
                cv2.putText(frame, fps_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (250, 165, 96), 2)
            else:
                # Show "scanning" message when detection is active but no faces found
                scan_text = "Scanning for faces..."
                cv2.putText(frame, scan_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Show alert if any sleepy, bored, or yawning students detected
            alert_count = sleepy_count + bored_count + yawning_count
            if alert_count > 0:
                alert_parts = []
                if sleepy_count > 0:
                    alert_parts.append(f"{sleepy_count} Sleepy")
                if bored_count > 0:
                    alert_parts.append(f"{bored_count} Bored")
                if yawning_count > 0:
                    alert_parts.append(f"{yawning_count} Yawning")
                
                alert_text = f"ALERT: {', '.join(alert_parts)} Student(s)!"
                cv2.putText(frame, alert_text, (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw skeleton pose landmarks (always, even when detection is inactive)
        if results_pose.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results_pose.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
            )
            
            # Display posture state on frame
            if posture_state:
                posture_color_map = {
                    'attentive': (0, 255, 0),    # Green
                    'active': (255, 165, 0),      # Orange
                    'bored': (0, 165, 255),       # Orange-red
                    'tired': (0, 0, 255)          # Red
                }
                posture_color = posture_color_map.get(posture_state, (255, 255, 255))
                posture_text = f"Posture: {posture_state.upper()}"
                cv2.putText(frame, posture_text, (10, frame_height - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, posture_color, 2)
            
            # Encode frame with optimized quality for speed
            try:
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not ret:
                    print("Failed to encode frame")
                    continue
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            except Exception as e:
                print(f"Error encoding frame: {e}")
                continue
            
    except Exception as e:
        print(f"Error in generate_frames: {e}")
    finally:
        # Cleanup is handled by stop_detection route
        pass

@app.route('/')
def index():
    if 'user_id' not in flask_session:
        return redirect(url_for('login'))
    return render_template('home.html', user_id=flask_session.get('user_id'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        
        user = User.query.filter_by(email=email).first()
        
        if user and check_password_hash(user.password_hash, password):
            flask_session['user_id'] = user.id
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'message': 'Invalid credentials'}), 401
    
    return render_template('login.html')

@app.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    
    if User.query.filter_by(email=email).first():
        return jsonify({'success': False, 'message': 'Email already exists'}), 400
    
    user = User(email=email, password_hash=generate_password_hash(password))
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
    global detection_active, current_session_id, student_trackers, next_student_id, student_snapshots
    
    if 'user_id' not in flask_session:
        return jsonify({'success': False, 'message': 'Not authenticated'}), 401
    
    detection_active = True
    student_trackers = {}
    next_student_id = 1
    student_snapshots = {}  # Clear snapshots for new session
    
    # Create new session
    new_session = Session(user_id=flask_session['user_id'])
    db.session.add(new_session)
    db.session.commit()
    current_session_id = new_session.id
    
    return jsonify({'success': True, 'session_id': current_session_id})

@app.route('/stop_detection', methods=['POST'])
def stop_detection():
    global detection_active, current_session_id, camera
    
    detection_active = False
    
    # Don't release camera here - it's still needed for video feed
    # Camera will be released when app shuts down
    
    if current_session_id:
        # Update session with final statistics
        session_obj = db.session.get(Session, current_session_id)
        if session_obj:
            session_obj.end_time = datetime.utcnow()
            session_obj.duration = int((session_obj.end_time - session_obj.start_time).total_seconds())
            
            # Calculate engagement statistics
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
        return jsonify({
            'active': False,
            'duration': 0,
            'students': 0,
            'engaged': 0,
            'neutral': 0,
            'bored': 0
        })
    
    session_obj = db.session.get(Session, current_session_id)
    if not session_obj:
        return jsonify({'active': False})
    
    duration = int((datetime.utcnow() - session_obj.start_time).total_seconds())
    
    # Get recent detections for statistics
    recent_detections = Detection.query.filter_by(session_id=current_session_id)\
        .filter(Detection.timestamp >= datetime.utcnow() - timedelta(seconds=5)).all()
    
    print(f"Session stats: Found {len(recent_detections)} recent detections, {len(student_trackers)} active trackers")
    
    if recent_detections:
        total = len(recent_detections)
        engaged = sum(1 for d in recent_detections if d.engagement_state == 'engaged')
        neutral = sum(1 for d in recent_detections if d.engagement_state == 'neutral')
        bored = sum(1 for d in recent_detections if d.engagement_state in ['bored', 'sleepy'])
        
        engaged_pct = (engaged / total) * 100
        neutral_pct = (neutral / total) * 100
        bored_pct = (bored / total) * 100
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
    
    sessions_data = [{
        'id': s.id,
        'start_time': s.start_time.strftime('%Y-%m-%d %H:%M:%S'),
        'duration': s.duration,
        'students': s.total_students,
        'engaged': round(s.engaged_percentage, 1),
        'neutral': round(s.neutral_percentage, 1),
        'bored': round(s.bored_percentage, 1)
    } for s in sessions]
    
    return jsonify({'sessions': sessions_data})

@app.route('/session_details/<int:session_id>')
def session_details(session_id):
    if 'user_id' not in flask_session:
        return jsonify({'success': False}), 401
    
    session_obj = db.session.get(Session, session_id)
    if not session_obj or session_obj.user_id != flask_session['user_id']:
        return jsonify({'success': False}), 404
    
    detections = Detection.query.filter_by(session_id=session_id)\
        .order_by(Detection.timestamp).all()
    
    # Group by student and track image paths
    student_data = defaultdict(list)
    student_images = {}  # Store one image per student
    
    for d in detections:
        student_data[d.student_id].append({
            'timestamp': d.timestamp.strftime('%H:%M:%S'),
            'state': d.engagement_state,
            'confidence': round(d.confidence, 2)
        })
        # Store the first image path found for each student
        if d.student_id not in student_images and d.image_path:
            student_images[d.student_id] = d.image_path
    
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
        'student_data': dict(student_data),
        'student_images': student_images
    })

def cleanup_camera():
    """Release camera resources on shutdown"""
    global camera
    if camera is not None:
        try:
            camera.release()
            print("Camera released successfully")
        except Exception as e:
            print(f"Error releasing camera: {e}")

if __name__ == '__main__':
    import atexit
    atexit.register(cleanup_camera)
    
    with app.app_context():
        db.create_all()
    
    try:
        # Run the app on all available network interfaces so you can access from other devices
        app.run(host='0.0.0.0', port=5002, debug=True)
    except KeyboardInterrupt:
        print("\nShutting down...")
        cleanup_camera()
