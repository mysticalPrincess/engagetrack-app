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
detection_active = False
current_session_id = None
student_trackers = {}
next_student_id = 1
engagement_history = defaultdict(lambda: deque(maxlen=30))
student_snapshots = {}

# ============================================================================
# MediaPipe Initialization
# ============================================================================
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

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

# Initialize model on startup
load_model()

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
    global student_trackers
    current_time = time.time()
    to_remove = [sid for sid, info in student_trackers.items() 
                 if current_time - info['last_seen'] > 2.0]
    for sid in to_remove:
        del student_trackers[sid]

def analyze_posture(pose_landmarks, frame_height):
    """Analyze body posture from MediaPipe pose landmarks"""
    if not pose_landmarks:
        return None
    
    try:
        landmarks = pose_landmarks.landmark
        nose = landmarks[mp_pose.PoseLandmark.NOSE]
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
        
        shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
        hip_y = (left_hip.y + right_hip.y) / 2
        head_y = nose.y
        
        torso_length = abs(shoulder_y - hip_y)
        head_to_shoulder = abs(head_y - shoulder_y)
        
        if torso_length < 0.15:
            return 'bored'
        elif head_y > shoulder_y + 0.05:
            return 'tired'
        elif head_to_shoulder < 0.08:
            return 'tired'
        elif shoulder_y < hip_y - 0.25:
            return 'bored'
        elif torso_length > 0.18 and head_y < shoulder_y:
            return 'attentive'
        else:
            return 'active'
    except Exception as e:
        print(f"Error analyzing posture: {e}")
        return None

def combine_engagement_signals(face_state, posture_state):
    """Combine facial expression and body posture"""
    if not posture_state:
        return face_state
    
    if face_state == 'sleepy' or posture_state == 'tired':
        return 'sleepy'
    if face_state == 'bored' or posture_state == 'bored':
        return 'bored'
    if face_state == 'engaged' and posture_state == 'attentive':
        return 'engaged'
    if face_state == 'engaged' or posture_state == 'attentive':
        return 'engaged'
    
    return 'neutral'

def save_student_snapshot(frame, student_id, session_id, bbox):
    """Save a snapshot image of a detected student"""
    try:
        session_dir = os.path.join('static', 'session_snapshots', str(session_id))
        os.makedirs(session_dir, exist_ok=True)
        
        snapshot = frame.copy()
        filename = f'student_{student_id}.jpg'
        filepath = os.path.join(session_dir, filename)
        
        cv2.imwrite(filepath, snapshot)
        return f'session_snapshots/{session_id}/{filename}'
    except Exception as e:
        print(f"Error saving snapshot: {e}")
        return None

# ============================================================================
# Video Processing
# ============================================================================
def generate_frames():
    """Generate video frames with detection and pose analysis"""
    global camera, detection_active, current_session_id
    
    # Initialize camera
    if camera is None or not camera.isOpened():
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
            # Read frame
            success, frame = camera.read()
            if not success or frame is None or frame.size == 0:
                print("Failed to read frame")
                break
            
            frame_height, frame_width = frame.shape[:2]
            posture_state = None
            current_detections = []
            detections_to_save = []
            
            # Process with MediaPipe Pose
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results_pose = pose.process(rgb_frame)
            
            if results_pose.pose_landmarks:
                posture_state = analyze_posture(results_pose.pose_landmarks, frame_height)
            
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
                            
                            # Add overlay text
                            if posture_state:
                                cv2.putText(frame, f"Posture: {posture_state}", (x1, y2 + 20), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                            
                            text_color = (0, 0, 255) if engagement_state in ['sleepy', 'bored'] else \
                                        (0, 255, 0) if engagement_state == 'engaged' else (255, 255, 255)
                            
                            y_offset = 40 if posture_state else 20
                            cv2.putText(frame, f"State: {engagement_state.upper()}", (x1, y2 + y_offset), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
                            
                            # Save to database
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
                
                # Alert system
                sleepy_count = bored_count = yawning_count = 0
                if len(results) > 0 and results[0].boxes is not None:
                    for box in results[0].boxes:
                        class_name = model.names[int(box.cls[0])].lower()
                        if class_name == 'sleepy':
                            sleepy_count += 1
                        elif class_name == 'bored':
                            bored_count += 1
                        elif class_name == 'yawning':
                            yawning_count += 1
                
                # Display stats
                if len(current_detections) > 0:
                    cv2.putText(frame, f"Students: {len(current_detections)}", (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (250, 165, 96), 2)
                else:
                    cv2.putText(frame, "Scanning for faces...", (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                # Alert
                alert_count = sleepy_count + bored_count + yawning_count
                if alert_count > 0:
                    alert_parts = []
                    if sleepy_count > 0:
                        alert_parts.append(f"{sleepy_count} Sleepy")
                    if bored_count > 0:
                        alert_parts.append(f"{bored_count} Bored")
                    if yawning_count > 0:
                        alert_parts.append(f"{yawning_count} Yawning")
                    
                    cv2.putText(frame, f"ALERT: {', '.join(alert_parts)} Student(s)!", (10, 60), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Draw skeleton
            if results_pose.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                )
                
                if posture_state:
                    posture_colors = {
                        'attentive': (0, 255, 0), 'active': (255, 165, 0),
                        'bored': (0, 165, 255), 'tired': (0, 0, 255)
                    }
                    color = posture_colors.get(posture_state, (255, 255, 255))
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
    global detection_active, current_session_id, student_trackers, next_student_id, student_snapshots
    
    if 'user_id' not in flask_session:
        return jsonify({'success': False, 'message': 'Not authenticated'}), 401
    
    detection_active = True
    student_trackers = {}
    next_student_id = 1
    student_snapshots = {}
    
    new_session = Session(user_id=flask_session['user_id'])
    db.session.add(new_session)
    db.session.commit()
    current_session_id = new_session.id
    
    return jsonify({'success': True, 'session_id': current_session_id})

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
    student_images = {}
    
    for d in detections:
        student_data[d.student_id].append({
            'timestamp': d.timestamp.strftime('%H:%M:%S'),
            'state': d.engagement_state,
            'confidence': round(d.confidence, 2)
        })
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
