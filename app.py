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
    
    if camera is None:
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 30)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        if detection_active and model is not None:
            # Run YOLO inference - lowered conf to 0.25 for better detection
            results = model(frame, conf=0.25, iou=0.45, verbose=False, stream=False)
            
            current_detections = []
            detection_count = 0
            detections_to_save = []  # Collect detections to save in batch
            
            for result in results:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    detection_count = len(boxes)
                    print(f"Detected {detection_count} objects")
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    print(f"Detected class: {class_name} with confidence {confidence:.2f}")
                    
                    # Assign student ID
                    student_id = assign_student_id([x1, y1, x2, y2], current_detections)
                    current_detections.append(student_id)
                    
                    # Get engagement state
                    engagement_state = get_engagement_state(class_name)
                    
                    # Update engagement history
                    engagement_history[student_id].append(engagement_state)
                    
                    # Determine color based on engagement - RED ALERT for sleepy
                    if engagement_state == 'sleepy' or engagement_state == 'bored':
                        color = (94, 93, 201)  # Red alert (#c95d5e in BGR)
                        thickness = 3
                        # Add ALERT text for sleepy students
                        alert_label = "ALERT: SLEEPY!"
                        cv2.putText(frame, alert_label, (x1, y2 + 20), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (94, 93, 201), 2)
                    elif engagement_state == 'engaged':
                        color = (250, 165, 96)  # Blue (#60a5fa in BGR)
                        thickness = 2
                    else:  # neutral
                        color = (81, 65, 55)  # Gray (#374151 in BGR)
                        thickness = 2
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                    
                    # Draw label background
                    label = f"ID:{student_id} {class_name} {confidence:.2f}"
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                                (x1 + label_size[0], y1), color, -1)
                    
                    # Draw label text
                    cv2.putText(frame, label, (x1, y1 - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # Prepare detection for database save
                    if current_session_id:
                        # Check if we need to save a snapshot for this student
                        snapshot_key = f"{current_session_id}_{student_id}"
                        image_path = None
                        
                        if snapshot_key not in student_snapshots:
                            # Save snapshot with bounding box (using the frame after drawing)
                            image_path = save_student_snapshot(frame, student_id, current_session_id, [x1, y1, x2, y2])
                            if image_path:
                                student_snapshots[snapshot_key] = image_path
                                print(f"Saved snapshot for student {student_id}: {image_path}")
                        else:
                            # Use existing snapshot path
                            image_path = student_snapshots[snapshot_key]
                        
                        detections_to_save.append({
                            'session_id': current_session_id,
                            'student_id': student_id,
                            'engagement_state': engagement_state,
                            'confidence': confidence,
                            'image_path': image_path
                        })
            
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
            
            # Add status display
            fps_text = f"Students: {len(current_detections)}"
            cv2.putText(frame, fps_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (250, 165, 96), 2)
            
            # Check for sleepy students and show alert
            sleepy_count = sum(1 for sid in current_detections 
                             if engagement_history[sid] and 
                             engagement_history[sid][-1] in ['sleepy', 'bored'])
            if sleepy_count > 0:
                alert_text = f"ALERT: {sleepy_count} Sleepy Student(s)!"
                cv2.putText(frame, alert_text, (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (94, 93, 201), 2)
        
        # Encode frame with optimized quality for speed
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

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
    global detection_active, current_session_id
    
    detection_active = False
    
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

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    # Run the app on all available network interfaces so you can access from other devices
    app.run(host='0.0.0.0', port=5002, debug=True)
