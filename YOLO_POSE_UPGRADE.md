# YOLO11-Pose Integration for Engagement Detection

## Overview
Replaced MediaPipe pose detection with **YOLO11-pose** for more accurate and integrated engagement detection using geometric feature extraction from 17 COCO keypoints.

## What Changed

### Before (MediaPipe)
- Used MediaPipe Pose with 33 landmarks
- Rule-based analysis with normalized coordinates
- Separate library dependency

### After (YOLO11-pose)
- Uses YOLO11n-pose.pt with 17 COCO keypoints
- Geometric feature extraction (angles, distances, ratios)
- Unified Ultralytics ecosystem
- More robust and accurate

## Architecture

### 1. Keypoint Detection
```python
pose_model = YOLO('yolo11n-pose.pt')
results = pose_model(frame, conf=0.3)
keypoints = results[0].keypoints.data  # Shape: (N, 17, 3)
```

**17 COCO Keypoints:**
- 0: Nose
- 1-2: Left/Right Eye
- 3-4: Left/Right Ear
- 5-6: Left/Right Shoulder
- 7-8: Left/Right Elbow
- 9-10: Left/Right Wrist
- 11-12: Left/Right Hip
- 13-14: Left/Right Knee
- 15-16: Left/Right Ankle

### 2. Feature Extraction
Extracts geometric features for engagement classification:

#### Key Features:
1. **Torso Angle** - Forward/backward lean
   - Calculated from shoulder-to-hip vector
   - Positive = leaning forward (bored/sleepy)
   - Negative = leaning back (yawning)

2. **Head Droop** - Vertical head position
   - Normalized by torso length (scale-invariant)
   - Positive = head below shoulders (sleepy)
   - Negative = head above shoulders (alert/yawning)

3. **Head Tilt** - Lateral head position
   - Normalized by torso length
   - High values = confused/frustrated

4. **Eye Confidence** - Keypoint detection confidence
   - Low confidence may indicate closed eyes (sleepy)

### 3. Classification Rules

```python
def classify_posture(features):
    torso_angle = features['torso_angle']
    head_droop = features['head_droop_normalized']
    head_tilt = features['head_tilt']
    eye_conf = features['eye_conf']
    
    # SLEEPY: Head drooped + low eye confidence + hunched
    if head_droop > 0.15 and (eye_conf < 0.4 or torso_angle > 15):
        return 'sleepy'
    
    # YAWNING: Head tilted back
    elif head_droop < -0.1 and torso_angle < -10:
        return 'yawning'
    
    # BORED: Forward lean or slouched
    elif torso_angle > 20 or (head_droop > 0.08 and torso_angle > 10):
        return 'bored'
    
    # FRUSTRATED: Extreme head tilt
    elif head_tilt > 0.15:
        return 'frustrated'
    
    # CONFUSED: Moderate head tilt
    elif head_tilt > 0.08:
        return 'confused'
    
    # ENGAGED: Upright, aligned, alert
    elif abs(torso_angle) < 15 and head_droop < 0.05 and eye_conf > 0.5:
        return 'engaged'
    
    else:
        return 'engaged'
```

## Tunable Thresholds

Adjust these based on your camera angle and environment:

| Feature | Threshold | Description |
|---------|-----------|-------------|
| `head_droop` | 0.15 | Sleepy detection sensitivity |
| `torso_angle` | 20° | Bored/slouch detection |
| `head_tilt` | 0.15 | Frustrated detection |
| `eye_conf` | 0.4 | Eye closure threshold |

### Camera Angle Considerations:

**Frontal View (recommended):**
- Best for torso angle and head droop
- Current thresholds optimized for this

**Side View:**
- Increase torso_angle thresholds (25-30°)
- Head droop more reliable

**Elevated View:**
- Decrease head_droop thresholds (0.10-0.12)
- May need to adjust torso angle interpretation

## Real-Time Alert System

Alerts trigger only on **state transitions** (engaged/neutral → disengaged):

```python
# Example flow:
engaged → sleepy    # ✓ ALERT: "Student #1 appears SLEEPY"
sleepy → sleepy     # ✗ No alert (already notified)
sleepy → engaged    # ✗ No alert (positive change)
engaged → bored     # ✓ ALERT: "Student #1 appears BORED"
```

**Features:**
- 10-second cooldown per student (prevents spam)
- 5-second alert display duration
- Prominent red banner with warning symbol
- Console logging for tracking

## Testing

### Test Pose Features:
```bash
python test_pose_features.py
```
- Opens webcam
- Shows real-time posture classification
- Displays geometric features
- Press 'q' to quit

### Test Alert System:
```bash
python test_alert_system.py
```
- Validates alert triggering logic
- Tests state transitions
- Verifies cooldown mechanism

## Performance

**YOLO11n-pose:**
- Speed: ~50-100 FPS on GPU, ~10-20 FPS on CPU
- Accuracy: High for frontal/semi-frontal views
- Model size: ~6MB (lightweight)

**Recommendations:**
- Use GPU for real-time processing
- Reduce frame resolution if needed (640x480 works well)
- Consider temporal smoothing for noisy environments

## Future Improvements

### 1. ML Classifier (Optional)
Instead of hand-crafted rules, train a classifier:

```python
# Collect labeled data
features = []
labels = []

for frame in training_data:
    kpts = get_keypoints(frame)
    feat = get_pose_features(kpts)
    features.append(feat)
    labels.append(manual_label)  # 'engaged', 'bored', etc.

# Train classifier
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(features, labels)
```

### 2. Temporal Smoothing
Add moving average over 1-2 seconds:

```python
from collections import deque

posture_history = deque(maxlen=30)  # 1 sec at 30 FPS
posture_history.append(current_posture)
smoothed_posture = max(set(posture_history), key=posture_history.count)
```

### 3. Multi-Person Tracking
Current implementation uses first detected person. Extend to track multiple students:

```python
for i, kpts in enumerate(all_keypoints):
    features = get_pose_features(kpts)
    posture = classify_posture(features)
    # Associate with student ID via bounding box tracking
```

## Dependencies

```bash
pip install ultralytics opencv-python numpy
```

**Removed:**
- mediapipe (no longer needed)

## Files Modified

- `app.py` - Main application with YOLO11-pose integration
- `test_pose_features.py` - Test script for pose detection
- `test_alert_system.py` - Test script for alert logic

## Usage

1. **Start the app:**
   ```bash
   python app.py
   ```

2. **Navigate to Live Detection**

3. **Click "Start Detection"**
   - Face detection runs continuously
   - Pose detection analyzes body posture
   - Alerts trigger on disengagement

4. **Toggle Pose Detection** (optional)
   - Button to enable/disable pose analysis
   - Face detection continues independently

## Troubleshooting

**Model not found:**
- YOLO will auto-download yolo11n-pose.pt on first run
- Or manually download from Ultralytics

**Low accuracy:**
- Ensure good lighting
- Check camera angle (frontal is best)
- Adjust thresholds in `classify_posture()`

**Performance issues:**
- Reduce frame resolution
- Lower pose detection confidence threshold
- Use GPU if available

## References

- [YOLO11 Pose Documentation](https://docs.ultralytics.com/tasks/pose/)
- [COCO Keypoint Format](https://cocodataset.org/#keypoints-2020)
- Ultralytics YOLO11-pose pretrained models
