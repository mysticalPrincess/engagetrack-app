# Camera Error Fix Summary

## Issues Identified

### 1. OpenCV Matrix Assertion Error
```
error: (-215:Assertion failed) _step >= minstep in function 'cv::Mat::Mat'
```

**Cause:** Frame was being overwritten by `pose_results[0].plot()` after YOLO detection had already drawn on it, causing dimension/stride conflicts.

### 2. MSMF Stream Selection Warnings
```
WARN: global cap_msmf.cpp:930 CvCapture_MSMF::initStream Failed to select stream 0
```

**Cause:** Windows Media Foundation backend issues with camera initialization.

## Fixes Applied

### 1. Manual Skeleton Drawing
**Before:**
```python
frame = pose_results[0].plot()  # Overwrites frame!
```

**After:**
```python
# Draw keypoints and connections manually on existing frame
for i, (x, y, conf) in enumerate(pose_keypoints):
    if conf > 0.5:
        cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)

# Draw skeleton connections
connections = [(5, 6), (5, 7), ...]  # Shoulders, arms, torso, legs
for start_idx, end_idx in connections:
    cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
```

**Benefit:** No frame overwriting, preserves YOLO detection boxes.

### 2. DirectShow Backend (Windows)
**Added:**
```python
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
```

**Benefit:** More reliable camera access on Windows, reduces MSMF warnings.

### 3. Enhanced Frame Validation
**Added:**
```python
# Validate frame before processing
if not success or frame is None:
    time.sleep(0.1)
    continue

if frame.size == 0 or len(frame.shape) != 3:
    time.sleep(0.1)
    continue

if frame_height == 0 or frame_width == 0:
    continue
```

**Benefit:** Prevents processing invalid frames that cause assertion errors.

### 4. Camera Warm-up
**Added:**
```python
# Warm up camera after initialization
for _ in range(5):
    camera.read()
```

**Benefit:** Ensures camera is fully initialized before streaming.

### 5. Proper Camera Release
**Added:**
```python
if camera is not None:
    camera.release()
    time.sleep(0.5)  # Give time for camera to release
```

**Benefit:** Prevents conflicts when reinitializing camera.

## Testing

### Test Camera Directly:
```bash
python test_camera.py
```

**What it does:**
- Tests DirectShow backend
- Validates frame reading
- Displays FPS
- Shows live camera feed
- Press 'q' to quit

**Expected output:**
```
[Test 1] Trying DirectShow backend...
✓ Camera opened with DirectShow

[Test 2] Setting camera properties...
  Resolution: 640x480
  FPS: 30

[Test 3] Warming up camera...
  Frame 1: ✓ (480, 640, 3)
  ...

[Test 4] Reading frames...
Frames: 30 | FPS: 29.8
```

### Run Full App:
```bash
python app.py
```

Should now see:
```
✓ Loaded best.onnx successfully
✓ Loaded yolo11n-pose.pt successfully
✓ Camera initialized successfully
```

## Skeleton Visualization

The manual skeleton drawing includes:

**Keypoints (green circles):**
- Nose, eyes, ears
- Shoulders, elbows, wrists
- Hips, knees, ankles

**Connections (green lines):**
- Arms: shoulder → elbow → wrist
- Torso: shoulders ↔ hips
- Legs: hip → knee → ankle

**Confidence threshold:** Only draws keypoints with confidence > 0.5

## Performance Impact

**Before:** Frame overwriting caused crashes
**After:** Stable frame processing

**Drawing overhead:** ~1-2ms per frame (negligible)

## Troubleshooting

### If camera still fails:

1. **Check camera access:**
   ```bash
   python test_camera.py
   ```

2. **Try different camera index:**
   ```python
   camera = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # Try index 1, 2, etc.
   ```

3. **Check camera permissions:**
   - Windows Settings → Privacy → Camera
   - Ensure Python has camera access

4. **Close other apps using camera:**
   - Zoom, Teams, Skype, etc.

5. **Update OpenCV:**
   ```bash
   pip install --upgrade opencv-python
   ```

### If skeleton not showing:

- Check `pose_detection_active` is True
- Verify pose model loaded: "✓ Loaded yolo11n-pose.pt"
- Ensure person is visible in frame
- Check console for "Posture: ..." messages

## Files Modified

- `app.py` - Fixed frame handling and camera initialization
- `test_camera.py` (NEW) - Camera diagnostic tool
- `CAMERA_FIX_SUMMARY.md` (NEW) - This document

## Summary

✓ Fixed frame overwriting issue
✓ Added DirectShow backend for Windows
✓ Enhanced frame validation
✓ Added camera warm-up
✓ Manual skeleton drawing (no frame conflicts)
✓ Created camera test tool

The app should now run without camera errors!
