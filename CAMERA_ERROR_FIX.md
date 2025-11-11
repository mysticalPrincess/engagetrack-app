# Camera Error Fix

## Error Fixed

```
cv2.error: OpenCV(4.11.0) error: (-215:Assertion failed) _step >= minstep
```

This error occurs when the camera is accessed improperly or the frame data is corrupted.

## Changes Made

### 1. Camera Initialization Check
```python
if camera is None or not camera.isOpened():
    if camera is not None:
        camera.release()  # Release if exists but not opened
    
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error: Cannot open camera")
        return
```

### 2. Frame Validation
```python
success, frame = camera.read()
if not success or frame is None:
    print("Failed to read frame from camera")
    break

# Validate frame
if frame.size == 0:
    print("Empty frame received")
    continue
```

### 3. Error Handling in Encoding
```python
try:
    ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ret:
        print("Failed to encode frame")
        continue
    frame_bytes = buffer.tobytes()
    yield frame_bytes
except Exception as e:
    print(f"Error encoding frame: {e}")
    continue
```

### 4. Proper Cleanup
```python
def cleanup_camera():
    """Release camera resources on shutdown"""
    global camera
    if camera is not None:
        try:
            camera.release()
            print("Camera released successfully")
        except Exception as e:
            print(f"Error releasing camera: {e}")

# Register cleanup on app exit
atexit.register(cleanup_camera)
```

## Why This Happens

1. **Multiple Access** - Camera accessed by multiple processes
2. **Improper Release** - Camera not released after previous use
3. **Corrupted Frames** - Invalid frame data from camera
4. **Hot Reload** - Flask debug mode restarting while camera is open

## Prevention

✅ **Check camera state** before opening
✅ **Validate frames** before processing
✅ **Handle encoding errors** gracefully
✅ **Release camera** on shutdown
✅ **Try-except blocks** around critical operations

## If Error Persists

1. **Restart the app** - Ctrl+C and run again
2. **Check camera access** - Close other apps using camera (Zoom, Teams, etc.)
3. **Disable debug mode** - Change `debug=True` to `debug=False` to prevent hot reloads
4. **Restart computer** - If camera is locked by another process

## Testing

After these changes:
- Camera errors are caught and logged
- Invalid frames are skipped
- App continues running even if a frame fails
- Camera is properly released on shutdown

## Debug Output

You'll now see helpful messages:
- "Failed to read frame from camera"
- "Empty frame received"
- "Failed to encode frame"
- "Camera released successfully"

These help identify exactly where issues occur.
