"""
Test camera initialization and frame reading
Helps diagnose camera issues before running the full app
"""

import cv2
import time

print("=" * 60)
print("Camera Test Script")
print("=" * 60)

# Test 1: Try DirectShow (Windows)
print("\n[Test 1] Trying DirectShow backend...")
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if camera.isOpened():
    print("✓ Camera opened with DirectShow")
else:
    print("✗ DirectShow failed, trying default backend...")
    camera = cv2.VideoCapture(0)
    if camera.isOpened():
        print("✓ Camera opened with default backend")
    else:
        print("✗ Cannot open camera!")
        exit(1)

# Test 2: Set properties
print("\n[Test 2] Setting camera properties...")
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
camera.set(cv2.CAP_PROP_FPS, 30)
camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

actual_width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
actual_height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
actual_fps = camera.get(cv2.CAP_PROP_FPS)

print(f"  Resolution: {int(actual_width)}x{int(actual_height)}")
print(f"  FPS: {int(actual_fps)}")

# Test 3: Warm up camera
print("\n[Test 3] Warming up camera (reading 5 frames)...")
for i in range(5):
    ret, frame = camera.read()
    if ret:
        print(f"  Frame {i+1}: ✓ {frame.shape}")
    else:
        print(f"  Frame {i+1}: ✗ Failed")

# Test 4: Read and display frames
print("\n[Test 4] Reading frames (Press 'q' to quit)...")
print("-" * 60)

frame_count = 0
start_time = time.time()

try:
    while True:
        ret, frame = camera.read()
        
        if not ret or frame is None:
            print("Failed to read frame")
            time.sleep(0.1)
            continue
        
        if frame.size == 0 or len(frame.shape) != 3:
            print("Invalid frame")
            continue
        
        frame_count += 1
        
        # Calculate FPS
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed
            print(f"Frames: {frame_count} | FPS: {fps:.1f}")
        
        # Display frame
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'q' to quit", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Camera Test', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nInterrupted by user")
except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
finally:
    camera.release()
    cv2.destroyAllWindows()
    
    elapsed = time.time() - start_time
    avg_fps = frame_count / elapsed if elapsed > 0 else 0
    
    print("\n" + "=" * 60)
    print("Test Summary:")
    print(f"  Total frames: {frame_count}")
    print(f"  Duration: {elapsed:.1f}s")
    print(f"  Average FPS: {avg_fps:.1f}")
    print("✓ Camera test completed")
    print("=" * 60)
