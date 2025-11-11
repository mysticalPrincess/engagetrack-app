"""
Simpler test - just capture one frame and test detection
"""
import cv2
from ultralytics import YOLO
import os

# Load model
model = YOLO('best.onnx', task='detect')
print(f"Model classes: {model.names}")

# Capture one frame
camera = cv2.VideoCapture(0)
ret, frame = camera.read()
camera.release()

if not ret:
    print("Failed to capture frame")
    exit(1)

print(f"Frame shape: {frame.shape}")

# Try detection with very low confidence
print("\nTrying detection with conf=0.05...")
results = model(frame, conf=0.05, verbose=True)

for result in results:
    boxes = result.boxes
    if boxes is not None and len(boxes) > 0:
        print(f"\n✓ Found {len(boxes)} detections!")
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            print(f"  Detection {i+1}: {class_name} ({confidence:.3f}) at [{x1},{y1},{x2},{y2}]")
            
            # Draw on frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name} {confidence:.2f}", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    else:
        print("\n✗ No detections found")
        print("This might mean:")
        print("  1. No face in frame")
        print("  2. Face too small/far")
        print("  3. Model trained on different image sizes")
        print("  4. Model expects preprocessed images")

# Save test image
cv2.imwrite('test_detection.jpg', frame)
print(f"\nSaved test image to: test_detection.jpg")
print("Check this image to see if your face is visible")
