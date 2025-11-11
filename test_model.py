"""
Quick test script to verify YOLO model detection
"""
import cv2
from ultralytics import YOLO
import os

# Load model
if os.path.exists('best.onnx'):
    model = YOLO('best.onnx', task='detect')
    print("✓ Loaded best.onnx model")
elif os.path.exists('engagement_detection_model.onnx'):
    model = YOLO('engagement_detection_model.onnx', task='detect')
    print("✓ Loaded engagement_detection_model.onnx")
else:
    print("✗ No model found!")
    exit(1)

print(f"Model classes: {model.names}")

# Open webcam
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("✗ Cannot open camera")
    exit(1)

print("✓ Camera opened")
print("\nTesting detection with different confidence thresholds...")
print("Press 'q' to quit\n")

frame_count = 0
while True:
    ret, frame = camera.read()
    if not ret:
        break
    
    frame_count += 1
    
    # Test with very low confidence every 30 frames
    if frame_count % 30 == 0:
        for conf_threshold in [0.1, 0.2, 0.3, 0.5]:
            results = model(frame, conf=conf_threshold, verbose=False)
            
            detection_count = 0
            for result in results:
                if result.boxes is not None:
                    detection_count = len(result.boxes)
                    if detection_count > 0:
                        print(f"Conf {conf_threshold}: Found {detection_count} detections")
                        for box in result.boxes:
                            class_id = int(box.cls[0])
                            confidence = float(box.conf[0])
                            class_name = model.names[class_id]
                            print(f"  - {class_name}: {confidence:.3f}")
            
            if detection_count == 0:
                print(f"Conf {conf_threshold}: No detections")
        print("-" * 50)
    
    # Display frame
    cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow('Model Test', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
print("\nTest complete!")
