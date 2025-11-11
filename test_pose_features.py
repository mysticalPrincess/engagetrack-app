import cv2
import numpy as np
from ultralytics import YOLO

# Load pretrained pose model (downloads if needed)
model = YOLO("daise\yolo11n-pose.pt")

# simple rule-based engagement classifier
def classify_engagement(kpts):
    # kpts: (17,3) array: x,y,conf
    if kpts is None: 
        return "No person"
    # keypoint indices (COCO)
    L_sh, R_sh = kpts[5][:2], kpts[6][:2]
    L_hp, R_hp = kpts[11][:2], kpts[12][:2]
    nose = kpts[0][:2]
    sh_mid = (L_sh + R_sh) / 2
    hip_mid = (L_hp + R_hp) / 2
    torso_vec = sh_mid - hip_mid
    # torso forward lean: positive x means forward (camera-left/right depends on view)
    torso_angle = np.degrees(np.arctan2(torso_vec[0], torso_vec[1]))
    # head droop: nose lower than shoulder midpoint (y increases downwards)
    head_droop = nose[1] - sh_mid[1]
    # eye confidence as a proxy for open eyes
    eye_conf = (kpts[1][2] + kpts[2][2]) / 2

    if head_droop > 30 or eye_conf < 0.2:
        return "Sleepy/Bored"
    if abs(torso_angle) > 15:
        return "Hunched"
    return "Upright"

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Webcam not found")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # run pose inference (returns list[Results])
    results = model(frame, imgsz=640, conf=0.25)  # adjust conf/imgsz as needed
    label = "No person"
    for r in results:
        # r.keypoints.data shape: (N,17,3) if multiple persons; take first person
        kpts_data = None
        if r.keypoints is not None and len(r.keypoints.data):
            kpts_data = r.keypoints.data[0]  # (17,3) numpy
            # draw keypoints
            for x,y,c in kpts_data:
                if c > 0.2:
                    cv2.circle(frame, (int(x), int(y)), 3, (0,255,0), -1)
            label = classify_engagement(kpts_data)

    # overlay label
    cv2.putText(frame, f"Engagement: {label}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
    cv2.imshow("YOLO Pose Webcam", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()