# ============================================
# Doll Detection with YOLO11 and OpenCV
# Compatible with:
# ultralytics>=8.3.0
# torch==2.5.1
# opencv-python==4.10.0.84
# numpy==1.26.4
# ============================================

from ultralytics import YOLO
import cv2
import math
import os

# === Path to your trained model ===
model_path = YOLO("best.pt")

# === Check if model file exists ===
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model file not found: {model_path}")

# === Load YOLO model ===
print("🧠 Loading YOLO model...")
model = YOLO(model_path)
classNames = model.names
print(" Loaded custom classes:", classNames)

# === Initialize camera ===
capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not capture.isOpened():
    raise RuntimeError("Could not access the webcam. Check camera permissions or device index.")

print("🎥 Press 'q' to quit the camera window.")

# === Real-time detection loop ===
while True:
    success, frame = capture.read()
    if not success:
        print(" Failed to read frame from webcam.")
        break

    # Run YOLO inference
    results = model(frame, conf=0.6, verbose=False)

    # Draw detections
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            label = f"{classNames[cls]} {conf:.2f}"

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Display the frame
    cv2.imshow("Custom YOLO Doll Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("🛑 Camera stopped by user.")
        break

# === Clean up ===
capture.release()
cv2.destroyAllWindows()
print(" Camera released and all windows closed.")

