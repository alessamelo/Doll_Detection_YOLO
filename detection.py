
import numpy as np
import math
import cv2
from ultralytics import YOLO
from matplotlib import pyplot as plt

capture = cv2.VideoCapture(0)  # Open the default camera

model = YOLO("best.pt")

# Get the ACTUAL class names from your trained model
classNames = model.names  # This gets ['toothpaste'] from your custom model
print("Model classes:", classNames)

# Set the image width and heig<ht
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Image width
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Image height

# Confidence threshold (adjust as needed)
confidence_threshold = 0.999

print("Starting webcam detection... Press 'q' to quit")

# Start a loop to process camera frames
while True:
    success, img = capture.read()  # Capture a frame
    if not success:
        print("Failed to grab frame")
        break

    # Perform object detection
    results = model(img, stream=True)  # Use stream=True for real-time

    # Process the detection results
    for r in results:
        boxes = r.boxes
        
        if boxes is not None:  # Check if any detections
            # Iterate over the detected bounding boxes
            for box in boxes:
                # Filter by confidence
                confidence = box.conf.item()
                if confidence < confidence_threshold:
                    continue
                
                # Get the bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Draw the bounding box on the image
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)

                # Get class information
                cls = int(box.cls[0])
                class_name = classNames[cls]

                # Display confidence and class name
                label = f"{class_name} {confidence:.2f}"
                
                # Text background for better visibility
                (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(img, (x1, y1-text_height-5), (x1+text_width, y1), (255, 0, 255), -1)
                
                # Display the text
                cv2.putText(img, label, (x1, y1-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                print(f"Detected: {class_name} with confidence: {confidence:.2f}")

    # Display the image
    cv2.imshow('Glasses Detection', img)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
capture.release()
cv2.destroyAllWindows()
