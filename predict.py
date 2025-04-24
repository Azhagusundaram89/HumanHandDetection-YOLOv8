from ultralytics import YOLO
import cv2

# Load the custom YOLO model (replace with your own .pt path if needed)
model = YOLO(r"D:\hand detection\runs\detect\train2\weights\best.pt")  # or 'yolov8n.pt' etc.

# Open webcam (0 = default cam)
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Process frames from the webcam
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference on the frame
    results = model.predict(source=frame, conf=0.25, stream=False)

    # Plot the results on the frame
    annotated_frame = results[0].plot()

    # Display the frame
    cv2.imshow("YOLOv8 Webcam Detection", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
