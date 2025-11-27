import cv2
from ultralytics import YOLO
from skimage.metrics import structural_similarity as compare_ssim
import numpy as np


# Assign custom colors per class ID (extend as needed)
CLASS_COLORS = {
    0: (0, 255, 0),      # green
    1: (255, 0, 0),      # blue
    2: (0, 0, 255),      # red
    3: (255, 255, 0),    # cyan
    4: (255, 0, 255),    # magenta
    5: (0, 255, 255),    # yellow
    6: (128, 0, 128),    # purple
    7: (0, 128, 255),    # orange
    8: (128, 128, 128),  # gray
    9: (255, 255, 255),  # white
}


def draw_boxes_with_labels(frame, results):
    """
    Draws bounding boxes with class labels and custom colors on the frame.
    """
    for result in results:
        boxes = result.boxes
        names = result.names  # dict: class_id -> name

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            label = names[class_id]
            color = CLASS_COLORS.get(class_id, (255, 255, 255))  # white if not found

            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw label background
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_y = y1 - 10 if y1 - 10 > 10 else y1 + 10
            cv2.rectangle(frame, (x1, label_y - label_size[1]), (x1 + label_size[0], label_y), color, -1)

            # Put text
            cv2.putText(frame, label, (x1, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return frame


def stream_camera():
    cap = cv2.VideoCapture("http://192.168.224.190:4747/video")
    model = YOLO("models/pieces_best_s_2.pt")

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    prev_gray = None
    last_results = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is None:
            prev_gray = gray
            continue

        score, _ = compare_ssim(prev_gray, gray, full=True)

        if score < 0.90:
            print(f"Significant change detected (SSIM={score:.2f}) - Running detection")
            results = model.track(frame)
            last_results = results
        else:
            print(f"No significant change (SSIM={score:.2f}) - Reusing previous boxes")

        prev_gray = gray.copy()

        # Draw previous detections on current live frame
        if last_results:
            frame = draw_boxes_with_labels(frame, last_results)

        cv2.imshow('Camera Stream', cv2.resize(frame, (640, 480)))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    stream_camera()
