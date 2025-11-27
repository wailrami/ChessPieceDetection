from ultralytics import YOLO
import cv2 as cv
import numpy as np
import os


def get_board_corners(image):
    model_path = os.path.join(os.getcwd(), 'detection', 'models', 'board_seg.pt')
    model = YOLO(model_path)
    results = model(image)
    corners = None
    for result in results:
        for mask in result.masks.xy:
            mask_np = np.array(mask, np.int32).reshape((-1, 1, 2))

            # Approximate the polygon to get the corner points
            epsilon = 0.02 * cv.arcLength(mask_np, True)  # Adjust for more/less detail
            corners = cv.approxPolyDP(mask_np, epsilon, True)

            print(f"Corners: {corners.reshape(-1, 2).tolist()}")
            if len(corners) < 4:
                print("Not a rectangle")
                return False, None

            pt1 = corners[0][0]
            pt2 = corners[1][0]
            pt3 = corners[2][0]
            pt4 = corners[3][0]
            # corners = [[pt1], [pt2], [pt3], [pt4]]
            print(f"pt1: {pt1}, pt2: {pt2}, pt3: {pt3}, pt4: {pt4}")

        frame = result.plot()

    return True, corners


def get_board_box(image):
    model = YOLO('models/board_seg.pt')
    results = model(image)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            return x1, y1, x2, y2
    return None

