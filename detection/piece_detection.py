from ultralytics import YOLO
import os
import cv2


def get_all_pieces(original_img, mode=0):
    model_path = os.path.join(os.getcwd(), 'detection', 'models', 'best_2d.pt' if mode == 0 else 'pieces_best_s_2.pt')
    model = YOLO(model_path)
    results = model.predict(original_img)

    if results is None:
        return False, None
    # getting all the bounding boxes
    classes = results[0].names

    pieces = {}
    i = 0  # id for each piece (avoiding duplicates)
    frame = original_img.copy()
    for res in results:
        for box in res.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            piece = classes[int(box.cls)]
            pieces[(piece, i)] = (x1, y1, x2, y2)
            i += 1
        frame = res.plot()
    frame = cv2.resize(frame, (640, 640))
    # cv2.imshow('frame', frame)
    # cv2.waitKey(0)
    return True, pieces
