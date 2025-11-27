from ultralytics import YOLO
import cv2 as cv
import numpy as np

model = YOLO('models/board_seg.pt')
model2 = YOLO('models/pieces_best_s_2.pt')

image = cv.imread('images/img_1.png')

image = cv.resize(image, (640, 640))

# image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
results = model(image)
# image = cv.resize(image, (720,640))
def calculate_angle(center, pt1, pt2):
    # Create vectors
    vector1 = np.array(pt1) - np.array(center)
    vector2 = np.array(pt2) - np.array(center)

    # Calculate dot product and magnitudes
    dot_product = np.dot(vector1, vector2)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)

    # Calculate angle in radians and convert to degrees
    angle = np.arccos(dot_product / (magnitude1 * magnitude2))
    return np.degrees(angle)

for result in results:
    # print(result)
    for box in result.boxes:
        # center of box
        center = box.xywh[0]
        center = center.cpu().numpy().astype(int)
        center = center[:2]
        print("center :", center)
        x1, y1, x2, y2 = box.xyxy[0]
        # cv.circle(image, (center[0], center[1]), 5, (0, 0, 255), -1)
        # cv.putText(image, 'center box', (center[0] + 10, center[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv.LINE_AA)

    for mask in result.masks.xy:
        mask_np = np.array(mask, np.int32).reshape((-1, 1, 2))

        # Approximate the polygon to get the corner points
        epsilon = 0.02 * cv.arcLength(mask_np, True)  # Adjust for more/less detail
        corners = cv.approxPolyDP(mask_np, epsilon, True)

        print(f"Corners: {corners.reshape(-1, 2).tolist()}")

        pt1 = corners[0][0]
        pt2 = corners[1][0]
        pt3 = corners[2][0]
        pt4 = corners[3][0]
        # Points p1, p2, p3, p4
        points = [pt1, pt2, pt3, pt4]

        # Calculate the center
        center_x = int(sum(point[0] for point in points) / len(points))
        center_y = int(sum(point[1] for point in points) / len(points))
        board_center = (center_x, center_y)

        # Draw the center point and label
        # cv.circle(image, (center_x, center_y), 5, (0, 0, 255), -1)
        # cv.putText(image, 'center board', (center_x + 10, center_y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1,
        #            cv.LINE_AA)
        angles = {
            "pt1-pt2": calculate_angle(board_center, pt1, pt2),
            "pt1-pt4": calculate_angle(board_center, pt1, pt4),
            "pt2-pt3": calculate_angle(board_center, pt2, pt3),
            "pt3-pt4": calculate_angle(board_center, pt3, pt4),
        }

        print(f"pt1: {pt1}, pt2: {pt2}, pt3: {pt3}, pt4: {pt4}")
        for pair, angle in angles.items():
            print(f"Angle between {pair}: {angle:.2f} degrees")
        # Draw corners on the image
        for i, (x, y) in enumerate(corners.reshape(-1, 2)):
            # Draw red dots for corners
            cv.circle(image, (x, y), 5, (0, 0, 255), -1)
            # Add labels (p1, p2, p3, p4)
            cv.putText(image, f'p{i + 1}', (x + 10, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv.LINE_AA)

    annotated_frame = result.plot()
cv.imshow('YOLO', annotated_frame)
cv.waitKey(0)

# perspective transform
pts1 = np.float32([[x1, y1], [x2, y1], [x1, y2], [x2, y2]])
pts1 = np.float32([pt1, pt4, pt2, pt3])
pts2 = np.float32([[0, 0], [720, 0], [0, 640], [720, 640]])
matrix = cv.getPerspectiveTransform(pts1, pts2)
img_crop = cv.warpPerspective(image, matrix, (720, 640))
#
# img_crop = cv.cvtColor(img_crop, cv.COLOR_BGR2GRAY)
# img_crop = cv.GaussianBlur(img_crop, (5, 5), 0)
# adaptive = cv.Canny(img_crop, 100, 200, apertureSize=5, L2gradient=True)
# # adaptive = cv.adaptiveThreshold(img_crop, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
# cv.imshow('Canny', adaptive)
# cv.waitKey(0)

# img_crop = cv.cvtColor(img_crop, cv.COLOR_GRAY2BGR)

# lines = cv.HoughLines(adaptive, 1, np.pi / 180, 200)
# for line in lines:
#     rho, theta = line[0]
#     a = np.cos(theta)
#     b = np.sin(theta)
#     x0 = a * rho
#     y0 = b * rho
#     x1 = int(x0 + 1000 * -b)
#     y1 = int(y0 + 1000 * a)
#     x2 = int(x0 - 1000 * -b)
#     y2 = int(y0 - 1000 * a)
#
#     cv.line(img_crop, (x1, y1), (x2, y2), (0, 255, 0), 2)
#
# cv.imshow('Chessboard Lines', img_crop)
# cv.waitKey(0)
#
# ret, corners = cv.findChessboardCorners(img_crop, (7, 7), None)
# print(ret)
# if ret:
#     cv.drawChessboardCorners(img_crop, (7, 7), corners, ret)


img_crop = cv.resize(img_crop, (640, 640))

for i in range(8):
    p = i * 640 / 8
    cv.line(img_crop, (int(p), 0), (int(p), 640), (0, 255, 0), 2)
    cv.line(img_crop, (0, int(p)), (640, int(p)), (0, 255, 0), 2)

cv.imshow('Chessboard', img_crop)
cv.waitKey(0)

res = model2.predict(image, conf=0.5)
print("RESULTS:\n", res)
classes = res[0].names
for r in res:
    for box in r.boxes:
        # print(int(box.cls), " ", classes[int(box.cls)], " ", box.xyxy)
        print(box)
    image = r.plot(labels=False)

cv.imshow("Chess Pieces", image)
cv.waitKey(0)

# img_crop = np.array(image[int(y1):int(y2), int(x1):int(x2)])
# img_crop = cv.resize(img_crop, (720, 640))

# gray = cv.cvtColor(img_crop, cv.COLOR_BGR2GRAY)
# img = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 5, 3)
# # find contours
# contours, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
#
# cn = cv.drawContours(img_crop, contours, -1, (0, 255, 0), 2)
#
# cv.imshow('Chessboard', cn)
#
# cv.imshow('Adaptive Threshold', img)
# cv.waitKey(0)

# search good features (corners)
# corners = cv.goodFeaturesToTrack(gray, 100, 0.01, 10)
# corners = np.int0(corners)
# for corner in corners:
#     x, y = corner.ravel()
#     cv.circle(img_crop, (x, y), 5, 255, -1)
# cv.imshow('Chessboard', img_crop)
# cv.waitKey(0)
