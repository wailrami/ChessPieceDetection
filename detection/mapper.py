import cv2
import numpy as np
import cv2 as cv

from .piece_detection import get_all_pieces
from .board_detection import get_board_corners
from .piece_classifier import classify_all_pieces

img_cropped_size = 640


def distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def assign_to_corner(points, corner='tl', img_shape=(img_cropped_size, img_cropped_size)):
    ref_point = None
    returned_point = None
    if corner == 'tl':
        # Get the top left corner
        ref_point = (0, 0)
    elif corner == 'tr':
        # Get the top right corner
        ref_point = (img_shape[0], 0)
    elif corner == 'bl':
        # Get the bottom left corner
        ref_point = (0, img_shape[1])
    elif corner == 'br':
        # Get the bottom right corner
        ref_point = (img_shape[0], img_shape[1])

    # calculate the distance between points and ref_point and return the min one
    min_distance = float('inf')
    for point in points:
        dist = distance(point, ref_point)
        if dist < min_distance:
            min_distance = dist
            returned_point = point
        print(f"point: {point}, ref_point: {ref_point}, distance: {dist}")
    return returned_point


def warp_point(point, corners):
    pt1 = corners[0]
    pt2 = corners[1]
    pt3 = corners[2]
    pt4 = corners[3]

    # perspective transform
    pts1 = np.float32([pt1, pt2, pt3, pt4])
    pts2 = np.float32([[0, 0], [img_cropped_size, 0], [0, img_cropped_size], [img_cropped_size, img_cropped_size]])
    matrix = cv.getPerspectiveTransform(pts1, pts2)
    point = np.array([[point]], dtype='float32')
    warped_point = cv.perspectiveTransform(point, matrix)
    print(f"point: {point}, warped_point: {warped_point}")
    return warped_point[0][0]


def map_pieces_to_board(image, detection_mode=0):
    # Get the board corners
    success, corners = get_board_corners(image)
    if not success:
        return False, None

    # Get the board corners
    p1 = corners[0][0]
    p2 = corners[1][0]
    p3 = corners[2][0]
    p4 = corners[3][0]

    # assign pt1 -> top left, pt2 -> top right, pt3 -> bottom left, pt4 -> bottom right
    extracted_corners = [p1, p2, p3, p4]
    pt1 = assign_to_corner(extracted_corners, 'tl', img_shape=image.shape)

    extracted_corners = [pt for pt in extracted_corners if not np.array_equal(pt, pt1)]
    pt2 = assign_to_corner(extracted_corners, 'tr', img_shape=image.shape)

    extracted_corners = [pt for pt in extracted_corners if not np.array_equal(pt, pt2)]
    pt3 = assign_to_corner(extracted_corners, 'bl', img_shape=image.shape)

    extracted_corners = [pt for pt in extracted_corners if not np.array_equal(pt, pt3)]
    pt4 = assign_to_corner(extracted_corners, 'br', img_shape=image.shape)

    print(f"pt1: {pt1}, pt2: {pt2}, pt3: {pt3}, pt4: {pt4}")

    frame = image.copy()
    print("Drawing corners on the image...")
    i = 1
    for (x, y) in [pt1, pt2, pt3, pt4]:
        cv.circle(frame, (x, y), 5, (0, 0, 255), -1)
        cv.putText(frame, f'pt{i}', (x, y), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
        i += 1

    frame = cv.resize(frame, (640, 640))

    print("Showing the image with corners...")
    # cv.imshow('frame', frame)
    # cv.waitKey(0)
    print("After Showing the image with corners...")
    # Get the piece bounding boxes
    success, pieces = get_all_pieces(image, mode=detection_mode)
    if not success:
        return False, None

    if detection_mode == 2:
        pieces = classify_all_pieces(pieces, image)

    # mapping
    mapped_pieces = {}  # piece -> (x, y) in the board
    for (piece, i), (x1, y1, x2, y2) in pieces.items():
        bottom_center = (x1 + (x2 - x1) // 2, y2 - (y2 - y1) // 4)

        warped_bottom_center = warp_point(bottom_center, [pt1, pt2, pt3, pt4])
        # Check if the warped point is inside the board
        if warped_bottom_center[0] < 0 or warped_bottom_center[0] > img_cropped_size or \
                warped_bottom_center[1] < 0 or warped_bottom_center[1] > img_cropped_size:
            print(f"Invalid point: {warped_bottom_center}")
            continue
        mapped_pieces[(piece, i)] = (warped_bottom_center[0], warped_bottom_center[1])

    pts1 = np.float32([pt1, pt2, pt3, pt4])
    pts2 = np.float32([[0, 0], [img_cropped_size, 0], [0, img_cropped_size], [img_cropped_size, img_cropped_size]])
    matrix = cv.getPerspectiveTransform(pts1, pts2)

    img_crop = cv.warpPerspective(image, matrix, (img_cropped_size, img_cropped_size))
    # draw circles of the pieces on the image
    for (piece, i), (x, y) in mapped_pieces.items():
        cv.circle(img_crop, (int(x), int(y)), 5, (0, 0, 255), -1)
        cv.putText(img_crop, f'{piece}', (int(x), int(y)), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
    for i in range(8):
        p = i * 640 / 8
        cv.line(img_crop, (int(p), 0), (int(p), 640), (0, 255, 0), 2)
        cv.line(img_crop, (0, int(p)), (640, int(p)), (0, 255, 0), 2)
    # cv2.imshow('frametespiece', img_crop)
    # cv2.waitKey(0)

    return True, mapped_pieces


def assign_pieces_to_squares(mapped_pieces):
    board = np.zeros((8, 8), dtype=object)
    for (piece, i), (x, y) in mapped_pieces.items():
        row = int(y / (img_cropped_size / 8))
        col = int(x / (img_cropped_size / 8))
        print(f"piece: {piece}, i: {i}, x: {x}, y: {y}, row: {row}, col: {col}")
        if row < 0 or row > 7 or col < 0 or col > 7:
            print(f"Invalid row or col: {row}, {col}")
            continue
        board[row, col] = piece

    return board
