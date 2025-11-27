import cv2

from .mapper import *


def get_board_object_from_image(image, mode=0):
    board = np.zeros((8, 8), dtype=object)
    ret, mapped_pieces = map_pieces_to_board(image, detection_mode=mode)
    if not ret:
        return False, None

    board = assign_pieces_to_squares(mapped_pieces)
    return True, board
