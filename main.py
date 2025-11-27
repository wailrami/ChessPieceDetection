from chess_gui.ui import ChessApp
from detection import get_board_object_from_image
from utils import board_to_fen
import cv2


def generate_fen_from_image(image):
    ret, board_object = get_board_object_from_image(image)
    if not ret:
        return False, None
    return True, board_to_fen(board_object)


if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    chess_app = ChessApp()
    chess_app.show()




    # image = cv2.imread('detection/images/mychess2.png')
    # # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # # image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    #
    # ret, fen = generate_fen_from_image(image)
    # if not ret:
    #     print('Failed to generate FEN string.')
    #     sys.exit()
    #
    # board = Board(fen)
    # board.show()
    sys.exit(app.exec_())
