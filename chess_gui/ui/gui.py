import chess
import chess.svg
import chess.pgn
import chess.engine
import chess.polyglot
import chess.syzygy
import chess.variant
import cv2
import numpy as np


from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QPalette, QColor, QBrush, QPixmap, QImage, QPainter
from PyQt5.QtSvg import QSvgRenderer
from PyQt5.QtWidgets import (
    QWidget, QLabel, QGridLayout, QFrame, QSizePolicy,
    QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QApplication, QButtonGroup, QRadioButton
)
import sys
from detection import get_board_object_from_image
from utils import board_to_fen
from PyQt5.QtCore import QObject, QThread, pyqtSignal


class Worker(QThread):
    finished = pyqtSignal(bool, str)

    def __init__(self, image, mode):
        super(Worker, self).__init__()
        self.image = image
        self.mode = mode
        print("Worker initialized with image of shape:", self.image.shape)

    def run(self):
        """Long-running task."""
        print("Starting image processing...")
        ret, fen = generate_fen_from_image(self.image, self.mode)
        print("Image processing completed.")

        self.finished.emit(ret, fen)


def generate_fen_from_image(image, mode=0):
    ret, board_object = get_board_object_from_image(image, mode=mode)
    if not ret:
        return False, None
    return True, board_to_fen(board_object)


class Board(QWidget):
    def __init__(self, fen, parent=None):
        super(Board, self).__init__(parent)

        self.grid_layout = QGridLayout()
        self.grid_layout.setSpacing(0)
        self.grid_layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.grid_layout)

        self.board = chess.Board(fen)
        self.board_size = 8
        self.piece_size = 64
        self.squares = [[None for _ in range(self.board_size)] for _ in range(self.board_size)]

        self.setup_board()

    def setup_board(self):
        for row in range(self.board_size):
            for col in range(self.board_size):
                square = QLabel()
                square.setAlignment(Qt.AlignCenter)
                square.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                square.setMinimumSize(QSize(self.piece_size, self.piece_size))
                square.setStyleSheet('background-color: transparent;')
                self.grid_layout.addWidget(square, row, col)
                self.squares[row][col] = square

        self.update_board()

    def update_board(self):
        for row in range(self.board_size):
            for col in range(self.board_size):
                square = self.squares[row][col]
                square.clear()

                if (row + col) % 2 == 0:
                    square.setStyleSheet('background-color: #f0d9b5;')
                else:
                    square.setStyleSheet('background-color: #b58863;')

                piece = self.board.piece_at(chess.square(col, 7 - row))

                if piece is not None:
                    piece_svg = chess.svg.piece(piece)
                    svg_renderer = QSvgRenderer()
                    svg_renderer.load(bytearray(piece_svg, encoding='utf-8'))
                    image = QImage(self.piece_size, self.piece_size, QImage.Format_ARGB32)
                    image.fill(0)
                    painter = QPainter(image)
                    svg_renderer.render(painter)
                    painter.end()
                    pixmap = QPixmap.fromImage(image)
                    square.setPixmap(pixmap)

    def set_board(self, board):
        self.board = board
        self.update_board()

    # def resizeEvent(self, event):
    #     size = max(self.width(), self.height())
    #     self.setFixedSize(size, size)
    #     super().resizeEvent(event)

    def update_board_fen(self, fen):
        """
        Update the board based on a new FEN string.
        """
        self.board.set_fen(fen)
        self.update_board()


class ChessApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Chess Piece Detection GUI")
        self.setGeometry(100, 100, 1200, 600)
        self.setup_ui()

    def setup_ui(self):
        layout = QHBoxLayout(self)

        # Left: Optional image viewer
        self.image_label = QLabel("Image Preview")
        self.image_label.setFixedSize(500, 500)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid gray;")
        layout.addWidget(self.image_label)

        # Center: Chess board

        self.board_widget = Board(fen="8/8/8/8/8/8/8/8 w - - 0 1")
        board_container = QWidget()
        board_layout = QVBoxLayout()
        board_layout.setContentsMargins(0, 0, 0, 0)
        board_layout.addWidget(self.board_widget, alignment=Qt.AlignCenter)
        board_container.setLayout(board_layout)

        layout.addWidget(board_container)

        # Right: Options panel
        self.right_panel = QVBoxLayout()

        btn_open = QPushButton("Open File")
        btn_open.clicked.connect(self.open_file)
        self.right_panel.addWidget(btn_open)

        # Add other buttons
        self.btn_predict = QPushButton("Predict")
        self.btn_predict.clicked.connect(self.predict_board)
        # btn_predict.setEnabled(False)  # Initially disabled until an image is loaded
        self.right_panel.addWidget(self.btn_predict)

        # adding group of three radio buttons
        self.radio_group = QButtonGroup(self)
        self.radio_2d = QRadioButton("2D Detection (YOLO)")
        self.radio_3d = QRadioButton("3D Detection (YOLO)")
        self.radio_3d_class = QRadioButton("3D Detection (YOLO + Classifier)")

        self.radio_group.addButton(self.radio_2d)
        self.radio_group.addButton(self.radio_3d)
        self.radio_group.addButton(self.radio_3d_class)

        self.right_panel.addWidget(self.radio_2d)
        self.right_panel.addWidget(self.radio_3d)
        self.right_panel.addWidget(self.radio_3d_class)

        # Set default selection
        self.radio_2d.setChecked(True)

        self.right_panel.addStretch()
        layout.addLayout(self.right_panel)

    def open_file(self):
        self.file_path, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.jpg *.jpeg)")
        if self.file_path:
            pixmap = QPixmap(self.file_path).scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            # get img path from pixmap

            self.image_label.setPixmap(pixmap)

    def predict_board(self):
        # Placeholder for prediction logic
        # This function should call the detection logic and update the board
        global mode
        pixmap = self.image_label.pixmap()

        if pixmap:
            print("Image collected from label.")
            if self.file_path:
                print("File path:", self.file_path)
                image = cv2.imread(self.file_path)
            else:
                qimage = pixmap.toImage().convertToFormat(QImage.Format_RGB888)

                width = qimage.width()
                height = qimage.height()
                ptr = qimage.bits()
                ptr.setsize(height * width * 3)  # 4 bytes per pixel (RGBA)
                image = np.array(ptr).reshape((height, width, 3))  # RGBA image
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert to RGB format

            print("Image shape:", image.shape)
            print("Image type:", type(image))
            if self.radio_2d.isChecked():
                mode = 0
            elif self.radio_3d.isChecked():
                mode = 1
            elif self.radio_3d_class.isChecked():
                mode = 2
            self.btn_predict.setEnabled(False)
            self.worker = Worker(image, mode)
            self.worker.finished.connect(self.on_prediction_complete)
            self.worker.start()

    def on_prediction_complete(self, success, fen):
        if success:
            print("Prediction successful. FEN:", fen)
            self.board_widget.update_board_fen(fen)
        else:
            print("Prediction failed.")

        self.btn_predict.setEnabled(True)





# if __name__ == '__main__':
#     import sys
#     from PyQt5.QtWidgets import QApplication
#
#     app = QApplication(sys.argv)
#     board = ChessApp()
#     board.show()
#     sys.exit(app.exec_())











# def mousePressEvent(self, event):
    #     if event.button() == Qt.LeftButton:
    #         row = self.grid_layout.rowCount()
    #         col = self.grid_layout.columnCount()
    #
    #         for i in range(row):
    #             for j in range(col):
    #                 square = self.grid_layout.itemAtPosition(i, j).widget()
    #                 if square.geometry().contains(event.pos()):
    #                     square.setStyleSheet('background-color: #f0d9b5;')
    #                     self.selected_square = square
    #                     self.selected_square_pos = (i, j)
    #
    #     super(Board, self).mousePressEvent(event)
    #
    # def mouseReleaseEvent(self, event):
    #     if event.button() == Qt.LeftButton:
    #         row = self.grid_layout.rowCount()
    #         col = self.grid_layout.columnCount()
    #
    #         for i in range(row):
    #             for j in range(col):
    #                 square = self.grid_layout.itemAtPosition(i, j).widget()
    #                 if square.geometry().contains(event.pos()):
    #                     square.setStyleSheet('background-color: transparent;')
    #                     self.selected_square.setStyleSheet('background-color: transparent;')
    #                     self.selected_square = None
    #                     self.selected_square_pos = None
    #
    #     super(Board, self).mouseReleaseEvent(event)
