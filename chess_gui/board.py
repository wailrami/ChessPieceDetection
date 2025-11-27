import chess
import chess.svg
import chess.pgn
import chess.engine
import chess.polyglot
import chess.syzygy
import chess.variant

from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QPalette, QColor, QBrush, QPixmap, QImage, QPainter
from PyQt5.QtSvg import QSvgRenderer
from PyQt5.QtWidgets import (
    QWidget, QLabel, QGridLayout, QFrame, QSizePolicy,
    QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QApplication
)
import sys


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
        self.image_label.setFixedSize(400, 400)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid gray;")
        layout.addWidget(self.image_label)

        # Center: Chess board
        self.board_widget = Board(fen="8/8/8/8/8/8/8/8 w - - 0 1")
        layout.addWidget(self.board_widget)

        # Right: Options panel
        right_panel = QVBoxLayout()
        btn_open = QPushButton("Open File")
        btn_open.clicked.connect(self.open_file)
        right_panel.addWidget(btn_open)

        # Add other buttons
        for name in ["Detect Board", "Show Mask", "Save Board", "Save Mask", "Debug", "Change", "Play/Reset"]:
            right_panel.addWidget(QPushButton(name))

        right_panel.addStretch()
        layout.addLayout(right_panel)

    def open_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.jpg *.jpeg)")
        if file_path:
            pixmap = QPixmap(file_path).scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(pixmap)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ChessApp()
    window.show()
    sys.exit(app.exec_())
