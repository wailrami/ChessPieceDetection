# Description: This file contains the function to convert a board to a FEN string.
from detection.piece_classes import classes

fen_classes = {classes[0]: 'b', classes[1]: 'k', classes[2]: 'n', classes[3]: 'p', classes[4]: 'q', classes[5]: 'r',
               classes[6]: 'B', classes[7]: 'K', classes[8]: 'N', classes[9]: 'P', classes[10]: 'Q', classes[11]: 'R'}


def board_to_fen(board):
    fen = ''
    empty = 0
    print(board)
    for row in board:
        for square in row:
            if square == 0:
                empty += 1
            else:
                if empty > 0:
                    fen += str(empty)
                    empty = 0
                print(f"square: {square}")
                fen += fen_classes[square]
        if empty > 0:
            fen += str(empty)
            empty = 0
        fen += '/'
    return fen[:-1]

