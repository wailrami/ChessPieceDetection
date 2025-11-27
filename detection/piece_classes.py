classes = {0: 'black-bishop', 1: 'black-king', 2: 'black-knight', 3: 'black-pawn', 4: 'black-queen', 5: 'black-rook',
           6: 'white-bishop', 7: 'white-king', 8: 'white-knight', 9: 'white-pawn', 10: 'white-queen', 11: 'white-rook'}

class_names_inverse = {'bb': 0, 'bk': 1, 'bn': 2, 'bp': 3, 'bq': 4, 'br': 5,
                       'wb': 6, 'wk': 7, 'wn': 8, 'wp': 9, 'wq': 10, 'wr': 11}

class_names = {v: k for k, v in class_names_inverse.items()}
