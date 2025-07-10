import importlib

class Encoder:
    def name(self):
        raise NotImplementedError

    def encode(self, game_state):
        raise NotImplementedError

    def encode_point(self, point):
        raise NotImplementedError

    def decode_point_index(self, index):
        raise NotImplementedError

    def num_points(self):
        raise NotImplementedError

    def shape(self):
        raise NotImplementedError

def get_encoder_by_name(name, boardsize):
    if isinstance(boardsize, int):
        boardsize = (boardsize, boardsize)
    module = importlib.import_module('hnefatafl.encoders.' + name)
    constructor = getattr(module, 'create')
    return constructor(boardsize)