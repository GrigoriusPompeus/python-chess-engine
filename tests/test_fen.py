import pytest
from chess_project.io.fen import parse_fen, format_fen, STARTING_FEN
from chess_project.core.types import Color, PieceType, CastlingRights, make_piece


class TestFenRoundtrip:
    def test_starting_position(self):
        pos = parse_fen(STARTING_FEN)
        assert format_fen(pos) == STARTING_FEN

    def test_turn(self):
        pos = parse_fen(STARTING_FEN)
        assert pos.turn == Color.WHITE

    def test_castling_all(self):
        pos = parse_fen(STARTING_FEN)
        assert pos.castling_rights == CastlingRights.ALL

    def test_king_squares(self):
        pos = parse_fen(STARTING_FEN)
        assert pos.king_sq[Color.WHITE] == 4   # e1
        assert pos.king_sq[Color.BLACK] == 60  # e8

    def test_custom_fen_roundtrip(self):
        fen = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1"
        pos = parse_fen(fen)
        assert format_fen(pos) == fen

    def test_ep_square(self):
        fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
        pos = parse_fen(fen)
        assert pos.ep_square == 20  # e3
        assert format_fen(pos) == fen

    def test_no_castling(self):
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1"
        pos = parse_fen(fen)
        assert pos.castling_rights == CastlingRights.NONE

    def test_black_to_move(self):
        fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
        pos = parse_fen(fen)
        assert pos.turn == Color.BLACK
