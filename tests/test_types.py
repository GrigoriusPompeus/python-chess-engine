import pytest
from chess_project.core.types import (
    Color, PieceType, MoveKind, CastlingRights,
    Move, make_piece, piece_color, piece_type_of,
    piece_char, sq_name, sq_from_name,
)


class TestPieceEncoding:
    def test_white_pawn(self):
        p = make_piece(Color.WHITE, PieceType.PAWN)
        assert piece_color(p) == Color.WHITE
        assert piece_type_of(p) == PieceType.PAWN
        assert piece_char(p) == "P"

    def test_black_queen(self):
        p = make_piece(Color.BLACK, PieceType.QUEEN)
        assert piece_color(p) == Color.BLACK
        assert piece_type_of(p) == PieceType.QUEEN
        assert piece_char(p) == "q"

    def test_all_pieces_roundtrip(self):
        for color in Color:
            for pt in PieceType:
                p = make_piece(color, pt)
                assert piece_color(p) == color
                assert piece_type_of(p) == pt


class TestSquareHelpers:
    def test_a1(self):
        assert sq_name(0) == "a1"
        assert sq_from_name("a1") == 0

    def test_h8(self):
        assert sq_name(63) == "h8"
        assert sq_from_name("h8") == 63

    def test_e4(self):
        assert sq_from_name("e4") == 28
        assert sq_name(28) == "e4"


class TestMove:
    def test_uci_quiet(self):
        m = Move(12, 28, MoveKind.QUIET)
        assert m.to_uci() == "e2e4"

    def test_uci_promotion(self):
        m = Move(52, 60, MoveKind.PROMOTION, PieceType.QUEEN)
        assert m.to_uci() == "e7e8q"

    def test_uci_knight_promotion(self):
        m = Move(52, 60, MoveKind.PROMOTION, PieceType.KNIGHT)
        assert m.to_uci() == "e7e8n"


class TestColor:
    def test_flip(self):
        assert Color.WHITE.flip() == Color.BLACK
        assert Color.BLACK.flip() == Color.WHITE


class TestCastlingRights:
    def test_flags(self):
        rights = CastlingRights.WK | CastlingRights.BQ
        assert rights & CastlingRights.WK
        assert not (rights & CastlingRights.WQ)
        assert rights & CastlingRights.BQ
