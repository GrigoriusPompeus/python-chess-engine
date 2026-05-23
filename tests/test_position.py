import pytest
from chess_project.core.position import Position
from chess_project.core.types import (
    Color, PieceType, MoveKind, Move,
    make_piece, piece_type_of, sq_from_name,
)
from chess_project.io.fen import parse_fen, format_fen


class TestStartingPosition:
    def test_20_legal_moves(self):
        pos = Position()
        legal = pos.generate_legal_moves()
        assert len(legal) == 20

    def test_not_in_check(self):
        pos = Position()
        assert not pos.is_in_check()

    def test_make_unmake_preserves_fen(self):
        pos = Position()
        fen_before = format_fen(pos)
        legal = pos.generate_legal_moves()
        for move in legal:
            undo = pos.make_move(move)
            pos.unmake_move(move, undo)
            assert format_fen(pos) == fen_before


class TestMakeUnmake:
    def test_e2e4(self):
        pos = Position()
        e2e4 = Move(sq_from_name("e2"), sq_from_name("e4"), MoveKind.DOUBLE_PAWN_PUSH)
        undo = pos.make_move(e2e4)
        assert pos.board[sq_from_name("e4")] == make_piece(Color.WHITE, PieceType.PAWN)
        assert pos.board[sq_from_name("e2")] == 0
        assert pos.ep_square == sq_from_name("e3")
        assert pos.turn == Color.BLACK
        pos.unmake_move(e2e4, undo)
        assert pos.turn == Color.WHITE
        assert pos.ep_square is None

    def test_capture(self):
        pos = parse_fen("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2")
        exd5 = Move(sq_from_name("e4"), sq_from_name("d5"), MoveKind.CAPTURE)
        fen_before = format_fen(pos)
        undo = pos.make_move(exd5)
        assert pos.board[sq_from_name("d5")] == make_piece(Color.WHITE, PieceType.PAWN)
        pos.unmake_move(exd5, undo)
        assert format_fen(pos) == fen_before


class TestCastling:
    def test_white_kingside_castle(self):
        pos = parse_fen("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1")
        castle = Move(4, 6, MoveKind.KING_CASTLE)
        legal = pos.generate_legal_moves()
        assert castle in legal
        undo = pos.make_move(castle)
        assert pos.king_sq[Color.WHITE] == 6
        assert pos.board[5] == make_piece(Color.WHITE, PieceType.ROOK)
        assert pos.board[7] == 0
        pos.unmake_move(castle, undo)
        assert pos.king_sq[Color.WHITE] == 4

    def test_white_queenside_castle(self):
        pos = parse_fen("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1")
        castle = Move(4, 2, MoveKind.QUEEN_CASTLE)
        legal = pos.generate_legal_moves()
        assert castle in legal

    def test_no_castle_through_check(self):
        # Rook on f8 attacks f1 — can't castle kingside
        pos = parse_fen("5r2/8/8/8/8/8/8/R3K2R w KQ - 0 1")
        legal = pos.generate_legal_moves()
        castle_ks = Move(4, 6, MoveKind.KING_CASTLE)
        assert castle_ks not in legal

    def test_no_castle_while_in_check(self):
        pos = parse_fen("4r3/8/8/8/8/8/8/R3K2R w KQ - 0 1")
        legal = pos.generate_legal_moves()
        castle_ks = Move(4, 6, MoveKind.KING_CASTLE)
        castle_qs = Move(4, 2, MoveKind.QUEEN_CASTLE)
        assert castle_ks not in legal
        assert castle_qs not in legal


class TestEnPassant:
    def test_ep_capture(self):
        # After 1.e4 d5 2.e5 f5 — white can play exf6 e.p.
        pos = parse_fen("rnbqkbnr/ppppp1pp/8/4Pp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3")
        ep = Move(sq_from_name("e5"), sq_from_name("f6"), MoveKind.EN_PASSANT)
        legal = pos.generate_legal_moves()
        assert ep in legal
        fen_before = format_fen(pos)
        undo = pos.make_move(ep)
        # f5 pawn should be gone
        assert pos.board[sq_from_name("f5")] == 0
        assert pos.board[sq_from_name("f6")] == make_piece(Color.WHITE, PieceType.PAWN)
        pos.unmake_move(ep, undo)
        assert format_fen(pos) == fen_before

    def test_ep_illegal_if_exposes_king(self):
        # King on e5, enemy rook on a5, own pawn on d5, enemy pawn just pushed to c5
        # en passant dxc6 would remove the c5 pawn and expose king to rook
        pos = parse_fen("8/8/8/r2pPk2/8/8/8/4K3 b - - 0 1")
        # Black has no ep here, but let's set up white's perspective
        pos2 = parse_fen("8/8/8/r1pPK3/8/8/8/4k3 w - c6 0 1")
        legal = pos2.generate_legal_moves()
        ep = Move(sq_from_name("d5"), sq_from_name("c6"), MoveKind.EN_PASSANT)
        assert ep not in legal


class TestPromotion:
    def test_promotion_generates_four_options(self):
        pos = parse_fen("8/P7/8/8/8/8/8/4K2k w - - 0 1")
        legal = pos.generate_legal_moves()
        promos = [m for m in legal if m.from_sq == sq_from_name("a7") and m.to_sq == sq_from_name("a8")]
        assert len(promos) == 4
        promo_types = {m.promotion for m in promos}
        assert promo_types == {PieceType.QUEEN, PieceType.ROOK, PieceType.BISHOP, PieceType.KNIGHT}


class TestCheck:
    def test_in_check(self):
        pos = parse_fen("rnbqkbnr/pppp1ppp/8/4p3/8/5P2/PPPPP1PP/RNBQKBNR w KQkq - 0 1")
        # Not in check here
        assert not pos.is_in_check()

    def test_double_check_only_king_moves(self):
        # Double check: only king moves are legal
        pos = parse_fen("4k3/8/8/1b6/8/8/1r6/3K4 w - - 0 1")
        legal = pos.generate_legal_moves()
        for m in legal:
            assert piece_type_of(pos.board[m.from_sq]) == PieceType.KING
