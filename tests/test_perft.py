"""Perft (move-path enumeration) tests — the gold standard for movegen correctness."""

import pytest
from chess_project.core.position import Position
from chess_project.io.fen import parse_fen


def perft(pos: Position, depth: int) -> int:
    if depth == 0:
        return 1
    count = 0
    for move in pos.generate_legal_moves():
        undo = pos.make_move(move)
        count += perft(pos, depth - 1)
        pos.unmake_move(move, undo)
    return count


class TestPerftStartPosition:
    """Starting position: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"""

    def test_depth_1(self):
        assert perft(Position(), 1) == 20

    def test_depth_2(self):
        assert perft(Position(), 2) == 400

    def test_depth_3(self):
        assert perft(Position(), 3) == 8_902

    @pytest.mark.slow
    def test_depth_4(self):
        assert perft(Position(), 4) == 197_281


class TestPerftKiwipete:
    """Position 2 (Kiwipete): exercises castling, checks, and tactical legality."""

    FEN = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"

    def test_depth_1(self):
        assert perft(parse_fen(self.FEN), 1) == 48

    def test_depth_2(self):
        assert perft(parse_fen(self.FEN), 2) == 2_039

    @pytest.mark.slow
    def test_depth_3(self):
        assert perft(parse_fen(self.FEN), 3) == 97_862


class TestPerftPosition4:
    """Position 4: stresses promotions and en passant."""

    FEN = "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1"

    def test_depth_1(self):
        assert perft(parse_fen(self.FEN), 1) == 6

    def test_depth_2(self):
        assert perft(parse_fen(self.FEN), 2) == 264

    @pytest.mark.slow
    def test_depth_3(self):
        assert perft(parse_fen(self.FEN), 3) == 9_467
