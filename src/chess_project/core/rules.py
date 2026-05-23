"""Game session with history, outcome detection, and draw rules."""

from __future__ import annotations

from chess_project.core.types import (
    Color, PieceType, Move, UndoState,
    piece_color, piece_type_of,
)
from chess_project.core.position import Position
from chess_project.io.fen import parse_fen, format_fen, STARTING_FEN


class Game:
    def __init__(self, fen: str | None = None) -> None:
        if fen is not None:
            self.position = parse_fen(fen)
        else:
            self.position = Position()
        self.history: list[tuple[Move, UndoState]] = []
        self.position_keys: list[str] = [self._position_key()]
        self.headers: dict[str, str] = {
            "Event": "?",
            "Site": "?",
            "Date": "????.??.??",
            "White": "?",
            "Black": "?",
            "Result": "*",
        }

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def push(self, move: Move) -> None:
        undo = self.position.make_move(move)
        self.history.append((move, undo))
        self.position_keys.append(self._position_key())

    def pop(self) -> Move:
        move, undo = self.history.pop()
        self.position_keys.pop()
        self.position.unmake_move(move, undo)
        return move

    def legal_moves(self) -> list[Move]:
        return self.position.generate_legal_moves()

    def reset(self, fen: str | None = None) -> None:
        self.__init__(fen)  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Outcome detection
    # ------------------------------------------------------------------

    def outcome(self) -> tuple[str, str] | None:
        """Return (result, reason) or None if the game is still in progress."""
        legal = self.position.generate_legal_moves()
        in_check = self.position.is_in_check()

        if not legal:
            if in_check:
                winner = "1-0" if self.position.turn == Color.BLACK else "0-1"
                return (winner, "checkmate")
            return ("1/2-1/2", "stalemate")

        if self.is_insufficient_material():
            return ("1/2-1/2", "insufficient material")
        if self.is_fifty_move_rule():
            return ("1/2-1/2", "fifty-move rule")
        if self.is_threefold_repetition():
            return ("1/2-1/2", "threefold repetition")
        return None

    def is_game_over(self) -> bool:
        return self.outcome() is not None

    # ------------------------------------------------------------------
    # Draw helpers
    # ------------------------------------------------------------------

    def is_fifty_move_rule(self) -> bool:
        return self.position.halfmove_clock >= 100

    def is_threefold_repetition(self) -> bool:
        current = self.position_keys[-1]
        return self.position_keys.count(current) >= 3

    def is_insufficient_material(self) -> bool:
        pieces: list[tuple[Color, PieceType]] = []
        for sq in range(64):
            p = self.position.board[sq]
            if p != 0:
                pieces.append((piece_color(p), piece_type_of(p)))

        # Filter out kings
        non_kings = [(c, pt) for c, pt in pieces if pt != PieceType.KING]

        if len(non_kings) == 0:
            return True  # K vs K
        if len(non_kings) == 1:
            _, pt = non_kings[0]
            if pt in (PieceType.KNIGHT, PieceType.BISHOP):
                return True  # K+N vs K  or  K+B vs K
        if len(non_kings) == 2:
            # K+B vs K+B with bishops on same colour
            if all(pt == PieceType.BISHOP for _, pt in non_kings):
                bishop_sqs = [
                    sq for sq in range(64)
                    if self.position.board[sq] != 0
                    and piece_type_of(self.position.board[sq]) == PieceType.BISHOP
                ]
                if len(bishop_sqs) == 2:
                    # Same colour square?
                    colours = [(sq % 8 + sq // 8) % 2 for sq in bishop_sqs]
                    if colours[0] == colours[1]:
                        return True
        return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _position_key(self) -> str:
        fen = format_fen(self.position)
        # Use only piece placement + turn + castling + ep for repetition
        return " ".join(fen.split()[:4])
