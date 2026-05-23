"""Move ordering via MVV-LVA, killer moves, and history heuristic."""

from __future__ import annotations

from chess_project.core.types import (
    Move, MoveKind, PieceType,
    piece_type_of,
)
from chess_project.core.position import Position

# Piece values for MVV-LVA scoring (indexed by PieceType int value)
_MVV_LVA_VICTIM: list[int] = [0, 100, 300, 310, 500, 900, 0]
_MVV_LVA_ATTACKER: list[int] = [0, 5, 4, 3, 2, 1, 6]  # lower = better attacker

_CAPTURE_KINDS = frozenset({
    MoveKind.CAPTURE, MoveKind.PROMOTION_CAPTURE, MoveKind.EN_PASSANT,
})


class MvvLvaOrderer:
    """Implements MoveOrderer protocol.

    Scoring tiers (highest first):
      1. TT move             (set externally via `tt_move`)
      2. Captures by MVV-LVA (Most Valuable Victim – Least Valuable Attacker)
      3. Promotions
      4. Killer moves
      5. History heuristic    (quiet moves that caused beta cut-offs)
    """

    __slots__ = ("killers", "history", "tt_move")

    def __init__(self) -> None:
        # killers[depth] = [slot0, slot1]
        self.killers: list[list[Move | None]] = [[None, None] for _ in range(64)]
        # history[from_sq][to_sq] — accumulated depth^2 bonus
        self.history: list[list[int]] = [[0] * 64 for _ in range(64)]
        self.tt_move: Move | None = None

    # ---- MoveOrderer protocol ----

    def score(self, position: object, move: Move, depth: int) -> int:
        """Return an integer score; higher = should be searched first."""
        pos: Position = position  # type: ignore[assignment]

        # TT move is always first
        if move == self.tt_move:
            return 300_000

        board = pos.board
        kind = move.kind
        s = 0

        # Captures: MVV-LVA
        if kind in _CAPTURE_KINDS:
            if kind is MoveKind.EN_PASSANT:
                victim_val = _MVV_LVA_VICTIM[PieceType.PAWN]
            else:
                captured = board[move.to_sq]
                victim_val = _MVV_LVA_VICTIM[captured & 7] if captured else 0
            attacker_bonus = _MVV_LVA_ATTACKER[piece_type_of(board[move.from_sq])]
            s = 100_000 + victim_val * 10 + attacker_bonus

        # Promotions
        if move.promotion is not None:
            s += 90_000 + _MVV_LVA_VICTIM[move.promotion]

        # Killer moves (quiet only)
        if s == 0:
            ki = min(depth, 63)
            if move == self.killers[ki][0]:
                s = 80_000
            elif move == self.killers[ki][1]:
                s = 70_000
            else:
                s = self.history[move.from_sq][move.to_sq]

        return s

    # ---- Maintenance helpers (called by search) ----

    def record_killer(self, move: Move, depth: int) -> None:
        ki = min(depth, 63)
        if move != self.killers[ki][0]:
            self.killers[ki][1] = self.killers[ki][0]
            self.killers[ki][0] = move

    def record_history(self, move: Move, depth: int) -> None:
        self.history[move.from_sq][move.to_sq] += depth * depth

    def age_history(self) -> None:
        """Decay history scores between iterative-deepening iterations."""
        for row in self.history:
            for i in range(64):
                row[i] >>= 2

    def reset_killers(self) -> None:
        self.killers = [[None, None] for _ in range(64)]
