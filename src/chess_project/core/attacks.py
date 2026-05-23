"""Pre-computed attack tables for non-sliding pieces and ray helpers."""

from __future__ import annotations

from chess_project.core.types import FILE_OF, RANK_OF


# ---------------------------------------------------------------------------
# Lookup tables  (populated once at import time)
# ---------------------------------------------------------------------------

KNIGHT_ATTACKS: list[list[int]] = [[] for _ in range(64)]
KING_ATTACKS: list[list[int]] = [[] for _ in range(64)]
PAWN_ATTACKS: list[list[list[int]]] = [[[] for _ in range(64)] for _ in range(2)]

_KNIGHT_OFFSETS = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                   (1, -2), (1, 2), (2, -1), (2, 1)]
_KING_OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                 (0, 1), (1, -1), (1, 0), (1, 1)]


def _init_tables() -> None:
    for sq in range(64):
        r, f = RANK_OF[sq], FILE_OF[sq]

        for dr, df in _KNIGHT_OFFSETS:
            nr, nf = r + dr, f + df
            if 0 <= nr < 8 and 0 <= nf < 8:
                KNIGHT_ATTACKS[sq].append(nr * 8 + nf)

        for dr, df in _KING_OFFSETS:
            nr, nf = r + dr, f + df
            if 0 <= nr < 8 and 0 <= nf < 8:
                KING_ATTACKS[sq].append(nr * 8 + nf)

        # White pawn attacks (upward diagonals)
        if r < 7:
            if f > 0:
                PAWN_ATTACKS[0][sq].append((r + 1) * 8 + (f - 1))
            if f < 7:
                PAWN_ATTACKS[0][sq].append((r + 1) * 8 + (f + 1))
        # Black pawn attacks (downward diagonals)
        if r > 0:
            if f > 0:
                PAWN_ATTACKS[1][sq].append((r - 1) * 8 + (f - 1))
            if f < 7:
                PAWN_ATTACKS[1][sq].append((r - 1) * 8 + (f + 1))


_init_tables()


# ---------------------------------------------------------------------------
# Ray directions for sliding pieces
# ---------------------------------------------------------------------------

BISHOP_DIRS: list[tuple[int, int]] = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
ROOK_DIRS: list[tuple[int, int]] = [(-1, 0), (1, 0), (0, -1), (0, 1)]
QUEEN_DIRS: list[tuple[int, int]] = BISHOP_DIRS + ROOK_DIRS
