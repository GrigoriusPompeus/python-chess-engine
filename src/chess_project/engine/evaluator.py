"""Static hand-crafted position evaluation with tapered piece-square tables.

Uses PeSTO-style tables (tuned via Texel's method) with smooth
interpolation between middlegame and endgame scores based on the
current game phase.  This replaces the older single-phase evaluation
and avoids the hard if/else endgame switch.

References:
    - PeSTO's Evaluation Function (chessprogramming.org)
    - Tapered Eval (chessprogramming.org)
"""

from __future__ import annotations

from chess_project.core.types import Color, PieceType
from chess_project.core.position import Position

# ---------------------------------------------------------------------------
# Piece values -- separate middlegame (mg) and endgame (eg) values
# indexed by PieceType int (0 unused, 1=PAWN .. 6=KING)
# ---------------------------------------------------------------------------

MG_PIECE_VALUE = [0, 82, 337, 365, 477, 1025, 0]
EG_PIECE_VALUE = [0, 94, 281, 297, 512,  936, 0]

# Kept for compatibility with code that references a single value table
# (e.g. ordering, quiescence delta pruning).
PIECE_VALUE = [0, 100, 320, 330, 500, 900, 20_000]

# ---------------------------------------------------------------------------
# Game phase increments per piece type.
# Total starting phase = 4*N + 4*B + 4*R + 2*Q = 4*1 + 4*1 + 4*2 + 2*4 = 24
# ---------------------------------------------------------------------------

_PHASE_INC = [0, 0, 1, 1, 2, 4, 0]
_TOTAL_PHASE = 24  # max phase at game start

# ---------------------------------------------------------------------------
# PeSTO piece-square tables (from White's perspective, a8 = index 0)
#
# These tables are laid out with rank 8 at the top (index 0..7) and
# rank 1 at the bottom (index 56..63).  Our board uses a1=0, so we
# mirror them vertically during the pre-computation step below.
# ---------------------------------------------------------------------------

# fmt: off
_MG_PAWN_TABLE = [
      0,   0,   0,   0,   0,   0,   0,   0,
     98, 134,  61,  95,  68, 126,  34, -11,
     -6,   7,  26,  31,  65,  56,  25, -20,
    -14,  13,   6,  21,  23,  12,  17, -23,
    -27,  -2,  -5,  12,  17,   6,  10, -25,
    -26,  -4,  -4, -10,   3,   3,  33, -12,
    -35,  -1, -20, -23, -15,  24,  38, -22,
      0,   0,   0,   0,   0,   0,   0,   0,
]

_EG_PAWN_TABLE = [
      0,   0,   0,   0,   0,   0,   0,   0,
    178, 173, 158, 134, 147, 132, 165, 187,
     94, 100,  85,  67,  56,  53,  82,  84,
     32,  24,  13,   5,  -2,   4,  17,  17,
     13,   9,  -3,  -7,  -7,  -8,   3,  -1,
      4,   7,  -6,   1,   0,  -5,  -1,  -8,
     13,   8,   8,  10,  13,   0,   2,  -7,
      0,   0,   0,   0,   0,   0,   0,   0,
]

_MG_KNIGHT_TABLE = [
   -167, -89, -34, -49,  61, -97, -15,-107,
    -73, -41,  72,  36,  23,  62,   7, -17,
    -47,  60,  37,  65,  84, 129,  73,  44,
     -9,  17,  19,  53,  37,  69,  18,  22,
    -13,   4,  16,  13,  28,  19,  21,  -8,
    -23,  -9,  12,  10,  19,  17,  25, -16,
    -29, -53, -12,  -3,  -1,  18, -14, -19,
   -105, -21, -58, -33, -17, -28, -19, -23,
]

_EG_KNIGHT_TABLE = [
    -58, -38, -13, -28, -31, -27, -63, -99,
    -25,  -8, -25,  -2,  -9, -25, -24, -52,
    -24, -20,  10,   9,  -1,  -9, -19, -41,
    -17,   3,  22,  22,  22,  11,   8, -18,
    -18,  -6,  16,  25,  16,  17,   4, -18,
    -23,  -3,  -1,  15,  10,  -3, -20, -22,
    -42, -20, -10,  -5,  -2, -20, -23, -44,
    -29, -51, -23, -15, -22, -18, -50, -64,
]

_MG_BISHOP_TABLE = [
    -29,   4, -82, -37, -25, -42,   7,  -8,
    -26,  16, -18, -13,  30,  59,  18, -47,
    -16,  37,  43,  40,  35,  50,  37,  -2,
     -4,   5,  19,  50,  37,  37,   7,  -2,
     -6,  13,  13,  26,  34,  12,  10,   4,
      0,  15,  15,  15,  14,  27,  18,  10,
      4,  15,  16,   0,   7,  21,  33,   1,
    -33,  -3, -14, -21, -13, -12, -39, -21,
]

_EG_BISHOP_TABLE = [
    -14, -21, -11,  -8,  -7,  -9, -17, -24,
     -8,  -4,   7, -12,  -3, -13,  -4, -14,
      2,  -8,   0,  -1,  -2,   6,   0,   4,
     -3,   9,  12,   9,  14,  10,   3,   2,
     -6,   3,  13,  19,   7,  10,  -3,  -9,
    -12,  -3,   8,  10,  13,   3,  -7, -15,
    -14, -18,  -7,  -1,   4,  -9, -15, -27,
    -23,  -9, -23,  -5,  -9, -16,  -5, -17,
]

_MG_ROOK_TABLE = [
     32,  42,  32,  51,  63,   9,  31,  43,
     27,  32,  58,  62,  80,  67,  26,  44,
     -5,  19,  26,  36,  17,  45,  61,  16,
    -24, -11,   7,  26,  24,  35,  -8, -20,
    -36, -26, -12,  -1,   9,  -7,   6, -23,
    -45, -25, -16, -17,   3,   0,  -5, -33,
    -44, -16, -20,  -9,  -1,  11,  -6, -71,
    -19, -13,   1,  17,  16,   7, -37, -26,
]

_EG_ROOK_TABLE = [
     13,  10,  18,  15,  12,  12,   8,   5,
     11,  13,  13,  11,  -3,   3,   8,   3,
      7,   7,   7,   5,   4,  -3,  -5,  -3,
      4,   3,  13,   1,   2,   1,  -1,   2,
      3,   5,   8,   4,  -5,  -6,  -8, -11,
     -4,   0,  -5,  -1,  -7, -12,  -8, -16,
     -6,  -6,   0,   2,  -9,  -9, -11,  -3,
     -9,   2,   3,  -1,  -5, -13,   4, -20,
]

_MG_QUEEN_TABLE = [
    -28,   0,  29,  12,  59,  44,  43,  45,
    -24, -39,  -5,   1, -16,  57,  28,  54,
    -13, -17,   7,   8,  29,  56,  47,  57,
    -27, -27, -16, -16,  -1,  17,  -2,   1,
     -9, -26,  -9, -10,  -2,  -4,   3,  -3,
    -14,   2, -11,  -2,  -5,   2,  14,   5,
    -35,  -8,  11,   2,   8,  15,  -3,   1,
     -1, -18,  -9,  10, -15, -25, -31, -50,
]

_EG_QUEEN_TABLE = [
     -9,  22,  22,  27,  27,  19,  10,  20,
    -17,  20,  32,  41,  58,  25,  30,   0,
    -20,   6,   9,  49,  47,  35,  19,   9,
      3,  22,  24,  45,  57,  40,  57,  36,
    -18,  28,  19,  47,  31,  34,  39,  23,
    -16, -27,  15,   6,   9,  17,  10,   5,
    -22, -23, -30, -16, -16, -23, -36, -32,
    -33, -28, -22, -43,  -5, -32, -20, -41,
]

_MG_KING_TABLE = [
    -65,  23,  16, -15, -56, -34,   2,  13,
     29,  -1, -20,  -7,  -8,  -4, -38, -29,
     -9,  24,   2, -16, -20,   6,  22, -22,
    -17, -20, -12, -27, -30, -25, -14, -36,
    -49,  -1, -27, -39, -46, -44, -33, -51,
    -14, -14, -22, -46, -44, -30, -15, -27,
      1,   7,  -8, -64, -43, -16,   9,   8,
    -15,  36,  12, -54,   8, -28,  24,  14,
]

_EG_KING_TABLE = [
    -74, -35, -18, -18, -11,  15,   4, -17,
    -12,  17,  14,  17,  17,  38,  23,  11,
     10,  17,  23,  15,  20,  45,  44,  13,
     -8,  22,  24,  27,  26,  33,  26,   3,
    -18,  -4,  21,  24,  27,  23,   9, -11,
    -19,  -3,  11,  21,  23,  16,   7,  -9,
    -27, -11,   4,  13,  14,   4,  -5, -17,
    -53, -34, -21, -11, -28, -14, -24, -43,
]
# fmt: on

# Raw tables grouped by piece type (1=PAWN .. 6=KING) for mg and eg.
# Laid out with rank 8 at index 0 (a8-h8) .. rank 1 at index 56 (a1-h1).
_MG_TABLES = [
    [],  # 0 unused
    _MG_PAWN_TABLE,
    _MG_KNIGHT_TABLE,
    _MG_BISHOP_TABLE,
    _MG_ROOK_TABLE,
    _MG_QUEEN_TABLE,
    _MG_KING_TABLE,
]

_EG_TABLES = [
    [],  # 0 unused
    _EG_PAWN_TABLE,
    _EG_KNIGHT_TABLE,
    _EG_BISHOP_TABLE,
    _EG_ROOK_TABLE,
    _EG_QUEEN_TABLE,
    _EG_KING_TABLE,
]

# ---------------------------------------------------------------------------
# Pre-compute lookup arrays for fast evaluation.
#
# Our board layout: a1 = 0, h1 = 7, a2 = 8, ..., h8 = 63.
# PeSTO layout:     a8 = 0, h8 = 7, a7 = 8, ..., h1 = 63.
#
# To convert: pesto_index = (7 - rank) * 8 + file
# For Black we also mirror vertically: pesto_index = rank * 8 + file
#
# We bake piece values directly into the tables so evaluation is a
# single table lookup per piece (no separate value addition needed).
# ---------------------------------------------------------------------------

# mg_pst_w[piece_type][sq], eg_pst_w[piece_type][sq]  -- White's tables
# mg_pst_b[piece_type][sq], eg_pst_b[piece_type][sq]  -- Black's tables
_mg_pst_w: list[list[int]] = [[] for _ in range(7)]
_eg_pst_w: list[list[int]] = [[] for _ in range(7)]
_mg_pst_b: list[list[int]] = [[] for _ in range(7)]
_eg_pst_b: list[list[int]] = [[] for _ in range(7)]

for _pt in range(1, 7):
    _mg_raw = _MG_TABLES[_pt]
    _eg_raw = _EG_TABLES[_pt]
    _mg_val = MG_PIECE_VALUE[_pt]
    _eg_val = EG_PIECE_VALUE[_pt]

    _mw: list[int] = [0] * 64
    _ew: list[int] = [0] * 64
    _mb: list[int] = [0] * 64
    _eb: list[int] = [0] * 64

    for _sq in range(64):
        _rank = _sq >> 3
        _file = _sq & 7
        # White: flip rank for PeSTO lookup
        _pesto_w = (7 - _rank) * 8 + _file
        # Black: direct mapping (PeSTO is already from White's POV,
        # so Black's mirrored view = rank * 8 + file)
        _pesto_b = _rank * 8 + _file

        _mw[_sq] = _mg_val + _mg_raw[_pesto_w]
        _ew[_sq] = _eg_val + _eg_raw[_pesto_w]
        _mb[_sq] = _mg_val + _mg_raw[_pesto_b]
        _eb[_sq] = _eg_val + _eg_raw[_pesto_b]

    _mg_pst_w[_pt] = _mw
    _eg_pst_w[_pt] = _ew
    _mg_pst_b[_pt] = _mb
    _eg_pst_b[_pt] = _eb


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(position: Position) -> int:
    """Score in centipawns, positive favours White.

    Uses tapered evaluation: computes separate middlegame and endgame
    scores, then interpolates between them based on the current game
    phase (how much material is left on the board).
    """
    board = position.board
    mg_score = 0
    eg_score = 0
    phase = 0
    w_bishops = 0
    b_bishops = 0

    for sq in range(64):
        piece = board[sq]
        if piece == 0:
            continue
        pt = piece & 7   # piece_type_of inlined
        color = piece >> 3  # piece_color inlined

        if color == 0:  # WHITE
            mg_score += _mg_pst_w[pt][sq]
            eg_score += _eg_pst_w[pt][sq]
            if pt == 3:
                w_bishops += 1
        else:  # BLACK
            mg_score -= _mg_pst_b[pt][sq]
            eg_score -= _eg_pst_b[pt][sq]
            if pt == 3:
                b_bishops += 1

        phase += _PHASE_INC[pt]

    # Bishop pair bonus
    if w_bishops >= 2:
        mg_score += 30
        eg_score += 50
    if b_bishops >= 2:
        mg_score -= 30
        eg_score -= 50

    # Tapered interpolation: phase = 0 means endgame, _TOTAL_PHASE means
    # full middlegame.  Clamp in case of promotions pushing phase above max.
    if phase > _TOTAL_PHASE:
        phase = _TOTAL_PHASE

    score = (mg_score * phase + eg_score * (_TOTAL_PHASE - phase)) // _TOTAL_PHASE

    # Small tempo bonus for the side to move (avoids score oscillation
    # between odd and even search depths).
    if position.turn == Color.WHITE:
        score += 15
    else:
        score -= 15

    return score
