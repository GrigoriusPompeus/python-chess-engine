from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, IntFlag, Enum, auto
from typing import Protocol


# ---------------------------------------------------------------------------
# Colour & piece helpers
# ---------------------------------------------------------------------------

class Color(IntEnum):
    WHITE = 0
    BLACK = 1

    def flip(self) -> Color:
        return Color(1 - self)


class PieceType(IntEnum):
    PAWN = 1
    KNIGHT = 2
    BISHOP = 3
    ROOK = 4
    QUEEN = 5
    KING = 6


def make_piece(color: Color, piece_type: PieceType) -> int:
    """Encode a piece as color * 8 + piece_type.  0 means empty."""
    return color * 8 + piece_type


def piece_color(piece: int) -> Color:
    return Color(piece >> 3)


def piece_type_of(piece: int) -> PieceType:
    return PieceType(piece & 7)


#                   0  1  2  3  4  5  6  7  8  9 10 11 12 13 14
PIECE_CHARS = list(".PNBRQK..pnbrqk")


def piece_char(piece: int) -> str:
    return PIECE_CHARS[piece]


CHAR_TO_PIECE = {ch: i for i, ch in enumerate(PIECE_CHARS) if ch != "."}
CHAR_TO_PIECE["."] = 0


# ---------------------------------------------------------------------------
# Square helpers  (a1 = 0, h8 = 63)
# ---------------------------------------------------------------------------

FILE_OF = [sq % 8 for sq in range(64)]
RANK_OF = [sq // 8 for sq in range(64)]

FILE_NAMES = "abcdefgh"
RANK_NAMES = "12345678"


def sq_name(sq: int) -> str:
    return FILE_NAMES[FILE_OF[sq]] + RANK_NAMES[RANK_OF[sq]]


def sq_from_name(name: str) -> int:
    return (ord(name[0]) - ord("a")) + (int(name[1]) - 1) * 8


# ---------------------------------------------------------------------------
# Move types
# ---------------------------------------------------------------------------

class MoveKind(Enum):
    QUIET = auto()
    CAPTURE = auto()
    DOUBLE_PAWN_PUSH = auto()
    KING_CASTLE = auto()
    QUEEN_CASTLE = auto()
    EN_PASSANT = auto()
    PROMOTION = auto()
    PROMOTION_CAPTURE = auto()


class CastlingRights(IntFlag):
    NONE = 0
    WK = 1
    WQ = 2
    BK = 4
    BQ = 8
    ALL = 15


@dataclass(frozen=True, slots=True)
class Move:
    from_sq: int
    to_sq: int
    kind: MoveKind
    promotion: PieceType | None = None

    def to_uci(self) -> str:
        s = sq_name(self.from_sq) + sq_name(self.to_sq)
        if self.promotion is not None:
            s += "nbrq"[self.promotion - PieceType.KNIGHT]
        return s


@dataclass(slots=True)
class UndoState:
    captured_piece: int
    castling_rights: CastlingRights
    ep_square: int | None
    halfmove_clock: int


# ---------------------------------------------------------------------------
# Protocol interfaces for future AI plug-ins
# ---------------------------------------------------------------------------

class SearchService(Protocol):
    def choose_move(
        self,
        position: object,
        *,
        time_ms: int | None = None,
        depth: int | None = None,
    ) -> Move: ...


class Evaluator(Protocol):
    def evaluate(self, position: object) -> int: ...


class MoveOrderer(Protocol):
    def score(self, position: object, move: Move, depth: int) -> int: ...
