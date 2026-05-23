"""UCI and SAN move parsing / formatting."""

from __future__ import annotations

from chess_project.core.types import (
    Color, PieceType, MoveKind, Move,
    make_piece, piece_color, piece_type_of,
    sq_from_name, sq_name, FILE_NAMES, RANK_OF, FILE_OF,
)
from chess_project.core.position import Position

_PROMO_FROM_CHAR = {"n": PieceType.KNIGHT, "b": PieceType.BISHOP,
                    "r": PieceType.ROOK, "q": PieceType.QUEEN}
_PIECE_LETTERS = {PieceType.KNIGHT: "N", PieceType.BISHOP: "B",
                  PieceType.ROOK: "R", PieceType.QUEEN: "Q", PieceType.KING: "K"}


# ------------------------------------------------------------------
# UCI
# ------------------------------------------------------------------

def parse_uci(pos: Position, uci: str) -> Move:
    """Convert a UCI string (e.g. 'e2e4', 'e7e8q') into a Move."""
    from_sq = sq_from_name(uci[0:2])
    to_sq = sq_from_name(uci[2:4])
    promotion = _PROMO_FROM_CHAR.get(uci[4].lower()) if len(uci) == 5 else None

    piece = pos.board[from_sq]
    pt = piece_type_of(piece)
    target = pos.board[to_sq]
    color = pos.turn

    # King specials
    if pt == PieceType.KING:
        if to_sq - from_sq == 2:
            return Move(from_sq, to_sq, MoveKind.KING_CASTLE)
        if from_sq - to_sq == 2:
            return Move(from_sq, to_sq, MoveKind.QUEEN_CASTLE)

    # Pawn specials
    if pt == PieceType.PAWN:
        if to_sq == pos.ep_square:
            return Move(from_sq, to_sq, MoveKind.EN_PASSANT)
        if abs(to_sq - from_sq) == 16:
            return Move(from_sq, to_sq, MoveKind.DOUBLE_PAWN_PUSH)
        if promotion is not None:
            kind = MoveKind.PROMOTION_CAPTURE if target != 0 else MoveKind.PROMOTION
            return Move(from_sq, to_sq, kind, promotion)

    # Regular move
    if target != 0:
        return Move(from_sq, to_sq, MoveKind.CAPTURE)
    return Move(from_sq, to_sq, MoveKind.QUIET)


# ------------------------------------------------------------------
# SAN generation  (for PGN export & display)
# ------------------------------------------------------------------

def move_to_san(pos: Position, move: Move) -> str:
    """Produce SAN for *move* given the current *pos* (before the move)."""
    if move.kind == MoveKind.KING_CASTLE:
        san = "O-O"
    elif move.kind == MoveKind.QUEEN_CASTLE:
        san = "O-O-O"
    else:
        piece = pos.board[move.from_sq]
        pt = piece_type_of(piece)
        color = pos.turn
        dst = sq_name(move.to_sq)
        capture = move.kind in (MoveKind.CAPTURE, MoveKind.PROMOTION_CAPTURE, MoveKind.EN_PASSANT)

        if pt == PieceType.PAWN:
            if capture:
                san = FILE_NAMES[FILE_OF[move.from_sq]] + "x" + dst
            else:
                san = dst
        else:
            letter = _PIECE_LETTERS[pt]
            disambig = _disambiguation(pos, move, pt, color)
            cap = "x" if capture else ""
            san = letter + disambig + cap + dst

        # Promotion suffix
        if move.promotion is not None:
            san += "=" + _PIECE_LETTERS[move.promotion]

    # Check / checkmate suffix
    undo = pos.make_move(move)
    if pos.is_in_check():
        if len(pos.generate_legal_moves()) == 0:
            san += "#"
        else:
            san += "+"
    pos.unmake_move(move, undo)
    return san


def _disambiguation(pos: Position, move: Move, pt: PieceType, color: Color) -> str:
    """Return file, rank, or both if another piece of same type can reach to_sq."""
    legal = pos.generate_legal_moves()
    ambiguous = [
        m for m in legal
        if m.to_sq == move.to_sq
        and m.from_sq != move.from_sq
        and piece_type_of(pos.board[m.from_sq]) == pt
    ]
    if not ambiguous:
        return ""
    same_file = any(FILE_OF[m.from_sq] == FILE_OF[move.from_sq] for m in ambiguous)
    same_rank = any(RANK_OF[m.from_sq] == RANK_OF[move.from_sq] for m in ambiguous)
    if not same_file:
        return FILE_NAMES[FILE_OF[move.from_sq]]
    if not same_rank:
        return str(RANK_OF[move.from_sq] + 1)
    return sq_name(move.from_sq)


# ------------------------------------------------------------------
# SAN parsing  (for PGN import)
# ------------------------------------------------------------------

def parse_san(pos: Position, san: str) -> Move:
    """Find the legal move matching a SAN string."""
    legal = pos.generate_legal_moves()

    # Strip check/mate markers
    clean = san.rstrip("+#")

    # Castling
    if clean == "O-O" or clean == "0-0":
        for m in legal:
            if m.kind == MoveKind.KING_CASTLE:
                return m
        raise ValueError(f"Illegal SAN: {san}")
    if clean in ("O-O-O", "0-0-0"):
        for m in legal:
            if m.kind == MoveKind.QUEEN_CASTLE:
                return m
        raise ValueError(f"Illegal SAN: {san}")

    # Promotion?
    promotion: PieceType | None = None
    if "=" in clean:
        clean, promo_ch = clean.split("=")
        promotion = _PROMO_FROM_CHAR[promo_ch.lower()]

    # Piece type
    if clean[0] in "NBRQK":
        pt = {"N": PieceType.KNIGHT, "B": PieceType.BISHOP, "R": PieceType.ROOK,
              "Q": PieceType.QUEEN, "K": PieceType.KING}[clean[0]]
        rest = clean[1:]
    else:
        pt = PieceType.PAWN
        rest = clean

    # Remove capture marker
    rest = rest.replace("x", "")

    # Target square is always the last two characters
    to_sq = sq_from_name(rest[-2:])
    disambig = rest[:-2]

    # Filter legal moves
    candidates = [
        m for m in legal
        if m.to_sq == to_sq
        and piece_type_of(pos.board[m.from_sq]) == pt
        and m.promotion == promotion
    ]

    if len(disambig) == 1:
        if disambig.isdigit():
            rank = int(disambig) - 1
            candidates = [m for m in candidates if RANK_OF[m.from_sq] == rank]
        else:
            file = ord(disambig) - ord("a")
            candidates = [m for m in candidates if FILE_OF[m.from_sq] == file]
    elif len(disambig) == 2:
        fsq = sq_from_name(disambig)
        candidates = [m for m in candidates if m.from_sq == fsq]

    if len(candidates) != 1:
        raise ValueError(f"Ambiguous or illegal SAN: {san} ({len(candidates)} matches)")
    return candidates[0]
