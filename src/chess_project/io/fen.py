"""FEN (Forsyth-Edwards Notation) parser and formatter."""

from __future__ import annotations

from chess_project.core.types import (
    Color, PieceType, CastlingRights,
    make_piece, piece_color, piece_type_of, piece_char,
    CHAR_TO_PIECE, sq_from_name, sq_name,
)
from chess_project.core.position import Position

STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

_CASTLING_CHAR = {
    "K": CastlingRights.WK, "Q": CastlingRights.WQ,
    "k": CastlingRights.BK, "q": CastlingRights.BQ,
}


def parse_fen(fen: str) -> Position:
    parts = fen.split()
    if len(parts) < 4:
        raise ValueError(f"Invalid FEN: need at least 4 fields, got {len(parts)}")

    pos = Position.__new__(Position)
    pos.board = [0] * 64

    # 1 — piece placement (rank 8 → rank 1)
    ranks = parts[0].split("/")
    if len(ranks) != 8:
        raise ValueError("FEN piece placement must have 8 ranks")
    for rank_idx, rank_str in enumerate(ranks):
        rank = 7 - rank_idx
        file = 0
        for ch in rank_str:
            if ch.isdigit():
                file += int(ch)
            else:
                piece = CHAR_TO_PIECE.get(ch)
                if piece is None:
                    raise ValueError(f"Unknown piece character: {ch}")
                pos.board[rank * 8 + file] = piece
                file += 1

    # 2 — active colour
    pos.turn = Color.WHITE if parts[1] == "w" else Color.BLACK

    # 3 — castling availability
    pos.castling_rights = CastlingRights.NONE
    if parts[2] != "-":
        for ch in parts[2]:
            pos.castling_rights |= _CASTLING_CHAR[ch]

    # 4 — en passant target
    pos.ep_square = None if parts[3] == "-" else sq_from_name(parts[3])

    # 5, 6 — clocks (default 0 / 1 if missing)
    pos.halfmove_clock = int(parts[4]) if len(parts) > 4 else 0
    pos.fullmove_number = int(parts[5]) if len(parts) > 5 else 1

    # Derive king squares
    pos.king_sq = [0, 0]
    wk = make_piece(Color.WHITE, PieceType.KING)
    bk = make_piece(Color.BLACK, PieceType.KING)
    for sq in range(64):
        if pos.board[sq] == wk:
            pos.king_sq[Color.WHITE] = sq
        elif pos.board[sq] == bk:
            pos.king_sq[Color.BLACK] = sq

    return pos


def format_fen(pos: Position) -> str:
    # 1 — piece placement
    rank_strs: list[str] = []
    for rank in range(7, -1, -1):
        empty = 0
        row: list[str] = []
        for file in range(8):
            p = pos.board[rank * 8 + file]
            if p == 0:
                empty += 1
            else:
                if empty:
                    row.append(str(empty))
                    empty = 0
                row.append(piece_char(p))
        if empty:
            row.append(str(empty))
        rank_strs.append("".join(row))
    placement = "/".join(rank_strs)

    # 2 — active colour
    color = "w" if pos.turn == Color.WHITE else "b"

    # 3 — castling
    castling = ""
    for ch, flag in _CASTLING_CHAR.items():
        if pos.castling_rights & flag:
            castling += ch
    if not castling:
        castling = "-"

    # 4 — en passant
    ep = "-" if pos.ep_square is None else sq_name(pos.ep_square)

    return f"{placement} {color} {castling} {ep} {pos.halfmove_clock} {pos.fullmove_number}"
