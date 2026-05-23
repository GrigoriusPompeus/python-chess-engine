"""Canonical mutable chess position with integrated move generation."""

from __future__ import annotations

from chess_project.core.types import (
    Color, PieceType, MoveKind, CastlingRights,
    Move, UndoState,
    make_piece, piece_color, piece_type_of,
    FILE_OF, RANK_OF,
)
from chess_project.core.attacks import (
    KNIGHT_ATTACKS, KING_ATTACKS, PAWN_ATTACKS,
    BISHOP_DIRS, ROOK_DIRS, QUEEN_DIRS,
)

# Piece constants for fast comparison
_WP = make_piece(Color.WHITE, PieceType.PAWN)
_BP = make_piece(Color.BLACK, PieceType.PAWN)

# Starting position board (index 0 = a1, 63 = h8)
_STARTING_BOARD: list[int] = [
    # rank 1
    make_piece(Color.WHITE, PieceType.ROOK),
    make_piece(Color.WHITE, PieceType.KNIGHT),
    make_piece(Color.WHITE, PieceType.BISHOP),
    make_piece(Color.WHITE, PieceType.QUEEN),
    make_piece(Color.WHITE, PieceType.KING),
    make_piece(Color.WHITE, PieceType.BISHOP),
    make_piece(Color.WHITE, PieceType.KNIGHT),
    make_piece(Color.WHITE, PieceType.ROOK),
    # rank 2
    *([_WP] * 8),
    # ranks 3-6
    *([0] * 32),
    # rank 7
    *([_BP] * 8),
    # rank 8
    make_piece(Color.BLACK, PieceType.ROOK),
    make_piece(Color.BLACK, PieceType.KNIGHT),
    make_piece(Color.BLACK, PieceType.BISHOP),
    make_piece(Color.BLACK, PieceType.QUEEN),
    make_piece(Color.BLACK, PieceType.KING),
    make_piece(Color.BLACK, PieceType.BISHOP),
    make_piece(Color.BLACK, PieceType.KNIGHT),
    make_piece(Color.BLACK, PieceType.ROOK),
]


class Position:
    __slots__ = (
        "board", "turn", "castling_rights", "ep_square",
        "halfmove_clock", "fullmove_number", "king_sq",
    )

    def __init__(self) -> None:
        self.board: list[int] = list(_STARTING_BOARD)
        self.turn: Color = Color.WHITE
        self.castling_rights: CastlingRights = CastlingRights.ALL
        self.ep_square: int | None = None
        self.halfmove_clock: int = 0
        self.fullmove_number: int = 1
        self.king_sq: list[int] = [4, 60]  # e1, e8

    def copy(self) -> Position:
        p = Position.__new__(Position)
        p.board = list(self.board)
        p.turn = self.turn
        p.castling_rights = self.castling_rights
        p.ep_square = self.ep_square
        p.halfmove_clock = self.halfmove_clock
        p.fullmove_number = self.fullmove_number
        p.king_sq = list(self.king_sq)
        return p

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def piece_at(self, sq: int) -> int:
        return self.board[sq]

    def is_in_check(self) -> bool:
        return self.is_square_attacked(self.king_sq[self.turn], self.turn.flip())

    def is_square_attacked(self, sq: int, by_color: Color) -> bool:
        board = self.board

        # Pawn attacks: reverse-lookup trick
        for asq in PAWN_ATTACKS[1 - by_color][sq]:
            p = board[asq]
            if p != 0 and piece_color(p) == by_color and piece_type_of(p) == PieceType.PAWN:
                return True

        # Knight
        for asq in KNIGHT_ATTACKS[sq]:
            p = board[asq]
            if p != 0 and piece_color(p) == by_color and piece_type_of(p) == PieceType.KNIGHT:
                return True

        # King
        for asq in KING_ATTACKS[sq]:
            p = board[asq]
            if p != 0 and piece_color(p) == by_color and piece_type_of(p) == PieceType.KING:
                return True

        # Sliding — bishops / queens on diagonals
        r, f = RANK_OF[sq], FILE_OF[sq]
        for dr, df in BISHOP_DIRS:
            nr, nf = r + dr, f + df
            while 0 <= nr < 8 and 0 <= nf < 8:
                p = board[nr * 8 + nf]
                if p != 0:
                    if piece_color(p) == by_color:
                        pt = piece_type_of(p)
                        if pt == PieceType.BISHOP or pt == PieceType.QUEEN:
                            return True
                    break
                nr += dr
                nf += df

        # Sliding — rooks / queens on ranks/files
        for dr, df in ROOK_DIRS:
            nr, nf = r + dr, f + df
            while 0 <= nr < 8 and 0 <= nf < 8:
                p = board[nr * 8 + nf]
                if p != 0:
                    if piece_color(p) == by_color:
                        pt = piece_type_of(p)
                        if pt == PieceType.ROOK or pt == PieceType.QUEEN:
                            return True
                    break
                nr += dr
                nf += df

        return False

    # ------------------------------------------------------------------
    # Make / unmake
    # ------------------------------------------------------------------

    def make_move(self, move: Move) -> UndoState:
        undo = UndoState(
            captured_piece=0,
            castling_rights=self.castling_rights,
            ep_square=self.ep_square,
            halfmove_clock=self.halfmove_clock,
        )

        piece = self.board[move.from_sq]
        pt = piece_type_of(piece)
        color = self.turn

        # --- captures ---------------------------------------------------
        if move.kind == MoveKind.EN_PASSANT:
            cap_sq = move.to_sq + (-8 if color == Color.WHITE else 8)
            undo.captured_piece = self.board[cap_sq]
            self.board[cap_sq] = 0
        elif move.kind in (MoveKind.CAPTURE, MoveKind.PROMOTION_CAPTURE):
            undo.captured_piece = self.board[move.to_sq]

        # --- move the piece ---------------------------------------------
        self.board[move.from_sq] = 0
        if move.promotion is not None:
            self.board[move.to_sq] = make_piece(color, move.promotion)
        else:
            self.board[move.to_sq] = piece

        # --- castling rook movement -------------------------------------
        if move.kind == MoveKind.KING_CASTLE:
            if color == Color.WHITE:
                self.board[7] = 0
                self.board[5] = make_piece(Color.WHITE, PieceType.ROOK)
            else:
                self.board[63] = 0
                self.board[61] = make_piece(Color.BLACK, PieceType.ROOK)
        elif move.kind == MoveKind.QUEEN_CASTLE:
            if color == Color.WHITE:
                self.board[0] = 0
                self.board[3] = make_piece(Color.WHITE, PieceType.ROOK)
            else:
                self.board[56] = 0
                self.board[59] = make_piece(Color.BLACK, PieceType.ROOK)

        # --- king square cache ------------------------------------------
        if pt == PieceType.KING:
            self.king_sq[color] = move.to_sq

        # --- castling rights --------------------------------------------
        if pt == PieceType.KING:
            if color == Color.WHITE:
                self.castling_rights &= ~(CastlingRights.WK | CastlingRights.WQ)
            else:
                self.castling_rights &= ~(CastlingRights.BK | CastlingRights.BQ)
        # Rook moved or captured on its starting square
        if move.from_sq == 0 or move.to_sq == 0:
            self.castling_rights &= ~CastlingRights.WQ
        if move.from_sq == 7 or move.to_sq == 7:
            self.castling_rights &= ~CastlingRights.WK
        if move.from_sq == 56 or move.to_sq == 56:
            self.castling_rights &= ~CastlingRights.BQ
        if move.from_sq == 63 or move.to_sq == 63:
            self.castling_rights &= ~CastlingRights.BK

        # --- en passant square ------------------------------------------
        if move.kind == MoveKind.DOUBLE_PAWN_PUSH:
            self.ep_square = (move.from_sq + move.to_sq) // 2
        else:
            self.ep_square = None

        # --- clocks -----------------------------------------------------
        if pt == PieceType.PAWN or undo.captured_piece != 0:
            self.halfmove_clock = 0
        else:
            self.halfmove_clock += 1
        if color == Color.BLACK:
            self.fullmove_number += 1

        # --- flip turn --------------------------------------------------
        self.turn = color.flip()
        return undo

    def unmake_move(self, move: Move, undo: UndoState) -> None:
        color = self.turn.flip()  # side that made the move

        moved_piece = self.board[move.to_sq]

        # Undo promotion → restore pawn
        if move.promotion is not None:
            original = make_piece(color, PieceType.PAWN)
        else:
            original = moved_piece

        # Restore piece to origin
        self.board[move.from_sq] = original
        self.board[move.to_sq] = 0

        # Restore captured piece
        if move.kind == MoveKind.EN_PASSANT:
            cap_sq = move.to_sq + (-8 if color == Color.WHITE else 8)
            self.board[cap_sq] = undo.captured_piece
        elif undo.captured_piece != 0:
            self.board[move.to_sq] = undo.captured_piece

        # Undo castling rook
        if move.kind == MoveKind.KING_CASTLE:
            if color == Color.WHITE:
                self.board[5] = 0
                self.board[7] = make_piece(Color.WHITE, PieceType.ROOK)
            else:
                self.board[61] = 0
                self.board[63] = make_piece(Color.BLACK, PieceType.ROOK)
        elif move.kind == MoveKind.QUEEN_CASTLE:
            if color == Color.WHITE:
                self.board[3] = 0
                self.board[0] = make_piece(Color.WHITE, PieceType.ROOK)
            else:
                self.board[59] = 0
                self.board[56] = make_piece(Color.BLACK, PieceType.ROOK)

        # Restore king square
        if piece_type_of(original) == PieceType.KING:
            self.king_sq[color] = move.from_sq

        # Restore saved state
        self.castling_rights = undo.castling_rights
        self.ep_square = undo.ep_square
        self.halfmove_clock = undo.halfmove_clock
        if color == Color.BLACK:
            self.fullmove_number -= 1
        self.turn = color

    # ------------------------------------------------------------------
    # Pseudo-legal move generation
    # ------------------------------------------------------------------

    def generate_pseudo_legal_moves(self) -> list[Move]:
        moves: list[Move] = []
        color = self.turn
        board = self.board

        for sq in range(64):
            piece = board[sq]
            if piece == 0 or piece_color(piece) != color:
                continue
            pt = piece_type_of(piece)
            if pt == PieceType.PAWN:
                self._gen_pawn(sq, color, moves)
            elif pt == PieceType.KNIGHT:
                self._gen_knight(sq, color, moves)
            elif pt == PieceType.BISHOP:
                self._gen_sliding(sq, color, BISHOP_DIRS, moves)
            elif pt == PieceType.ROOK:
                self._gen_sliding(sq, color, ROOK_DIRS, moves)
            elif pt == PieceType.QUEEN:
                self._gen_sliding(sq, color, QUEEN_DIRS, moves)
            elif pt == PieceType.KING:
                self._gen_king(sq, color, moves)
        return moves

    def generate_legal_moves(self) -> list[Move]:
        legal: list[Move] = []
        for move in self.generate_pseudo_legal_moves():
            undo = self.make_move(move)
            mover = self.turn.flip()
            if not self.is_square_attacked(self.king_sq[mover], self.turn):
                legal.append(move)
            self.unmake_move(move, undo)
        return legal

    # ------------------------------------------------------------------
    # Per-piece generators (private)
    # ------------------------------------------------------------------

    def _gen_pawn(self, sq: int, color: Color, moves: list[Move]) -> None:
        board = self.board
        direction = 8 if color == Color.WHITE else -8
        start_rank = 1 if color == Color.WHITE else 6
        promo_rank = 7 if color == Color.WHITE else 0

        # Single push
        target = sq + direction
        if 0 <= target < 64 and board[target] == 0:
            if RANK_OF[target] == promo_rank:
                for promo in (PieceType.QUEEN, PieceType.ROOK, PieceType.BISHOP, PieceType.KNIGHT):
                    moves.append(Move(sq, target, MoveKind.PROMOTION, promo))
            else:
                moves.append(Move(sq, target, MoveKind.QUIET))
                # Double push (only if single push was possible)
                if RANK_OF[sq] == start_rank:
                    target2 = sq + 2 * direction
                    if board[target2] == 0:
                        moves.append(Move(sq, target2, MoveKind.DOUBLE_PAWN_PUSH))

        # Captures
        for asq in PAWN_ATTACKS[color][sq]:
            tp = board[asq]
            if tp != 0 and piece_color(tp) != color:
                if RANK_OF[asq] == promo_rank:
                    for promo in (PieceType.QUEEN, PieceType.ROOK, PieceType.BISHOP, PieceType.KNIGHT):
                        moves.append(Move(sq, asq, MoveKind.PROMOTION_CAPTURE, promo))
                else:
                    moves.append(Move(sq, asq, MoveKind.CAPTURE))

        # En passant
        if self.ep_square is not None and self.ep_square in PAWN_ATTACKS[color][sq]:
            moves.append(Move(sq, self.ep_square, MoveKind.EN_PASSANT))

    def _gen_knight(self, sq: int, color: Color, moves: list[Move]) -> None:
        board = self.board
        for target in KNIGHT_ATTACKS[sq]:
            tp = board[target]
            if tp == 0:
                moves.append(Move(sq, target, MoveKind.QUIET))
            elif piece_color(tp) != color:
                moves.append(Move(sq, target, MoveKind.CAPTURE))

    def _gen_sliding(
        self, sq: int, color: Color,
        directions: list[tuple[int, int]], moves: list[Move],
    ) -> None:
        board = self.board
        r, f = RANK_OF[sq], FILE_OF[sq]
        for dr, df in directions:
            nr, nf = r + dr, f + df
            while 0 <= nr < 8 and 0 <= nf < 8:
                target = nr * 8 + nf
                tp = board[target]
                if tp == 0:
                    moves.append(Move(sq, target, MoveKind.QUIET))
                elif piece_color(tp) != color:
                    moves.append(Move(sq, target, MoveKind.CAPTURE))
                    break
                else:
                    break
                nr += dr
                nf += df

    def _gen_king(self, sq: int, color: Color, moves: list[Move]) -> None:
        board = self.board
        opp = color.flip()

        for target in KING_ATTACKS[sq]:
            tp = board[target]
            if tp == 0:
                moves.append(Move(sq, target, MoveKind.QUIET))
            elif piece_color(tp) != color:
                moves.append(Move(sq, target, MoveKind.CAPTURE))

        # Castling
        if color == Color.WHITE:
            if (self.castling_rights & CastlingRights.WK
                    and board[5] == 0 and board[6] == 0
                    and not self.is_square_attacked(4, opp)
                    and not self.is_square_attacked(5, opp)
                    and not self.is_square_attacked(6, opp)):
                moves.append(Move(4, 6, MoveKind.KING_CASTLE))
            if (self.castling_rights & CastlingRights.WQ
                    and board[1] == 0 and board[2] == 0 and board[3] == 0
                    and not self.is_square_attacked(4, opp)
                    and not self.is_square_attacked(3, opp)
                    and not self.is_square_attacked(2, opp)):
                moves.append(Move(4, 2, MoveKind.QUEEN_CASTLE))
        else:
            if (self.castling_rights & CastlingRights.BK
                    and board[61] == 0 and board[62] == 0
                    and not self.is_square_attacked(60, opp)
                    and not self.is_square_attacked(61, opp)
                    and not self.is_square_attacked(62, opp)):
                moves.append(Move(60, 62, MoveKind.KING_CASTLE))
            if (self.castling_rights & CastlingRights.BQ
                    and board[57] == 0 and board[58] == 0 and board[59] == 0
                    and not self.is_square_attacked(60, opp)
                    and not self.is_square_attacked(59, opp)
                    and not self.is_square_attacked(58, opp)):
                moves.append(Move(60, 58, MoveKind.QUEEN_CASTLE))

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def board_str(self) -> str:
        from chess_project.core.types import piece_char
        lines: list[str] = []
        for rank in range(7, -1, -1):
            row = []
            for file in range(8):
                p = self.board[rank * 8 + file]
                row.append(piece_char(p) if p != 0 else ".")
            lines.append(f"  {rank + 1}  {' '.join(row)}")
        lines.append("")
        lines.append("     a b c d e f g h")
        return "\n".join(lines)
