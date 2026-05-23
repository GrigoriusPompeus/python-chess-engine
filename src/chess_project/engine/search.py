"""Alpha-beta search with iterative deepening, quiescence, and Zobrist TT.

Implements the SearchService protocol from chess_project.core.types.
"""

from __future__ import annotations

import random
import time
from typing import Final

from chess_project.core.types import (
    Color, PieceType, Move, MoveKind, CastlingRights,
    piece_type_of, piece_color, make_piece,
)
from chess_project.core.position import Position
from chess_project.engine.ordering import MvvLvaOrderer

# ---------------------------------------------------------------------------
# Zobrist hashing
# ---------------------------------------------------------------------------

_RNG = random.Random(0xDEAD_BEEF)  # deterministic seed


def _rand64() -> int:
    return _RNG.getrandbits(64)


# piece_keys[piece_code][square]  — piece_code in 1..14 (0 = empty, skip)
PIECE_KEYS: Final[list[list[int]]] = [
    [0] * 64,  # index 0 unused (empty square)
    *[[_rand64() for _ in range(64)] for _ in range(14)],
]

TURN_KEY: Final[int] = _rand64()

# castling_keys[CastlingRights 0..15]
CASTLING_KEYS: Final[list[int]] = [_rand64() for _ in range(16)]

# ep_keys[file 0..7] — only the file matters for Zobrist
EP_FILE_KEYS: Final[list[int]] = [_rand64() for _ in range(8)]


def zobrist_hash(pos: Position) -> int:
    """Compute a full Zobrist hash from scratch."""
    h = 0
    for sq in range(64):
        pc = pos.board[sq]
        if pc:
            h ^= PIECE_KEYS[pc][sq]
    if pos.turn == Color.BLACK:
        h ^= TURN_KEY
    h ^= CASTLING_KEYS[pos.castling_rights]
    if pos.ep_square is not None:
        h ^= EP_FILE_KEYS[pos.ep_square % 8]
    return h


# ---------------------------------------------------------------------------
# Transposition table
# ---------------------------------------------------------------------------

TT_EXACT: Final[int] = 0
TT_ALPHA: Final[int] = 1  # upper-bound (failed low)
TT_BETA: Final[int] = 2   # lower-bound (failed high)


class _TTEntry:
    __slots__ = ("key", "depth", "score", "flag", "best_move")

    def __init__(
        self, key: int, depth: int, score: int, flag: int, best_move: Move | None,
    ) -> None:
        self.key = key
        self.depth = depth
        self.score = score
        self.flag = flag
        self.best_move = best_move


class TranspositionTable:
    """Fixed-size, replace-by-depth Zobrist transposition table."""

    __slots__ = ("_size", "_table")

    def __init__(self, size_bits: int = 19) -> None:
        self._size: int = 1 << size_bits  # ~500k buckets
        self._table: list[_TTEntry | None] = [None] * self._size

    def _index(self, key: int) -> int:
        return key & (self._size - 1)

    def probe(self, key: int) -> _TTEntry | None:
        entry = self._table[self._index(key)]
        if entry is not None and entry.key == key:
            return entry
        return None

    def store(
        self,
        key: int,
        depth: int,
        score: int,
        flag: int,
        best_move: Move | None,
    ) -> None:
        idx = self._index(key)
        existing = self._table[idx]
        # Replace if empty, same position, or shallower
        if existing is None or existing.key == key or existing.depth <= depth:
            self._table[idx] = _TTEntry(key, depth, score, flag, best_move)

    def clear(self) -> None:
        self._table = [None] * self._size


# ---------------------------------------------------------------------------
# Evaluator callback type
# ---------------------------------------------------------------------------
# We accept any object satisfying the Evaluator protocol.
# Import the concrete one the caller wants at construction time.

from chess_project.engine.evaluator import evaluate as _hce_evaluate  # noqa: E402


class _DefaultEvaluator:
    """Wraps the existing module-level evaluate() as an Evaluator protocol."""

    __slots__ = ()

    def evaluate(self, position: object) -> int:
        return _hce_evaluate(position)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHECKMATE_SCORE: Final[int] = 100_000
DRAW_SCORE: Final[int] = 0
NULL_MOVE_R: Final[int] = 3
FUTILITY_MARGINS: Final[list[int]] = [0, 200, 350, 500]

_CAPTURE_KINDS = frozenset({
    MoveKind.CAPTURE, MoveKind.PROMOTION_CAPTURE, MoveKind.EN_PASSANT,
})


# ---------------------------------------------------------------------------
# Search engine (SearchService protocol)
# ---------------------------------------------------------------------------

class AlphaBetaSearch:
    """Iterative-deepening alpha-beta with quiescence search.

    Satisfies the ``SearchService`` protocol::

        def choose_move(self, position, *, time_ms=None, depth=None) -> Move

    Constructor accepts pluggable Evaluator / MoveOrderer instances.
    """

    __slots__ = (
        "_evaluator", "_orderer", "_tt",
        "_max_depth", "_nodes", "_start", "_time_limit", "_timed_out",
        "_path_hashes", "_game_hashes",
    )

    def __init__(
        self,
        evaluator: object | None = None,
        orderer: MvvLvaOrderer | None = None,
        tt: TranspositionTable | None = None,
        max_depth: int = 33,
    ) -> None:
        self._evaluator = evaluator or _DefaultEvaluator()
        self._orderer = orderer or MvvLvaOrderer()
        self._tt = tt or TranspositionTable()
        self._max_depth = max_depth

        # Per-search state
        self._nodes: int = 0
        self._start: float = 0.0
        self._time_limit: float | None = None
        self._timed_out: bool = False
        # Repetition detection: hashes from the game so far + the search path
        self._game_hashes: set[int] = set()
        self._path_hashes: list[int] = []

    @property
    def nodes_searched(self) -> int:
        return self._nodes

    # ---- SearchService protocol ------------------------------------------

    def choose_move(
        self,
        position: object,
        *,
        time_ms: int | None = None,
        depth: int | None = None,
        history_hashes: list[int] | None = None,
    ) -> Move:
        pos: Position = position  # type: ignore[assignment]
        max_d = depth if depth is not None else self._max_depth

        # Reset per-search state
        self._nodes = 0
        self._timed_out = False
        self._start = time.time()
        self._time_limit = (time_ms / 1000.0) if time_ms else 8.0
        self._orderer.reset_killers()
        self._orderer.age_history()

        # Seed repetition detection with positions from the actual game
        self._game_hashes = set(history_hashes) if history_hashes else set()
        self._path_hashes = []

        legal = pos.generate_legal_moves()
        if not legal:
            raise RuntimeError("No legal moves available")
        if len(legal) == 1:
            return legal[0]

        best_move = legal[0]
        for d in range(1, max_d + 1):
            if self._timed_out:
                break
            candidate = self._search_root(pos, legal, d)
            if not self._timed_out:
                best_move = candidate
        return best_move

    # ---- Root search -----------------------------------------------------

    def _search_root(self, pos: Position, legal: list[Move], depth: int) -> Move:
        best_move = legal[0]
        best_score = -CHECKMATE_SCORE - 1
        alpha = -CHECKMATE_SCORE - 1
        beta = CHECKMATE_SCORE + 1

        root_key = zobrist_hash(pos)
        tt_entry = self._tt.probe(root_key)
        self._orderer.tt_move = tt_entry.best_move if tt_entry else None
        self._path_hashes = [root_key]

        self._order(pos, legal, depth)

        for i, move in enumerate(legal):
            if self._check_time():
                break
            undo = pos.make_move(move)
            child_key = zobrist_hash(pos)
            self._path_hashes.append(child_key)

            # Penalise moves that repeat a position from the game
            if child_key in self._game_hashes:
                score = DRAW_SCORE
            # LMR at root
            elif (i >= 4 and depth >= 3
                    and move.kind not in _CAPTURE_KINDS
                    and move.promotion is None
                    and not pos.is_in_check()):
                score = -self._alphabeta(pos, depth - 2, -beta, -alpha, True)
                if score > alpha:
                    score = -self._alphabeta(pos, depth - 1, -beta, -alpha, True)
            else:
                score = -self._alphabeta(pos, depth - 1, -beta, -alpha, True)

            self._path_hashes.pop()
            pos.unmake_move(move, undo)

            if score > best_score:
                best_score = score
                best_move = move
            if score > alpha:
                alpha = score

        self._tt.store(root_key, depth, best_score, TT_EXACT, best_move)
        return best_move

    # ---- Alpha-beta ------------------------------------------------------

    def _alphabeta(
        self, pos: Position, depth: int, alpha: int, beta: int, allow_null: bool,
    ) -> int:
        if self._check_time():
            return 0

        self._nodes += 1

        # Repetition detection: if this position was already seen on the
        # current search path or in the game history, treat it as a draw.
        key = zobrist_hash(pos)
        if key in self._game_hashes or self._path_hashes.count(key) >= 2:
            return DRAW_SCORE

        orig_alpha = alpha
        in_check = pos.is_in_check()

        if in_check:
            depth += 1
        tt_entry = self._tt.probe(key)
        tt_move: Move | None = None
        if tt_entry is not None:
            tt_move = tt_entry.best_move
            if tt_entry.depth >= depth:
                if tt_entry.flag == TT_EXACT:
                    return tt_entry.score
                if tt_entry.flag == TT_ALPHA and tt_entry.score <= alpha:
                    return alpha
                if tt_entry.flag == TT_BETA and tt_entry.score >= beta:
                    return beta

        # Leaf → quiescence
        if depth <= 0:
            return self._quiesce(pos, alpha, beta)

        static_eval: int | None = None

        # Null-move pruning
        if allow_null and depth >= NULL_MOVE_R + 1 and not in_check:
            if static_eval is None:
                static_eval = self._eval_relative(pos)
            if static_eval >= beta:
                pos.turn = pos.turn.flip()
                old_ep = pos.ep_square
                pos.ep_square = None
                r = NULL_MOVE_R + (1 if depth > 6 else 0)
                score = -self._alphabeta(pos, depth - 1 - r, -beta, -beta + 1, False)
                pos.turn = pos.turn.flip()
                pos.ep_square = old_ep
                if score >= beta:
                    return beta

        legal = pos.generate_legal_moves()
        if not legal:
            return -(CHECKMATE_SCORE + depth) if in_check else DRAW_SCORE

        # Futility pruning flag
        can_futility = False
        if (depth <= 3 and not in_check
                and abs(alpha) < CHECKMATE_SCORE - 100
                and abs(beta) < CHECKMATE_SCORE - 100):
            if static_eval is None:
                static_eval = self._eval_relative(pos)
            if static_eval + FUTILITY_MARGINS[depth] <= alpha:
                can_futility = True

        self._orderer.tt_move = tt_move
        self._order(pos, legal, depth)

        best_move: Move | None = None
        moves_searched = 0

        for move in legal:
            is_capture = move.kind in _CAPTURE_KINDS
            is_promo = move.promotion is not None

            # Futility skip
            if can_futility and moves_searched > 0 and not is_capture and not is_promo:
                undo = pos.make_move(move)
                gives_check = pos.is_in_check()
                pos.unmake_move(move, undo)
                if not gives_check:
                    continue

            undo = pos.make_move(move)
            self._path_hashes.append(zobrist_hash(pos))

            # Late Move Reduction
            if (moves_searched >= 4 and depth >= 3
                    and not is_capture and not is_promo
                    and not in_check and not pos.is_in_check()):
                reduction = 1 + (1 if moves_searched >= 8 else 0)
                score = -self._alphabeta(pos, depth - 1 - reduction, -beta, -alpha, True)
                if score <= alpha:
                    self._path_hashes.pop()
                    pos.unmake_move(move, undo)
                    moves_searched += 1
                    continue
                score = -self._alphabeta(pos, depth - 1, -beta, -alpha, True)
            else:
                score = -self._alphabeta(pos, depth - 1, -beta, -alpha, True)

            self._path_hashes.pop()
            pos.unmake_move(move, undo)
            moves_searched += 1

            if score >= beta:
                if not is_capture:
                    self._orderer.record_killer(move, depth)
                    self._orderer.record_history(move, depth)
                self._tt.store(key, depth, beta, TT_BETA, move)
                return beta

            if score > alpha:
                alpha = score
                best_move = move

        flag = TT_EXACT if alpha > orig_alpha else TT_ALPHA
        self._tt.store(key, depth, alpha, flag, best_move)
        return alpha

    # ---- Quiescence search -----------------------------------------------

    def _quiesce(self, pos: Position, alpha: int, beta: int) -> int:
        self._nodes += 1

        stand_pat = self._eval_relative(pos)

        if stand_pat >= beta:
            return beta
        if stand_pat + 1000 < alpha:  # delta pruning
            return alpha
        if stand_pat > alpha:
            alpha = stand_pat

        legal = pos.generate_legal_moves()
        captures = [m for m in legal if m.kind in _CAPTURE_KINDS]
        # Order captures by MVV-LVA
        captures.sort(
            key=lambda m: self._orderer.score(pos, m, 0), reverse=True,
        )

        for move in captures:
            if self._check_time():
                break
            # SEE-like delta prune
            cap_val = self._captured_value(pos.board, move)
            if stand_pat + cap_val + 200 < alpha:
                continue

            undo = pos.make_move(move)
            score = -self._quiesce(pos, -beta, -alpha)
            pos.unmake_move(move, undo)

            if score >= beta:
                return beta
            if score > alpha:
                alpha = score

        return alpha

    # ---- Helpers ---------------------------------------------------------

    def _eval_relative(self, pos: Position) -> int:
        """Evaluate from the perspective of the side to move."""
        raw: int = self._evaluator.evaluate(pos)  # type: ignore[union-attr]
        return raw if pos.turn == Color.WHITE else -raw

    @staticmethod
    def _captured_value(board: list[int], move: Move) -> int:
        if move.kind is MoveKind.EN_PASSANT:
            return 100  # pawn
        target = board[move.to_sq]
        return ([0, 100, 320, 330, 500, 900, 20_000][target & 7]) if target else 0

    def _order(self, pos: Position, moves: list[Move], depth: int) -> None:
        orderer = self._orderer
        moves.sort(key=lambda m: orderer.score(pos, m, depth), reverse=True)

    def _check_time(self) -> bool:
        if self._nodes & 1023 == 0 and self._time_limit:
            if time.time() - self._start > self._time_limit:
                self._timed_out = True
        return self._timed_out
