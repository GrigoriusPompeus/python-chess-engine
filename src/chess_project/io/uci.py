"""Universal Chess Interface (UCI) protocol loop.

Run with::

    python -m chess_project.io.uci

Or with the NN evaluator::

    python -m chess_project.io.uci --model model.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from chess_project.core.types import Move, MoveKind, PieceType, sq_from_name
from chess_project.core.position import Position
from chess_project.engine.search import AlphaBetaSearch
from chess_project.engine.ordering import MvvLvaOrderer
from chess_project.io.fen import parse_fen, STARTING_FEN


def _uci_to_move(pos: Position, uci: str) -> Move:
    """Match a UCI string (e.g. ``e2e4``, ``e7e8q``) to a legal move."""
    legal = pos.generate_legal_moves()
    for m in legal:
        if m.to_uci() == uci:
            return m
    raise ValueError(f"Illegal or unrecognised UCI move: {uci}")


def _send(line: str) -> None:
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def uci_loop(engine: AlphaBetaSearch) -> None:
    """Blocking stdin loop implementing the UCI protocol."""
    pos = Position()

    while True:
        try:
            raw = input()
        except EOFError:
            break

        line = raw.strip()
        if not line:
            continue

        tokens = line.split()
        cmd = tokens[0]

        # ---- uci --------------------------------------------------------
        if cmd == "uci":
            _send("id name ChessProject")
            _send("id author ChessProject")
            _send("uciok")

        # ---- isready -----------------------------------------------------
        elif cmd == "isready":
            _send("readyok")

        # ---- ucinewgame --------------------------------------------------
        elif cmd == "ucinewgame":
            pos = Position()
            engine._tt.clear()

        # ---- position ----------------------------------------------------
        elif cmd == "position":
            idx = 1
            if tokens[idx] == "startpos":
                pos = Position()
                idx = 2
            elif tokens[idx] == "fen":
                fen_parts: list[str] = []
                idx = 2
                while idx < len(tokens) and tokens[idx] != "moves":
                    fen_parts.append(tokens[idx])
                    idx += 1
                pos = parse_fen(" ".join(fen_parts))
            # Apply moves
            if idx < len(tokens) and tokens[idx] == "moves":
                idx += 1
                while idx < len(tokens):
                    move = _uci_to_move(pos, tokens[idx])
                    pos.make_move(move)
                    idx += 1

        # ---- go ----------------------------------------------------------
        elif cmd == "go":
            time_ms: int | None = None
            depth: int | None = None

            i = 1
            while i < len(tokens):
                if tokens[i] == "depth" and i + 1 < len(tokens):
                    depth = int(tokens[i + 1])
                    i += 2
                elif tokens[i] == "movetime" and i + 1 < len(tokens):
                    time_ms = int(tokens[i + 1])
                    i += 2
                elif tokens[i] == "wtime" and i + 1 < len(tokens):
                    # Simple time management: use 1/30th of remaining time
                    from chess_project.core.types import Color
                    if pos.turn == Color.WHITE:
                        time_ms = max(100, int(tokens[i + 1]) // 30)
                    i += 2
                elif tokens[i] == "btime" and i + 1 < len(tokens):
                    from chess_project.core.types import Color
                    if pos.turn == Color.BLACK:
                        time_ms = max(100, int(tokens[i + 1]) // 30)
                    i += 2
                elif tokens[i] == "infinite":
                    time_ms = None
                    depth = None
                    i += 1
                else:
                    i += 1

            best = engine.choose_move(pos, time_ms=time_ms, depth=depth)
            _send(f"info nodes {engine.nodes_searched}")
            _send(f"bestmove {best.to_uci()}")

        # ---- quit --------------------------------------------------------
        elif cmd == "quit":
            break

        # ---- unknown (ignore) --------------------------------------------
        # UCI spec says GUIs may send unknown commands; engines should ignore.


def main() -> None:
    parser = argparse.ArgumentParser(description="UCI chess engine")
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to PyTorch model.pt (omit to use HCE evaluator)",
    )
    parser.add_argument(
        "--depth", type=int, default=33,
        help="Maximum search depth (default: 33)",
    )
    args = parser.parse_args()

    evaluator = None
    if args.model and Path(args.model).exists():
        from chess_project.engine.ml_evaluator import NNEvaluator
        evaluator = NNEvaluator(args.model, device="cpu")

    engine = AlphaBetaSearch(
        evaluator=evaluator,
        orderer=MvvLvaOrderer(),
        max_depth=args.depth,
    )
    uci_loop(engine)


if __name__ == "__main__":
    main()
