"""Pluggable AI interfaces and a baseline random mover."""

from __future__ import annotations

import random
from chess_project.core.types import Move, SearchService, Evaluator, MoveOrderer
from chess_project.core.position import Position


class RandomMover:
    """Picks a random legal move — useful as a baseline / test opponent."""

    def choose_move(
        self,
        position: Position,
        *,
        time_ms: int | None = None,
        depth: int | None = None,
    ) -> Move:
        legal = position.generate_legal_moves()
        if not legal:
            raise RuntimeError("No legal moves available")
        return random.choice(legal)
