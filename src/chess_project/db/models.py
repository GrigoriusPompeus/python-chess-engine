"""MongoDB document models for persistent chess games (Beanie ODM)."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Annotated

from beanie import Document, Indexed
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Embedded sub-documents
# ---------------------------------------------------------------------------

class GameStatus(str, Enum):
    IN_PROGRESS = "in_progress"
    WHITE_WINS = "white_wins"
    BLACK_WINS = "black_wins"
    DRAW = "draw"


class Player(BaseModel):
    name: str = "?"
    rating: int | None = None
    is_engine: bool = False


class MoveRecord(BaseModel):
    """One half-move stored in UCI notation with an optional clock stamp."""
    uci: str
    clock_ms: int | None = None


# ---------------------------------------------------------------------------
# Main document
# ---------------------------------------------------------------------------

class ChessGame(Document):
    """Represents a single game persisted in MongoDB.

    Fields
    ------
    game_id : str
        Human-readable or externally-generated identifier (indexed, unique).
    fen : str
        *Current* board state as a FEN string.  Updated after every move.
    initial_fen : str
        The FEN the game started from (immutable after creation).
    moves : list[MoveRecord]
        Full ordered move history in UCI notation.
    white / black : Player
        Player info for each side.
    status : GameStatus
        Current game status; updated by the API after each move.
    result_reason : str | None
        ``"checkmate"``, ``"stalemate"``, ``"fifty-move rule"``, etc.
    created_at / updated_at : datetime
        Timestamps (UTC).
    """

    game_id: Annotated[str, Indexed(unique=True)]
    fen: str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    initial_fen: str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    moves: list[MoveRecord] = Field(default_factory=list)
    white: Player = Field(default_factory=Player)
    black: Player = Field(default_factory=Player)
    status: GameStatus = GameStatus.IN_PROGRESS
    result_reason: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Settings:
        name = "chess_games"
