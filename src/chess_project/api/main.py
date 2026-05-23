"""FastAPI server for stateless chess game management.

Run with::

    uvicorn chess_project.api.main:app --reload

Environment variables
---------------------
MONGO_URI      : MongoDB connection string  (default ``mongodb://localhost:27017``)
MONGO_DB       : Database name              (default ``chess``)
MODEL_PATH     : Path to PyTorch model.pt   (default ``model.pt``)
EVAL_DEVICE    : ``cpu`` | ``cuda``          (default auto-detect)
SEARCH_DEPTH   : Default engine depth       (default ``20``)
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator

from beanie import init_beanie
from fastapi import FastAPI, HTTPException
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field

from chess_project.core.types import MoveKind, PieceType, sq_from_name
from chess_project.core.rules import Game
from chess_project.db.models import ChessGame, GameStatus, MoveRecord, Player
from chess_project.io.fen import parse_fen, format_fen


# ---------------------------------------------------------------------------
# Config from environment
# ---------------------------------------------------------------------------

MONGO_URI: str = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB: str = os.getenv("MONGO_DB", "chess")
MODEL_PATH: str = os.getenv("MODEL_PATH", "model.pt")
EVAL_DEVICE: str | None = os.getenv("EVAL_DEVICE")
SEARCH_DEPTH: int = int(os.getenv("SEARCH_DEPTH", "20"))


# ---------------------------------------------------------------------------
# Lifespan: Mongo + engine init
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # --- startup ---
    client: AsyncIOMotorClient = AsyncIOMotorClient(MONGO_URI)  # type: ignore[type-arg]
    await init_beanie(
        database=client[MONGO_DB],
        document_models=[ChessGame],
    )
    app.state.mongo_client = client

    # Load search engine (with ML or HCE evaluator)
    from chess_project.engine.search import AlphaBetaSearch
    from chess_project.engine.ordering import MvvLvaOrderer

    evaluator = None
    if Path(MODEL_PATH).exists():
        from chess_project.engine.ml_evaluator import NNEvaluator
        evaluator = NNEvaluator(MODEL_PATH, device=EVAL_DEVICE)
        print(f"Loaded NN evaluator from {MODEL_PATH}")
    else:
        print(f"No model at {MODEL_PATH}, using HCE evaluator")

    app.state.engine = AlphaBetaSearch(
        evaluator=evaluator,
        orderer=MvvLvaOrderer(),
        max_depth=SEARCH_DEPTH,
    )

    yield

    # --- shutdown ---
    client.close()


app = FastAPI(
    title="Chess Engine API",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class CreateGameRequest(BaseModel):
    game_id: str
    fen: str | None = None
    white: Player = Field(default_factory=Player)
    black: Player = Field(default_factory=Player)


class CreateGameResponse(BaseModel):
    game_id: str
    fen: str
    status: GameStatus


class MakeMoveRequest(BaseModel):
    time_ms: int | None = None
    depth: int | None = None


class MakeMoveResponse(BaseModel):
    engine_move: str
    fen: str
    status: GameStatus
    result_reason: str | None = None


class GameStateResponse(BaseModel):
    game_id: str
    fen: str
    moves: list[str]
    status: GameStatus
    result_reason: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reconstruct_game(doc: ChessGame) -> Game:
    """Rebuild a ``Game`` from the DB document by replaying moves on the
    initial FEN.  This is stateless — we never store mutable Python objects."""
    game = Game(fen=doc.initial_fen)
    for rec in doc.moves:
        legal = game.legal_moves()
        matched = [m for m in legal if m.to_uci() == rec.uci]
        if not matched:
            raise HTTPException(
                status_code=500,
                detail=f"Stored move {rec.uci} is not legal — DB corrupt?",
            )
        game.push(matched[0])
    return game


async def _get_doc(game_id: str) -> ChessGame:
    doc = await ChessGame.find_one(ChessGame.game_id == game_id)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"Game {game_id!r} not found")
    return doc


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/v1/game", response_model=CreateGameResponse, status_code=201)
async def create_game(req: CreateGameRequest) -> CreateGameResponse:
    """Create a new game, optionally from a custom FEN."""
    existing = await ChessGame.find_one(ChessGame.game_id == req.game_id)
    if existing:
        raise HTTPException(status_code=409, detail="Game ID already exists")

    fen = req.fen or "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    # Validate FEN
    try:
        parse_fen(fen)
    except (ValueError, IndexError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid FEN: {exc}")

    doc = ChessGame(
        game_id=req.game_id,
        fen=fen,
        initial_fen=fen,
        white=req.white,
        black=req.black,
    )
    await doc.insert()
    return CreateGameResponse(game_id=doc.game_id, fen=doc.fen, status=doc.status)


@app.post("/api/v1/game/{game_id}/move", response_model=MakeMoveResponse)
async def engine_move(game_id: str, req: MakeMoveRequest) -> MakeMoveResponse:
    """Ask the engine to play a move in an existing game.

    Workflow:
    1. Fetch the game document from Mongo.
    2. Reconstruct the ``Game`` state by replaying moves on the initial FEN.
    3. Invoke ``SearchService.choose_move()`` to get the engine's move.
    4. Apply the move, update FEN / status in Mongo, return the result.
    """
    doc = await _get_doc(game_id)
    if doc.status != GameStatus.IN_PROGRESS:
        raise HTTPException(status_code=400, detail="Game is already finished")

    game = _reconstruct_game(doc)

    engine: AlphaBetaSearch = app.state.engine  # type: ignore[name-defined]
    move = engine.choose_move(
        game.position,
        time_ms=req.time_ms,
        depth=req.depth,
    )
    game.push(move)

    # Persist
    doc.moves.append(MoveRecord(uci=move.to_uci()))
    doc.fen = format_fen(game.position)
    doc.updated_at = datetime.now(timezone.utc)

    outcome = game.outcome()
    if outcome is not None:
        result, reason = outcome
        doc.result_reason = reason
        if result == "1-0":
            doc.status = GameStatus.WHITE_WINS
        elif result == "0-1":
            doc.status = GameStatus.BLACK_WINS
        else:
            doc.status = GameStatus.DRAW

    await doc.save()

    return MakeMoveResponse(
        engine_move=move.to_uci(),
        fen=doc.fen,
        status=doc.status,
        result_reason=doc.result_reason,
    )


@app.post("/api/v1/game/{game_id}/play", response_model=MakeMoveResponse)
async def player_move(game_id: str, uci: str) -> MakeMoveResponse:
    """Submit a human/client move in UCI notation (e.g. ``e2e4``)."""
    doc = await _get_doc(game_id)
    if doc.status != GameStatus.IN_PROGRESS:
        raise HTTPException(status_code=400, detail="Game is already finished")

    game = _reconstruct_game(doc)
    legal = game.legal_moves()
    matched = [m for m in legal if m.to_uci() == uci]
    if not matched:
        legal_uci = [m.to_uci() for m in legal]
        raise HTTPException(
            status_code=400,
            detail=f"Illegal move {uci!r}. Legal: {legal_uci}",
        )

    game.push(matched[0])

    doc.moves.append(MoveRecord(uci=uci))
    doc.fen = format_fen(game.position)
    doc.updated_at = datetime.now(timezone.utc)

    outcome = game.outcome()
    if outcome is not None:
        result, reason = outcome
        doc.result_reason = reason
        if result == "1-0":
            doc.status = GameStatus.WHITE_WINS
        elif result == "0-1":
            doc.status = GameStatus.BLACK_WINS
        else:
            doc.status = GameStatus.DRAW

    await doc.save()

    return MakeMoveResponse(
        engine_move=uci,
        fen=doc.fen,
        status=doc.status,
        result_reason=doc.result_reason,
    )


@app.get("/api/v1/game/{game_id}", response_model=GameStateResponse)
async def get_game(game_id: str) -> GameStateResponse:
    """Retrieve current game state."""
    doc = await _get_doc(game_id)
    return GameStateResponse(
        game_id=doc.game_id,
        fen=doc.fen,
        moves=[r.uci for r in doc.moves],
        status=doc.status,
        result_reason=doc.result_reason,
    )
