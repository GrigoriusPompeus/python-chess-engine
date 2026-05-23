"""Terminal REPL for playing chess."""

from __future__ import annotations

import sys
from chess_project.core.rules import Game
from chess_project.core.types import Color, Move
from chess_project.io.fen import format_fen
from chess_project.io.notation import parse_uci, move_to_san
from chess_project.engine.interfaces import RandomMover


def main() -> None:
    print("=== Python Chess ===")
    print("Commands: 'quit', 'undo', 'fen', 'moves', 'new'")
    print("Enter moves in UCI format (e.g. e2e4, g1f3, e7e8q)")
    print()

    mode = _choose_mode()
    game = Game()
    bot = RandomMover() if mode != "pvp" else None
    bot_color = Color.BLACK if mode == "pvb_white" else Color.WHITE if mode == "pvb_black" else None

    _play_loop(game, bot, bot_color)


def _choose_mode() -> str:
    print("Choose mode:")
    print("  1) Player vs Player")
    print("  2) Player vs Bot (you play White)")
    print("  3) Player vs Bot (you play Black)")
    while True:
        choice = input("Enter 1/2/3: ").strip()
        if choice == "1":
            return "pvp"
        if choice == "2":
            return "pvb_white"
        if choice == "3":
            return "pvb_black"
        print("Invalid choice.")


def _play_loop(game: Game, bot: RandomMover | None, bot_color: Color | None) -> None:
    while True:
        print()
        print(game.position.board_str())
        print(f"  Turn: {'White' if game.position.turn == Color.WHITE else 'Black'}")
        print(f"  FEN:  {format_fen(game.position)}")

        result = game.outcome()
        if result is not None:
            print(f"\n  Game over: {result[0]} ({result[1]})")
            break

        # Bot turn
        if bot is not None and game.position.turn == bot_color:
            move = bot.choose_move(game.position)
            san = move_to_san(game.position, move)
            game.push(move)
            print(f"  Bot plays: {move.to_uci()} ({san})")
            continue

        # Human turn
        raw = input("  Your move: ").strip().lower()
        if raw == "quit":
            print("Goodbye.")
            break
        if raw == "undo":
            if game.history:
                game.pop()
                if bot is not None and game.history:
                    game.pop()  # also undo bot move
            else:
                print("  Nothing to undo.")
            continue
        if raw == "fen":
            print(f"  {format_fen(game.position)}")
            continue
        if raw == "moves":
            legal = game.legal_moves()
            print(f"  Legal moves ({len(legal)}): ", end="")
            print(" ".join(m.to_uci() for m in legal))
            continue
        if raw == "new":
            game.reset()
            print("  New game started.")
            continue

        # Parse & validate
        try:
            move = parse_uci(game.position, raw)
        except (ValueError, IndexError, KeyError):
            print("  Invalid input. Use UCI format like e2e4")
            continue

        legal = game.legal_moves()
        if move not in legal:
            print("  Illegal move.")
            continue

        san = move_to_san(game.position, move)
        game.push(move)
        print(f"  Played: {san}")


if __name__ == "__main__":
    main()
