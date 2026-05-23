"""PGN (Portable Game Notation) import and export."""

from __future__ import annotations

import re
from chess_project.core.rules import Game
from chess_project.core.types import Color
from chess_project.io.fen import format_fen, STARTING_FEN
from chess_project.io.notation import move_to_san, parse_san


# ------------------------------------------------------------------
# Export
# ------------------------------------------------------------------

def write_game(game: Game) -> str:
    """Serialise a Game to a PGN string."""
    lines: list[str] = []

    # Tag pairs
    result = game.headers.get("Result", "*")
    for tag in ("Event", "Site", "Date", "Round", "White", "Black", "Result"):
        value = game.headers.get(tag, "?")
        lines.append(f'[{tag} "{value}"]')
    lines.append("")

    # Reconstruct SAN move-text by replaying from the start
    temp = Game(fen=STARTING_FEN)
    tokens: list[str] = []
    for i, (move, _undo) in enumerate(game.history):
        if temp.position.turn == Color.WHITE:
            tokens.append(f"{temp.position.fullmove_number}.")
        san = move_to_san(temp.position, move)
        tokens.append(san)
        temp.push(move)

    tokens.append(result)
    # Wrap at ~80 chars
    text = ""
    line = ""
    for tok in tokens:
        if len(line) + len(tok) + 1 > 80 and line:
            text += line.rstrip() + "\n"
            line = ""
        line += tok + " "
    text += line.rstrip()
    lines.append(text)
    lines.append("")
    return "\n".join(lines)


# ------------------------------------------------------------------
# Import
# ------------------------------------------------------------------

_TAG_RE = re.compile(r'\[(\w+)\s+"(.*)"\]')
_MOVE_NUM_RE = re.compile(r"\d+\.+")
_RESULT_TOKENS = {"1-0", "0-1", "1/2-1/2", "*"}


def read_game(pgn_text: str) -> Game:
    """Parse a single PGN game from text."""
    game = Game()
    lines = pgn_text.strip().splitlines()

    movetext_lines: list[str] = []
    for line in lines:
        line = line.strip()
        m = _TAG_RE.match(line)
        if m:
            game.headers[m.group(1)] = m.group(2)
            continue
        if not line:
            continue
        movetext_lines.append(line)

    movetext = " ".join(movetext_lines)
    # Remove comments
    movetext = re.sub(r"\{[^}]*\}", "", movetext)
    # Remove variations (simple one-level)
    movetext = re.sub(r"\([^)]*\)", "", movetext)
    # Remove NAGs
    movetext = re.sub(r"\$\d+", "", movetext)

    tokens = movetext.split()
    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue
        if _MOVE_NUM_RE.fullmatch(tok):
            continue
        if tok in _RESULT_TOKENS:
            game.headers["Result"] = tok
            break
        move = parse_san(game.position, tok)
        game.push(move)

    return game
