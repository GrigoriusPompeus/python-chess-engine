"""Pygame-based chess GUI — Lichess-inspired design with clock support."""

from __future__ import annotations

import sys
import time
import threading
from pathlib import Path
import pygame
from chess_project.core.position import Position
from chess_project.core.rules import Game
from chess_project.core.types import (
    Color, PieceType, Move, MoveKind,
    piece_color, piece_type_of, sq_name,
)
from chess_project.io.fen import format_fen
from chess_project.io.notation import move_to_san
from chess_project.engine.interfaces import RandomMover
from chess_project.engine.search import AlphaBetaSearch
from chess_project.engine.ordering import MvvLvaOrderer

# ---------------------------------------------------------------------------
# Constants — Lichess color scheme
# ---------------------------------------------------------------------------

SQUARE_SIZE = 76
BOARD_PX = SQUARE_SIZE * 8
PANEL_WIDTH = 300
WINDOW_W = BOARD_PX + PANEL_WIDTH
WINDOW_H = BOARD_PX

# Lichess board colors
LIGHT_SQ = (240, 238, 213)
DARK_SQ = (118, 150, 86)
HIGHLIGHT_SQ = (205, 210, 106, 180)
LEGAL_DOT = (20, 20, 20, 80)
LEGAL_CAPTURE = (20, 20, 20, 60)
CHECK_SQ = (235, 97, 80, 200)
LAST_MOVE_FROM = (170, 162, 58, 160)
LAST_MOVE_TO = (205, 210, 106, 180)

# Panel colors (dark like Lichess)
BG_PANEL = (44, 44, 44)
BG_PANEL_ALT = (55, 55, 55)
TEXT_COLOR = (200, 200, 200)
TEXT_BRIGHT = (255, 255, 255)
COORD_LIGHT = (118, 150, 86)
COORD_DARK = (240, 238, 213)
ACCENT_GREEN = (99, 171, 63)
ACCENT_RED = (200, 72, 72)
CLOCK_BG_ACTIVE = (255, 255, 255)
CLOCK_BG_INACTIVE = (60, 60, 60)
CLOCK_TEXT_ACTIVE = (30, 30, 30)
CLOCK_TEXT_INACTIVE = (150, 150, 150)
BTN_COLOR = (68, 68, 68)
BTN_HOVER = (88, 88, 88)
BTN_ACCENT = (99, 171, 63)

# Unicode chess pieces — use FILLED (black) glyphs for both colors
# We control the color via font rendering, not the unicode codepoint
_PIECE_GLYPH = {
    PieceType.KING:   "\u265A",
    PieceType.QUEEN:  "\u265B",
    PieceType.ROOK:   "\u265C",
    PieceType.BISHOP: "\u265D",
    PieceType.KNIGHT: "\u265E",
    PieceType.PAWN:   "\u265F",
}

# Engine brain presets: (label, key)
# The key is used to construct the right engine in ChessGUI.
ENGINE_PRESETS: list[tuple[str, str]] = [
    ("Random", "random"),
    ("Alpha-Beta", "alphabeta_hce"),
]

# Add NN option only if a trained model exists
_MODEL_PATH = Path(__file__).resolve().parents[3] / "model.pt"
if not _MODEL_PATH.exists():
    # Also check the current working directory
    _MODEL_PATH = Path("model.pt")
if _MODEL_PATH.exists():
    ENGINE_PRESETS.append(("Neural Net", "alphabeta_nn"))

# Time control presets: (label, minutes, increment_seconds)
TIME_PRESETS = [
    ("1+0", 1, 0),
    ("2+1", 2, 1),
    ("3+0", 3, 0),
    ("3+2", 3, 2),
    ("5+0", 5, 0),
    ("5+3", 5, 3),
    ("10+0", 10, 0),
    ("10+5", 10, 5),
    ("15+10", 15, 10),
    ("30+0", 30, 0),
    ("None", 0, 0),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sq_from_pixel(x: int, y: int, flipped: bool) -> int:
    file = x // SQUARE_SIZE
    rank = y // SQUARE_SIZE
    if flipped:
        file = 7 - file
        rank = rank
    else:
        rank = 7 - rank
    return rank * 8 + file


def _pixel_from_sq(sq: int, flipped: bool) -> tuple[int, int]:
    file = sq % 8
    rank = sq // 8
    if flipped:
        px = (7 - file) * SQUARE_SIZE
        py = rank * SQUARE_SIZE
    else:
        px = file * SQUARE_SIZE
        py = (7 - rank) * SQUARE_SIZE
    return px, py


def _format_clock(seconds: float) -> str:
    if seconds <= 0:
        return "0:00"
    m = int(seconds) // 60
    s = int(seconds) % 60
    if seconds < 60:
        tenths = int((seconds - int(seconds)) * 10)
        return f"{s}.{tenths}"
    return f"{m}:{s:02d}"


# ---------------------------------------------------------------------------
# Start screen — mode & time selection
# ---------------------------------------------------------------------------

class StartScreen:
    def __init__(self) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((500, 640))
        pygame.display.set_caption("Python Chess")
        self.clock = pygame.time.Clock()
        self.title_font = pygame.font.SysFont("segoeui", 32, bold=True)
        self.label_font = pygame.font.SysFont("segoeui", 18)
        self.btn_font = pygame.font.SysFont("segoeui", 16, bold=True)
        self.input_font = pygame.font.SysFont("consolas", 16)

        self.mode = "pvp"
        self.engine_idx = 1  # default: Alpha-Beta (HCE)
        self.time_idx = 4  # 5+0 default
        self.custom_min = ""
        self.custom_inc = ""
        self.custom_active: str | None = None  # "min" or "inc"
        self.result: dict | None = None

    def run(self) -> dict | None:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self._handle_click(event.pos)
                elif event.type == pygame.KEYDOWN:
                    self._handle_key(event)

            self._draw()
            self.clock.tick(30)

            if self.result is not None:
                return self.result

        return None

    def _handle_click(self, pos: tuple[int, int]) -> None:
        x, y = pos

        # Mode buttons (y=78..113)
        if 78 <= y <= 113:
            if 30 <= x <= 170:
                self.mode = "pvp"
            elif 180 <= x <= 330:
                self.mode = "pvb_white"
            elif 340 <= x <= 470:
                self.mode = "pvb_black"

        # Engine brain buttons (y=155..190)
        if 155 <= y <= 190:
            btn_w = (440 - 10 * (len(ENGINE_PRESETS) - 1)) // len(ENGINE_PRESETS)
            for i in range(len(ENGINE_PRESETS)):
                bx = 30 + i * (btn_w + 10)
                if bx <= x <= bx + btn_w:
                    self.engine_idx = i
                    return

        # Time preset buttons
        row_start = 242
        for i, (label, mins, inc) in enumerate(TIME_PRESETS):
            col = i % 4
            row = i // 4
            bx = 30 + col * 115
            by = row_start + row * 50
            if bx <= x <= bx + 105 and by <= y <= by + 40:
                self.time_idx = i
                self.custom_active = None
                return

        # Custom time inputs
        custom_y = row_start + (len(TIME_PRESETS) // 4 + 1) * 50
        if custom_y <= y <= custom_y + 35:
            if 120 <= x <= 200:
                self.custom_active = "min"
                self.time_idx = -1
                return
            elif 280 <= x <= 360:
                self.custom_active = "inc"
                self.time_idx = -1
                return

        # Play button
        play_y = custom_y + 60
        if 150 <= x <= 350 and play_y <= y <= play_y + 50:
            self._start_game()

    def _handle_key(self, event: pygame.event.Event) -> None:
        if event.key == pygame.K_RETURN:
            self._start_game()
            return
        if self.custom_active == "min":
            if event.key == pygame.K_BACKSPACE:
                self.custom_min = self.custom_min[:-1]
            elif event.unicode.isdigit() and len(self.custom_min) < 4:
                self.custom_min += event.unicode
            elif event.key == pygame.K_TAB:
                self.custom_active = "inc"
        elif self.custom_active == "inc":
            if event.key == pygame.K_BACKSPACE:
                self.custom_inc = self.custom_inc[:-1]
            elif event.unicode.isdigit() and len(self.custom_inc) < 3:
                self.custom_inc += event.unicode
            elif event.key == pygame.K_TAB:
                self.custom_active = "min"

    def _start_game(self) -> None:
        if self.time_idx == -1:
            # Custom
            mins = int(self.custom_min) if self.custom_min else 5
            inc = int(self.custom_inc) if self.custom_inc else 0
        elif self.time_idx < len(TIME_PRESETS):
            _, mins, inc = TIME_PRESETS[self.time_idx]
        else:
            mins, inc = 0, 0

        engine_key = ENGINE_PRESETS[self.engine_idx][1]
        self.result = {
            "mode": self.mode,
            "minutes": mins,
            "increment": inc,
            "engine": engine_key,
        }

    def _draw(self) -> None:
        self.screen.fill((38, 36, 33))

        # Title
        title = self.title_font.render("Python Chess", True, TEXT_BRIGHT)
        self.screen.blit(title, (160, 22))

        # Mode selection
        modes = [("Player vs Player", "pvp", 30), ("Play as White", "pvb_white", 180), ("Play as Black", "pvb_black", 340)]
        for label, mode, bx in modes:
            color = BTN_ACCENT if self.mode == mode else BTN_COLOR
            pygame.draw.rect(self.screen, color, (bx, 78, 130, 35), border_radius=5)
            surf = self.btn_font.render(label, True, TEXT_BRIGHT)
            rect = surf.get_rect(center=(bx + 65, 95))
            self.screen.blit(surf, rect)

        # Engine brain selector
        label = self.label_font.render("Engine Brain", True, TEXT_COLOR)
        self.screen.blit(label, (30, 128))

        btn_w = (440 - 10 * (len(ENGINE_PRESETS) - 1)) // len(ENGINE_PRESETS)
        for i, (elabel, _ekey) in enumerate(ENGINE_PRESETS):
            bx = 30 + i * (btn_w + 10)
            color = BTN_ACCENT if self.engine_idx == i else BTN_COLOR
            pygame.draw.rect(self.screen, color, (bx, 155, btn_w, 35), border_radius=5)
            surf = self.btn_font.render(elabel, True, TEXT_BRIGHT)
            rect = surf.get_rect(center=(bx + btn_w // 2, 172))
            self.screen.blit(surf, rect)

        # Time section label
        label = self.label_font.render("Time Control", True, TEXT_COLOR)
        self.screen.blit(label, (30, 210))

        # Time preset buttons
        row_start = 242
        for i, (lbl, mins, inc) in enumerate(TIME_PRESETS):
            col = i % 4
            row = i // 4
            bx = 30 + col * 115
            by = row_start + row * 50
            color = BTN_ACCENT if self.time_idx == i else BTN_COLOR
            pygame.draw.rect(self.screen, color, (bx, by, 105, 40), border_radius=5)
            surf = self.btn_font.render(lbl, True, TEXT_BRIGHT)
            rect = surf.get_rect(center=(bx + 52, by + 20))
            self.screen.blit(surf, rect)

        # Custom time inputs
        custom_y = row_start + (len(TIME_PRESETS) // 4 + 1) * 50
        label = self.label_font.render("Custom:", True, TEXT_COLOR)
        self.screen.blit(label, (30, custom_y + 7))

        # Minutes input
        min_color = BTN_ACCENT if self.custom_active == "min" else (80, 80, 80)
        pygame.draw.rect(self.screen, min_color, (120, custom_y, 80, 35), 2, border_radius=4)
        min_text = self.custom_min if self.custom_min else "min"
        min_surf = self.input_font.render(min_text, True, TEXT_BRIGHT if self.custom_min else (120, 120, 120))
        self.screen.blit(min_surf, (130, custom_y + 8))

        plus = self.label_font.render("+", True, TEXT_COLOR)
        self.screen.blit(plus, (215, custom_y + 7))

        # Increment input
        inc_color = BTN_ACCENT if self.custom_active == "inc" else (80, 80, 80)
        pygame.draw.rect(self.screen, inc_color, (240, custom_y, 80, 35), 2, border_radius=4)
        inc_text = self.custom_inc if self.custom_inc else "sec"
        inc_surf = self.input_font.render(inc_text, True, TEXT_BRIGHT if self.custom_inc else (120, 120, 120))
        self.screen.blit(inc_surf, (250, custom_y + 8))

        # Play button
        play_y = custom_y + 60
        pygame.draw.rect(self.screen, BTN_ACCENT, (150, play_y, 200, 50), border_radius=8)
        play_surf = self.title_font.render("Play", True, TEXT_BRIGHT)
        play_rect = play_surf.get_rect(center=(250, play_y + 25))
        self.screen.blit(play_surf, play_rect)

        pygame.display.flip()


# ---------------------------------------------------------------------------
# Engine factory
# ---------------------------------------------------------------------------

def _build_engine(key: str) -> object:
    """Construct the engine that matches the start-screen selection."""
    if key == "random":
        return RandomMover()
    if key == "alphabeta_nn":
        from chess_project.engine.ml_evaluator import NNEvaluator
        model_path = Path(__file__).resolve().parents[3] / "model.pt"
        if not model_path.exists():
            model_path = Path("model.pt")
        evaluator = NNEvaluator(str(model_path), device="cpu")
        return AlphaBetaSearch(evaluator=evaluator, orderer=MvvLvaOrderer(), max_depth=33)
    # Default: alpha-beta with hand-crafted evaluation
    return AlphaBetaSearch(orderer=MvvLvaOrderer(), max_depth=33)


# ---------------------------------------------------------------------------
# Main GUI class
# ---------------------------------------------------------------------------

class ChessGUI:
    def __init__(
        self,
        mode: str = "pvp",
        flipped: bool = False,
        time_minutes: int = 5,
        increment: int = 0,
        engine: str = "hce",
    ) -> None:
        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        pygame.display.set_caption("Python Chess")
        self.clock = pygame.time.Clock()

        self.game = Game()
        self.flipped = flipped
        self.mode = mode

        # Engine: selected from the start screen
        if mode != "pvp":
            self.bot = _build_engine(engine)
        else:
            self.bot = None
        self.bot_color = (
            Color.BLACK if mode == "pvb_white"
            else Color.WHITE if mode == "pvb_black"
            else None
        )

        self.selected_sq: int | None = None
        self.legal_targets: list[Move] = []
        self.last_move: Move | None = None
        self.move_log: list[str] = []
        self.promotion_pending: list[Move] | None = None
        self.status_msg = ""
        self.engine_key = engine

        # Bot thinking runs in a background thread to keep the GUI responsive
        self._bot_thinking = False
        self._bot_result: Move | None = None

        # Clock
        self.has_clock = time_minutes > 0
        self.increment = increment
        self.initial_time = time_minutes * 60.0
        self.time_left = [self.initial_time, self.initial_time]  # [white, black]
        self.clock_running = False
        self.last_tick = time.time()
        self.game_started = False

        # Fonts
        self.piece_font = pygame.font.SysFont("segoeuisymbol", SQUARE_SIZE - 10)
        self.coord_font = pygame.font.SysFont("consolas", 13)
        self.panel_font = pygame.font.SysFont("segoeui", 14)
        self.status_font = pygame.font.SysFont("segoeui", 18, bold=True)
        self.btn_font = pygame.font.SysFont("segoeui", 14, bold=True)
        self.clock_font = pygame.font.SysFont("consolas", 28, bold=True)
        self.clock_font_small = pygame.font.SysFont("consolas", 18, bold=True)
        self.move_font = pygame.font.SysFont("consolas", 13)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self._handle_click(event.pos)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_u:
                        self._undo()
                    elif event.key == pygame.K_n:
                        self._new_game()
                    elif event.key == pygame.K_f:
                        self.flipped = not self.flipped
                    elif event.key == pygame.K_q:
                        running = False

            # Clock tick
            if self.has_clock and self.clock_running and not self.game.is_game_over():
                now = time.time()
                elapsed = now - self.last_tick
                self.last_tick = now
                turn_idx = int(self.game.position.turn)
                self.time_left[turn_idx] -= elapsed
                if self.time_left[turn_idx] <= 0:
                    self.time_left[turn_idx] = 0
                    self.clock_running = False
                    winner = "White" if turn_idx == 1 else "Black"
                    self.status_msg = f"{winner} wins on time!"

            # Check if the background engine thread has produced a move
            if self._bot_result is not None:
                move = self._bot_result
                self._bot_result = None
                self._bot_thinking = False
                self._do_move(move)

            # Kick off bot thinking in a background thread
            if (self.bot is not None
                    and not self._bot_thinking
                    and not self.game.is_game_over()
                    and self.game.position.turn == self.bot_color
                    and self.promotion_pending is None
                    and not (self.has_clock and self.time_left[int(self.bot_color)] <= 0)):
                self._start_bot_think()

            self._draw()
            self.clock.tick(60)

        pygame.quit()

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------

    def _handle_click(self, pos: tuple[int, int]) -> None:
        x, y = pos

        # Panel button clicks
        if x >= BOARD_PX:
            self._handle_panel_click(x, y)
            return

        sq = _sq_from_pixel(x, y, self.flipped)

        # Promotion chooser
        if self.promotion_pending is not None:
            self._handle_promotion_click(sq)
            return

        if self.game.is_game_over():
            return

        # Ignore clicks while the engine is thinking
        if self._bot_thinking:
            return

        # If clock ran out
        if self.has_clock and self.time_left[int(self.game.position.turn)] <= 0:
            return

        # If bot's turn, ignore
        if self.bot is not None and self.game.position.turn == self.bot_color:
            return

        if self.selected_sq is None:
            self._select_piece(sq)
        else:
            self._try_move(sq)

    def _select_piece(self, sq: int) -> None:
        piece = self.game.position.board[sq]
        if piece != 0 and piece_color(piece) == self.game.position.turn:
            self.selected_sq = sq
            self.legal_targets = [
                m for m in self.game.legal_moves() if m.from_sq == sq
            ]
        else:
            self.selected_sq = None
            self.legal_targets = []

    def _try_move(self, sq: int) -> None:
        if sq == self.selected_sq:
            self.selected_sq = None
            self.legal_targets = []
            return

        piece = self.game.position.board[sq]
        if piece != 0 and piece_color(piece) == self.game.position.turn:
            self._select_piece(sq)
            return

        matches = [m for m in self.legal_targets if m.to_sq == sq]
        if not matches:
            self.selected_sq = None
            self.legal_targets = []
            return

        if any(m.promotion is not None for m in matches):
            promos = [m for m in matches if m.promotion is not None]
            if len(promos) > 1:
                self.promotion_pending = promos
                return

        self._do_move(matches[0])

    def _handle_promotion_click(self, sq: int) -> None:
        if self.promotion_pending is None:
            return
        target = self.promotion_pending[0].to_sq
        tx, ty = _pixel_from_sq(target, self.flipped)

        for i, move in enumerate(self.promotion_pending):
            row_y = ty + i * SQUARE_SIZE
            mx, my = pygame.mouse.get_pos()
            if tx <= mx < tx + SQUARE_SIZE and row_y <= my < row_y + SQUARE_SIZE:
                self.promotion_pending = None
                self._do_move(move)
                return

        self.promotion_pending = None
        self.selected_sq = None
        self.legal_targets = []

    def _handle_panel_click(self, x: int, y: int) -> None:
        pad = 15
        clock_h = 50
        btn_h = 32
        btn_w = (PANEL_WIDTH - pad * 2 - 10) // 2

        bottom_clock_y = WINDOW_H - pad - clock_h
        hints_y = bottom_clock_y - 20 if self.has_clock else WINDOW_H - 20
        btn_row_y = hints_y - btn_h - 8

        undo_x = BOARD_PX + pad
        new_x = BOARD_PX + pad + btn_w + 10

        # Undo button
        if undo_x <= x <= undo_x + btn_w and btn_row_y <= y <= btn_row_y + btn_h:
            self._undo()
        # New Game button
        if new_x <= x <= new_x + btn_w and btn_row_y <= y <= btn_row_y + btn_h:
            self._new_game()

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _do_move(self, move: Move) -> None:
        san = move_to_san(self.game.position, move)
        if self.game.position.turn == Color.WHITE:
            self.move_log.append(f"{self.game.position.fullmove_number}. {san}")
        else:
            if self.move_log:
                self.move_log[-1] += f" {san}"
            else:
                self.move_log.append(f"... {san}")

        # Clock: add increment for the player who just moved
        if self.has_clock and self.game_started:
            turn_idx = int(self.game.position.turn)
            self.time_left[turn_idx] += self.increment

        self.game.push(move)
        self.last_move = move
        self.selected_sq = None
        self.legal_targets = []
        self._update_status()

        # Start clock after first move
        if self.has_clock and not self.game_started:
            self.game_started = True
            self.clock_running = True
            self.last_tick = time.time()
        elif self.has_clock:
            self.last_tick = time.time()

    def _start_bot_think(self) -> None:
        """Launch engine search in a background thread."""
        self._bot_thinking = True
        self.status_msg = "Thinking..."
        # Snapshot the position so the engine works on a copy
        pos_copy = self.game.position.copy()
        time_ms: int | None = None
        if self.has_clock:
            remaining = self.time_left[int(self.bot_color)]
            time_ms = min(int(remaining * 1000 // 20), 3000)

        # Build Zobrist hashes of all positions seen so far in this game,
        # so the engine can avoid repeating them.
        from chess_project.engine.search import zobrist_hash as _zh, AlphaBetaSearch
        history_hashes: list[int] | None = None
        if isinstance(self.bot, AlphaBetaSearch):
            history_hashes = []
            # Replay the game to collect position hashes
            tmp = self.game.position.copy()
            # Undo all moves to get each historical position
            # Simpler: compute from the stored position keys concept
            # We can compute the current hash and all prior ones by
            # walking game.history in reverse
            # Actually, easiest: rebuild from initial position
            from chess_project.io.fen import parse_fen
            initial_fen = self.game.headers.get("FEN", None)
            tmp_pos = parse_fen(initial_fen) if initial_fen else Position()
            history_hashes.append(_zh(tmp_pos))
            for move, _undo in self.game.history:
                tmp_pos.make_move(move)
                history_hashes.append(_zh(tmp_pos))

        def _worker() -> None:
            self._bot_result = self.bot.choose_move(
                pos_copy, time_ms=time_ms, history_hashes=history_hashes,
            )

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

    def _undo(self) -> None:
        if self._bot_thinking:
            return
        if not self.game.history:
            return
        self.game.pop()
        if self.bot is not None and self.game.history:
            self.game.pop()
        self.selected_sq = None
        self.legal_targets = []
        self.promotion_pending = None
        self.last_move = self.game.history[-1][0] if self.game.history else None
        if self.move_log:
            self.move_log.pop()
        self._update_status()

    def _new_game(self) -> None:
        self._bot_thinking = False
        self._bot_result = None
        self.game.reset()
        self.selected_sq = None
        self.legal_targets = []
        self.last_move = None
        self.move_log.clear()
        self.promotion_pending = None
        self.status_msg = ""
        if self.has_clock:
            self.time_left = [self.initial_time, self.initial_time]
            self.clock_running = False
            self.game_started = False

    def _update_status(self) -> None:
        result = self.game.outcome()
        if result is not None:
            self.status_msg = f"{result[0]}  ({result[1]})"
            self.clock_running = False
        else:
            turn = "White" if self.game.position.turn == Color.WHITE else "Black"
            check = " (check!)" if self.game.position.is_in_check() else ""
            self.status_msg = f"{turn} to move{check}"

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _draw(self) -> None:
        self.screen.fill(BG_PANEL)
        self._draw_board()
        self._draw_overlays()
        self._draw_pieces()
        if self.promotion_pending:
            self._draw_promotion_chooser()
        self._draw_panel()
        pygame.display.flip()

    def _draw_board(self) -> None:
        for sq in range(64):
            px, py = _pixel_from_sq(sq, self.flipped)
            r, f = sq // 8, sq % 8
            color = LIGHT_SQ if (r + f) % 2 == 0 else DARK_SQ
            pygame.draw.rect(self.screen, color, (px, py, SQUARE_SIZE, SQUARE_SIZE))

        # Coordinates on the edge squares (Lichess style)
        for i in range(8):
            # File labels on bottom rank
            file_idx = i if not self.flipped else 7 - i
            label = "abcdefgh"[file_idx]
            is_light = (0 + file_idx) % 2 == 0  # rank 0
            color = COORD_DARK if not is_light else COORD_LIGHT
            surf = self.coord_font.render(label, True, color)
            self.screen.blit(surf, (i * SQUARE_SIZE + SQUARE_SIZE - 12, BOARD_PX - 14))

            # Rank labels on left column
            rank_idx = 7 - i if not self.flipped else i
            rank_label = str(rank_idx + 1)
            file_0 = 0 if not self.flipped else 7
            is_light = (rank_idx + file_0) % 2 == 0
            color = COORD_DARK if not is_light else COORD_LIGHT
            surf = self.coord_font.render(rank_label, True, color)
            self.screen.blit(surf, (3, i * SQUARE_SIZE + 2))

    def _draw_pieces(self) -> None:
        board = self.game.position.board
        for sq in range(64):
            piece = board[sq]
            if piece == 0:
                continue
            px, py = _pixel_from_sq(sq, self.flipped)
            pt = piece_type_of(piece)
            color = piece_color(piece)
            ch = _PIECE_GLYPH[pt]
            cx = px + SQUARE_SIZE // 2
            cy = py + SQUARE_SIZE // 2

            if color == Color.WHITE:
                # Dark outline (render slightly offset in all directions)
                outline_color = (40, 40, 40)
                outline_surf = self.piece_font.render(ch, True, outline_color)
                for dx, dy in [(-1,-1),(0,-1),(1,-1),(-1,0),(1,0),(-1,1),(0,1),(1,1)]:
                    r = outline_surf.get_rect(center=(cx + dx, cy + dy))
                    self.screen.blit(outline_surf, r)
                # White fill on top
                surf = self.piece_font.render(ch, True, (255, 255, 255))
            else:
                # Light outline for black pieces
                outline_surf = self.piece_font.render(ch, True, (200, 200, 200))
                for dx, dy in [(-1,-1),(0,-1),(1,-1),(-1,0),(1,0),(-1,1),(0,1),(1,1)]:
                    r = outline_surf.get_rect(center=(cx + dx, cy + dy))
                    self.screen.blit(outline_surf, r)
                # Black fill on top
                surf = self.piece_font.render(ch, True, (30, 30, 30))

            rect = surf.get_rect(center=(cx, cy))
            self.screen.blit(surf, rect)

    def _draw_overlays(self) -> None:
        overlay = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)

        # Last move highlight
        if self.last_move is not None:
            overlay.fill(LAST_MOVE_FROM)
            self.screen.blit(overlay, _pixel_from_sq(self.last_move.from_sq, self.flipped))
            overlay.fill(LAST_MOVE_TO)
            self.screen.blit(overlay, _pixel_from_sq(self.last_move.to_sq, self.flipped))

        # Selected square
        if self.selected_sq is not None:
            overlay.fill(HIGHLIGHT_SQ)
            self.screen.blit(overlay, _pixel_from_sq(self.selected_sq, self.flipped))

        # Legal move indicators
        for move in self.legal_targets:
            px, py = _pixel_from_sq(move.to_sq, self.flipped)
            dot_surface = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
            if self.game.position.board[move.to_sq] != 0 or move.kind == MoveKind.EN_PASSANT:
                # Capture — triangular corners or thick ring
                pygame.draw.circle(dot_surface, LEGAL_CAPTURE,
                                   (SQUARE_SIZE // 2, SQUARE_SIZE // 2),
                                   SQUARE_SIZE // 2 - 2, 5)
            else:
                # Quiet — small filled circle
                pygame.draw.circle(dot_surface, LEGAL_DOT,
                                   (SQUARE_SIZE // 2, SQUARE_SIZE // 2), 12)
            self.screen.blit(dot_surface, (px, py))

        # Check highlight
        if self.game.position.is_in_check():
            check_surf = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
            # Radial gradient effect (simplified as solid)
            check_surf.fill(CHECK_SQ)
            king_sq = self.game.position.king_sq[self.game.position.turn]
            self.screen.blit(check_surf, _pixel_from_sq(king_sq, self.flipped))

    def _draw_promotion_chooser(self) -> None:
        if self.promotion_pending is None:
            return
        target = self.promotion_pending[0].to_sq
        color = piece_color(self.game.position.board[self.promotion_pending[0].from_sq])
        tx, ty = _pixel_from_sq(target, self.flipped)

        # Dark overlay on whole board
        dark = pygame.Surface((BOARD_PX, BOARD_PX), pygame.SRCALPHA)
        dark.fill((0, 0, 0, 100))
        self.screen.blit(dark, (0, 0))

        pieces = [PieceType.QUEEN, PieceType.ROOK, PieceType.BISHOP, PieceType.KNIGHT]
        for i, pt in enumerate(pieces):
            ry = ty + i * SQUARE_SIZE
            pygame.draw.rect(self.screen, (240, 240, 240), (tx, ry, SQUARE_SIZE, SQUARE_SIZE))
            pygame.draw.rect(self.screen, (100, 100, 100), (tx, ry, SQUARE_SIZE, SQUARE_SIZE), 1)
            ch = _PIECE_GLYPH[pt]
            if color == Color.WHITE:
                surf = self.piece_font.render(ch, True, (255, 255, 255))
            else:
                surf = self.piece_font.render(ch, True, (30, 30, 30))
            rect = surf.get_rect(center=(tx + SQUARE_SIZE // 2, ry + SQUARE_SIZE // 2))
            self.screen.blit(surf, rect)

    def _draw_panel(self) -> None:
        panel_x = BOARD_PX
        pygame.draw.rect(self.screen, BG_PANEL, (panel_x, 0, PANEL_WIDTH, WINDOW_H))

        # Layout constants
        pad = 15
        clock_h = 50
        btn_h = 32
        btn_w = (PANEL_WIDTH - pad * 2 - 10) // 2  # two buttons side by side

        # --- Top clock ---
        top_y = pad
        if self.has_clock:
            top_color_idx = int(Color.WHITE) if self.flipped else int(Color.BLACK)
            self._draw_clock(panel_x + pad, top_y, top_color_idx)
            content_top = top_y + clock_h + 12
        else:
            content_top = top_y

        # --- Bottom clock ---
        bottom_clock_y = WINDOW_H - pad - clock_h
        if self.has_clock:
            bot_color_idx = int(Color.BLACK) if self.flipped else int(Color.WHITE)
            self._draw_clock(panel_x + pad, bottom_clock_y, bot_color_idx)

        # --- Buttons row (above bottom clock) ---
        hints_y = bottom_clock_y - 20 if self.has_clock else WINDOW_H - 20
        btn_row_y = hints_y - btn_h - 8

        # Undo button
        undo_x = panel_x + pad
        pygame.draw.rect(self.screen, BTN_COLOR, (undo_x, btn_row_y, btn_w, btn_h), border_radius=5)
        surf = self.btn_font.render("Undo [U]", True, TEXT_COLOR)
        rect = surf.get_rect(center=(undo_x + btn_w // 2, btn_row_y + btn_h // 2))
        self.screen.blit(surf, rect)

        # New Game button
        new_x = panel_x + pad + btn_w + 10
        pygame.draw.rect(self.screen, BTN_ACCENT, (new_x, btn_row_y, btn_w, btn_h), border_radius=5)
        surf = self.btn_font.render("New [N]", True, TEXT_BRIGHT)
        rect = surf.get_rect(center=(new_x + btn_w // 2, btn_row_y + btn_h // 2))
        self.screen.blit(surf, rect)

        # Keyboard hints
        hints = self.coord_font.render("[F]lip board    [Q]uit", True, (90, 90, 90))
        self.screen.blit(hints, (panel_x + pad, hints_y))

        # --- Status ---
        status_text = self.status_msg or "White to move"
        is_game_over = self.game.is_game_over() or (self.has_clock and min(self.time_left) <= 0)
        status_color = ACCENT_GREEN if not is_game_over else ACCENT_RED
        status = self.status_font.render(status_text, True, status_color)
        self.screen.blit(status, (panel_x + pad, content_top))

        # --- Move list ---
        move_y = content_top + 28
        move_bottom = btn_row_y - 10
        max_lines = (move_bottom - move_y) // 18
        display_moves = self.move_log[-max_lines:]

        for i, line in enumerate(display_moves):
            ly = move_y + i * 18
            if i % 2 == 0:
                pygame.draw.rect(self.screen, BG_PANEL_ALT, (panel_x + 10, ly, PANEL_WIDTH - 20, 18), border_radius=2)
            surf = self.move_font.render(line, True, TEXT_COLOR)
            self.screen.blit(surf, (panel_x + pad, ly + 2))

    def _draw_clock(self, x: int, y: int, color_idx: int) -> None:
        is_active = (int(self.game.position.turn) == color_idx
                     and self.clock_running
                     and not self.game.is_game_over())
        bg = CLOCK_BG_ACTIVE if is_active else CLOCK_BG_INACTIVE
        text_color = CLOCK_TEXT_ACTIVE if is_active else CLOCK_TEXT_INACTIVE

        w = PANEL_WIDTH - 30
        h = 50
        pygame.draw.rect(self.screen, bg, (x, y, w, h), border_radius=6)

        time_str = _format_clock(self.time_left[color_idx])
        label = "White" if color_idx == 0 else "Black"

        label_surf = self.panel_font.render(label, True, text_color)
        self.screen.blit(label_surf, (x + 10, y + 5))

        clock_surf = self.clock_font.render(time_str, True, text_color)
        rect = clock_surf.get_rect(midright=(x + w - 10, y + h // 2))
        self.screen.blit(clock_surf, rect)

        # Low time warning
        if self.time_left[color_idx] < 30 and self.time_left[color_idx] > 0:
            pygame.draw.rect(self.screen, ACCENT_RED, (x, y, w, h), 2, border_radius=6)


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def run_gui(
    mode: str = "pvp",
    time_minutes: int = 5,
    increment: int = 0,
    engine: str = "hce",
) -> None:
    flipped = mode == "pvb_black"
    gui = ChessGUI(
        mode=mode, flipped=flipped,
        time_minutes=time_minutes, increment=increment,
        engine=engine,
    )
    gui.run()


def main() -> None:
    pygame.init()
    start = StartScreen()
    config = start.run()
    if config is None:
        pygame.quit()
        return

    run_gui(
        mode=config["mode"],
        time_minutes=config["minutes"],
        increment=config["increment"],
        engine=config["engine"],
    )


if __name__ == "__main__":
    main()
