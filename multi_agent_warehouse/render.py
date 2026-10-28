"""Pygame renderer for the warehouse.

Always draws on an offscreen surface and returns the frame as an RGB array; in window mode it also
shows the frame on screen at a fixed frame rate. Pygame is imported only here, and the env builds
the renderer lazily, so an env without a render mode never loads pygame.
"""

import numpy as np
import pygame

from multi_agent_warehouse.config import RENDER_FPS
from multi_agent_warehouse.rules import Cell

CELL = 64
HUD_H = 40
TARGET_INSET = 8
TARGET_WIDTH = 6

FLOOR_COLOR = (220, 220, 220)
SHELF_COLOR = (92, 64, 51)
HUD_BG_COLOR = (30, 30, 30)
HUD_TEXT_COLOR = (240, 240, 240)
ROBOT_COLORS: tuple[tuple[int, int, int], ...] = (
    (230, 57, 70),
    (29, 111, 214),
    (46, 160, 67),
    (240, 160, 20),
    (142, 68, 173),
    (0, 160, 176),
)

_ROBOT_RADIUS = CELL // 2 - 10
_HUD_GAP = 14
_HUD_PAD = 10


class Renderer:
    """Draws the shelves, targets, robots and HUD; optionally in a window."""

    def __init__(self, shelves: np.ndarray, window: bool = False, fps: int = RENDER_FPS) -> None:
        """Set up for a boolean `shelves` grid. `window=True` opens a display at `fps`."""
        self.shelves = shelves
        self.rows, self.cols = shelves.shape
        self.width = self.cols * CELL
        self.height = HUD_H + self.rows * CELL
        self.window = window
        self.fps = fps
        self._closed = False

        pygame.font.init()
        self._font = pygame.font.Font(None, 24)
        self._surface = pygame.Surface((self.width, self.height))
        self._screen: pygame.Surface | None = None
        self._clock: pygame.time.Clock | None = None
        if window:
            pygame.display.init()
            self._screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Multi-agent warehouse")
            self._clock = pygame.time.Clock()

    def cell_origin(self, row: int, col: int) -> tuple[int, int]:
        """Top-left pixel `(x, y)` of a cell, below the HUD bar."""
        return col * CELL, HUD_H + row * CELL

    def draw(
        self, positions: dict[str, Cell], targets: dict[str, Cell], hud: list[str]
    ) -> np.ndarray:
        """Draw one frame and return it as a `(height, width, 3)` uint8 array.

        In window mode the frame is also shown, and the call waits to keep the frame rate.
        """
        surface = self._surface
        surface.fill(FLOOR_COLOR)
        for row in range(self.rows):
            for col in range(self.cols):
                if self.shelves[row, col]:
                    x, y = self.cell_origin(row, col)
                    pygame.draw.rect(surface, SHELF_COLOR, (x, y, CELL, CELL))

        for agent, (row, col) in targets.items():
            x, y = self.cell_origin(row, col)
            size = CELL - 2 * TARGET_INSET
            rect = (x + TARGET_INSET, y + TARGET_INSET, size, size)
            pygame.draw.rect(surface, _robot_color(agent), rect, TARGET_WIDTH)

        for agent, (row, col) in positions.items():
            x, y = self.cell_origin(row, col)
            pygame.draw.circle(
                surface, _robot_color(agent), (x + CELL // 2, y + CELL // 2), _ROBOT_RADIUS
            )

        self._draw_hud(hud)

        frame = np.ascontiguousarray(pygame.surfarray.array3d(surface).transpose(1, 0, 2))
        if self._screen is not None and self._clock is not None:
            self._screen.blit(surface, (0, 0))
            pygame.display.flip()
            pygame.event.pump()
            self._clock.tick(self.fps)
        return frame

    def close(self) -> None:
        """Release the window if there is one. Safe to call more than once."""
        if self._closed:
            return
        self._closed = True
        if self.window:
            pygame.display.quit()
        self._screen = None
        self._clock = None

    def _draw_hud(self, lines: list[str]) -> None:
        """Draw the HUD bar with the non-empty `lines` side by side on one row."""
        pygame.draw.rect(self._surface, HUD_BG_COLOR, (0, 0, self.width, HUD_H))
        x = _HUD_PAD
        for line in lines:
            if not line:
                continue
            text = self._font.render(line, True, HUD_TEXT_COLOR)
            self._surface.blit(text, (x, (HUD_H - text.get_height()) // 2))
            x += text.get_width() + _HUD_GAP


def _robot_color(agent: str) -> tuple[int, int, int]:
    """The colour for a robot named like `robot_3`."""
    return ROBOT_COLORS[int(agent.rsplit("_", 1)[1]) % len(ROBOT_COLORS)]
