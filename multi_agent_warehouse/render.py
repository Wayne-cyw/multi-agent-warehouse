"""Pygame renderer for the warehouse.

Always draws on an offscreen surface and returns the frame as an RGB array; in window mode it also
shows the frame on screen at a fixed frame rate. Pygame is imported only here, and the env builds
the renderer lazily, so an env without a render mode never loads pygame.
"""

import numpy as np
import pygame
from python_ta.contracts import check_contracts

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


@check_contracts
class Renderer:
    """Draws the shelves, targets, robots and HUD; optionally in a window.

    Attributes:
        shelves: A boolean grid that is True where a shelf stands.
        rows: The number of rows in the grid.
        cols: The number of columns in the grid.
        width: The width of the frame in pixels.
        height: The height of the frame in pixels, including the HUD bar.
        window: Whether frames are also shown in an on-screen window.
        fps: The frame rate of the window.

    Representation Invariants:
        - self.rows > 0 and self.cols > 0
        - self.shelves.shape == (self.rows, self.cols)
        - self.width == self.cols * CELL
        - self.height == HUD_H + self.rows * CELL
        - self.fps > 0
    """

    shelves: np.ndarray
    rows: int
    cols: int
    width: int
    height: int
    window: bool
    fps: int

    def __init__(self, shelves: np.ndarray, window: bool = False, fps: int = RENDER_FPS) -> None:
        """Set up for a boolean `shelves` grid. `window=True` opens a display at `fps`.

        Preconditions:
            - shelves.ndim == 2 and shelves.size > 0
            - fps > 0
        """
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
        """Top-left pixel `(x, y)` of a cell, below the HUD bar.

        Preconditions:
            - 0 <= row < self.rows
            - 0 <= col < self.cols
        """
        return col * CELL, HUD_H + row * CELL

    def draw(
        self, positions: dict[str, Cell], targets: dict[str, Cell], hud: list[str]
    ) -> np.ndarray:
        """Draw one frame and return it as a `(height, width, 3)` uint8 array.

        In window mode the frame is also shown, and the call waits to keep the frame rate.

        Preconditions:
            - not self._closed
            - all(0 <= r < self.rows and 0 <= c < self.cols for r, c in positions.values())
            - all(0 <= r < self.rows and 0 <= c < self.cols for r, c in targets.values())
        """
        self._surface.fill(FLOOR_COLOR)
        self._draw_shelves()
        self._draw_targets(targets)
        self._draw_robots(positions)
        self._draw_hud(hud)

        frame = np.ascontiguousarray(pygame.surfarray.array3d(self._surface).transpose(1, 0, 2))
        self._present()
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

    def _draw_shelves(self) -> None:
        """Fill every shelf cell."""
        for row in range(self.rows):
            for col in range(self.cols):
                if self.shelves[row, col]:
                    x, y = self.cell_origin(row, col)
                    pygame.draw.rect(self._surface, SHELF_COLOR, (x, y, CELL, CELL))

    def _draw_targets(self, targets: dict[str, Cell]) -> None:
        """Outline each robot's target cell in the robot's colour.

        Preconditions:
            - all(agent.startswith("robot_") for agent in targets)
        """
        size = CELL - 2 * TARGET_INSET
        for agent, (row, col) in targets.items():
            x, y = self.cell_origin(row, col)
            rect = (x + TARGET_INSET, y + TARGET_INSET, size, size)
            pygame.draw.rect(self._surface, _robot_color(agent), rect, TARGET_WIDTH)

    def _draw_robots(self, positions: dict[str, Cell]) -> None:
        """Draw each robot as a filled circle centred in its cell.

        Preconditions:
            - all(agent.startswith("robot_") for agent in positions)
        """
        for agent, (row, col) in positions.items():
            x, y = self.cell_origin(row, col)
            center = (x + CELL // 2, y + CELL // 2)
            pygame.draw.circle(self._surface, _robot_color(agent), center, _ROBOT_RADIUS)

    def _present(self) -> None:
        """In window mode, show the surface and wait to keep the frame rate; else do nothing."""
        if self._screen is None or self._clock is None:
            return
        self._screen.blit(self._surface, (0, 0))
        pygame.display.flip()
        pygame.event.pump()
        self._clock.tick(self.fps)

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


@check_contracts
def _robot_color(agent: str) -> tuple[int, int, int]:
    """The colour for a robot named like `robot_3`.

    Preconditions:
        - agent.startswith("robot_") and agent.rsplit("_", 1)[1].isdigit()
    """
    return ROBOT_COLORS[int(agent.rsplit("_", 1)[1]) % len(ROBOT_COLORS)]
