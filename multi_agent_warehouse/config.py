"""Settings and the warehouse map. Edit this file to change the game.

`LAYOUT` has one string per row: `.` is free floor and `#` is a shelf. It must be rectangular; the
environment reads its height and width from it.
"""

LAYOUT: tuple[str, ...] = (
    "........",
    ".##..##.",
    ".##..##.",
    "........",
    "........",
    ".##..##.",
    ".##..##.",
    "........",
)

DEFAULT_N_ROBOTS = 3
MAX_ROBOTS = 6
DEFAULT_MAX_STEPS = 200

DELIVERY_REWARD = 1.0
STEP_PENALTY = -0.01
COLLISION_PENALTY = -0.1
SHAPING_SCALE = 0.02

RENDER_MODES = ("human", "rgb_array")
RENDER_FPS = 10
