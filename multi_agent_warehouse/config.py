"""Settings and the warehouse map. Edit this file to change the game.

Each entry of `LAYOUTS` has one string per row: `.` is free floor and `#` is a shelf. A layout must
be rectangular, connected and 8 by 8; `LAYOUT` is the default one. `TRAIN_LAYOUTS` are the ones the
randomised-layout training uses and `HELDOUT_LAYOUTS` are only seen at evaluation.
"""

LAYOUTS: dict[str, tuple[str, ...]] = {
    "default": (
        "........",
        ".##..##.",
        ".##..##.",
        "........",
        "........",
        ".##..##.",
        ".##..##.",
        "........",
    ),
    "open": (
        "........",
        "........",
        "..##....",
        "..##....",
        "....##..",
        "....##..",
        "........",
        "........",
    ),
    "aisles": (
        "........",
        ".#.##.#.",
        ".#.##.#.",
        ".#....#.",
        ".#....#.",
        ".#.##.#.",
        ".#.##.#.",
        "........",
    ),
    "scatter": (
        "........",
        ".#...#..",
        "....#...",
        "..#...#.",
        ".....#..",
        "#...#...",
        "...#...#",
        "........",
    ),
    "wide_gap": (
        "........",
        "##....##",
        "##....##",
        "........",
        "........",
        "##....##",
        "##....##",
        "........",
    ),
    "maze_lite": (
        "........",
        ".####.#.",
        "......#.",
        ".#.##.#.",
        ".#....#.",
        ".#.####.",
        ".#......",
        "........",
    ),
}
LAYOUT: tuple[str, ...] = LAYOUTS["default"]
TRAIN_LAYOUTS: tuple[str, ...] = ("default", "open", "aisles", "scatter")
HELDOUT_LAYOUTS: tuple[str, ...] = ("wide_gap", "maze_lite")

DEFAULT_N_ROBOTS = 3
MAX_ROBOTS = 6
DEFAULT_MAX_STEPS = 200

DELIVERY_REWARD = 1.0
STEP_PENALTY = -0.01
COLLISION_PENALTY = -0.1
SHAPING_SCALE = 0.02

RENDER_MODES = ("human", "rgb_array")
RENDER_FPS = 10

OBSERVATION_MODES = ("basic", "rich")
OBS_SIZES = {"basic": 8, "rich": 14}
