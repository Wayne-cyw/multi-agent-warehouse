"""Random seed utilities for multi-seed training and evaluation experiments."""

import secrets

TRAIN_SEED_LIMIT = 1000
EVAL_FLOOR = 10_000
EVAL_LIMIT = 2**31


def draw_seeds(n: int) -> list[int]:
    """Draw `n` distinct random training seeds in `[0, 1000)`.

    Uses OS entropy, not NumPy's global state.

    Preconditions:
        - 0 <= n <= TRAIN_SEED_LIMIT
    """
    if not 0 <= n <= TRAIN_SEED_LIMIT:
        raise ValueError(f"n must be between 0 and {TRAIN_SEED_LIMIT}, got {n}")
    pool = list(range(TRAIN_SEED_LIMIT))
    chosen = []
    for _ in range(n):
        chosen.append(pool.pop(secrets.randbelow(len(pool))))
    return chosen


def draw_eval_base(n_episodes: int = 100) -> int:
    """Draw a random first evaluation seed so `base .. base + n_episodes - 1` fits in
    `[10_000, 2**31)`, which never overlaps the training seeds `[0, 1000)`.

    Preconditions:
        - 0 < n_episodes <= EVAL_LIMIT - EVAL_FLOOR
    """
    return EVAL_FLOOR + secrets.randbelow(EVAL_LIMIT - EVAL_FLOOR - n_episodes + 1)
