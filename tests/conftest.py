"""Shared test setup: a headless display and runtime contract checking switched on."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("WAREHOUSE_CONTRACTS", "1")
