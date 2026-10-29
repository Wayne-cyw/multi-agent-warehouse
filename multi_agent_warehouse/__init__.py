"""Multi-Agent Warehouse: a small multi-robot PettingZoo environment with a Pygame renderer.

Runtime contract checking (the docstring preconditions and invariants) is slow, so it is off
unless the environment variable `WAREHOUSE_CONTRACTS` is `1`. The flag must be set before the
submodules are imported, because the `check_contracts` decorators read it when they run.
"""

import os

import python_ta.contracts

python_ta.contracts.ENABLE_CONTRACT_CHECKING = os.environ.get("WAREHOUSE_CONTRACTS") == "1"

from multi_agent_warehouse.env import MultiAgentWarehouseEnv

__all__ = ["MultiAgentWarehouseEnv"]
