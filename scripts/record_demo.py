"""Record a greedy episode of the warehouse to docs/multi_agent_warehouse/demo.gif."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from pathlib import Path  # noqa: E402

from PIL import Image  # noqa: E402

from multi_agent_warehouse.agents import GreedyPolicy  # noqa: E402
from multi_agent_warehouse.env import MultiAgentWarehouseEnv  # noqa: E402

SEED = 0
STEPS = 100
OUTPUT = Path(__file__).resolve().parent.parent / "docs" / "multi_agent_warehouse" / "demo.gif"


def main() -> None:
    """Run a greedy episode and save its frames as a looping GIF."""
    env = MultiAgentWarehouseEnv(render_mode="rgb_array")
    policy = GreedyPolicy()
    observations, _ = env.reset(seed=SEED)
    policy.reset(SEED)
    frames = [Image.fromarray(env.render())]
    for _ in range(STEPS):
        if not env.agents:
            break
        observations, *_ = env.step(policy.act(observations))
        frames.append(Image.fromarray(env.render()))
    env.close()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(OUTPUT, save_all=True, append_images=frames[1:], duration=100, loop=0)
    print(f"wrote {OUTPUT} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
