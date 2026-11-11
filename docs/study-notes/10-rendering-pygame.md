# 10. Rendering with Pygame

[Index](README.md) · Previous: [09 move resolution](09-move-resolution.md) · Next: [11 policies and demo](11-policies-and-demo.md)

> **Status:** implemented (Stage 2, Task 3) in `multi_agent_warehouse/render.py`, with
> `env.render()` and `env.close()` wired to it. The numbers below come from the spec and are
> asserted by `tests/test_render.py`.

**Pygame** is a Python library for drawing, opening a window and reading keyboard events. In this
project only `render.py` imports it.

## Offscreen-first

The renderer is built so there is one drawing path:

1. It **always** draws to an in-memory `pygame.Surface`.
2. It converts that to a NumPy array of shape `(height_px, width_px, 3)`, dtype `uint8`
   (this is the `rgb_array` mode).
3. In `human` mode it additionally **blits** (copies) that surface to a real window at `render_fps`
   (10 by default).

Why: the same code serves the live window, the README GIF and the automated tests, and tests can run
on machines with no display by setting `SDL_VIDEODRIVER=dummy` ([13](13-testing-and-tooling.md)).

## Render modes

The standard PettingZoo and Gymnasium convention:

| `render_mode` | `env.render()` returns | Window? |
| --- | --- | --- |
| `None` | `None` | No (and Pygame is never imported) |
| `"rgb_array"` | the pixel array | No |
| `"human"` | `None` | Yes |

The constructor already validates the mode against `RENDER_MODES` from `config.py` *(verified)*.

## Layout and numbers

Drawing sizes live in `render.py`; only game settings live in `config.py`.

| Item | Value |
| --- | --- |
| HUD bar | 40 px tall: step, deliveries, collisions and a free-text note (for example `greedy | PAUSED`) |
| Cell size | 64 px |
| Frame size | `(40 + 8*64, 8*64, 3) = (552, 512, 3)` |
| Floor / shelves | light grey / dark brown |
| Robot | filled circle in its own colour (six distinct colours, one per possible robot) |
| Target | outlined square, inset 8 px, line width 6, in the same colour as its robot |

Exact numbers are fixed in the spec so tests can assert exact pixels, for example that the centre of
robot_0's cell equals `ROBOT_COLORS[0]` and the centre of a shelf cell equals `SHELF_COLOR`.

## Lazy creation

`MultiAgentWarehouseEnv` with no render mode must never import Pygame. So the renderer is created
**lazily on the first `render()` call**, and `render_mode=None` makes `render()` a no-op returning
`None` that opens no window.

## A NumPy and Pygame trap

Pygame arrays come out as `(x, y)` = `(width, height)`, but images are conventionally
`(rows, cols)` = `(height, width)`. Hence the **transpose** after `pygame.surfarray.array3d`.
Forgetting it gives a sideways image that still "works", which is why the plan tests the exact frame
shape.

## Rendering must not change the result

The renderer never touches the environment's random generator, so `render()` cannot change the
trajectory. The plan pins this with a test: two envs with the same seed and actions, one rendering
every step, produce identical observations and rewards.

## Check yourself

1. Why does the renderer always draw offscreen, even in `human` mode?
2. What does `render()` return in each of the three modes?
3. Why is Pygame imported lazily?
4. Why is the frame `(552, 512, 3)` and not `(512, 552, 3)`?
5. How does the plan guarantee that watching the demo cannot change a result?
