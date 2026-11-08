# Code Style

Every class and function follows the [PythonTA](https://www.cs.toronto.edu/~david/pyta/) contract
format, so a reader can see what a piece of code assumes without reading its body.

## Docstrings

- **Classes** have a description, then `Attributes:` (every public attribute and what it means) and
  `Representation Invariants:` (facts that hold between method calls). The attributes are also
  annotated at class level, so the types are checked.
- **Functions and methods** have a one-line summary, then `Preconditions:` for what the caller
  must guarantee. A function with no precondition has no section; a precondition that only repeats a
  type hint is left out.
- **Modules** open with a docstring that says what the module is for.
- Each condition is one valid Python expression on its own `-` line, using the parameter names and,
  in invariants, `self`.

```python
@check_contracts
class Renderer:
    """Draws the shelves, targets, robots and HUD; optionally in a window.

    Attributes:
        rows: The number of rows in the grid.
        cols: The number of columns in the grid.
        width: The width of the frame in pixels.

    Representation Invariants:
        - self.rows > 0 and self.cols > 0
        - self.width == self.cols * CELL
    """

    rows: int
    cols: int
    width: int

    def cell_origin(self, row: int, col: int) -> tuple[int, int]:
        """Top-left pixel `(x, y)` of a cell, below the HUD bar.

        Preconditions:
            - 0 <= row < self.rows
            - 0 <= col < self.cols
        """
```

## Runtime checking

`@check_contracts` turns the docstring conditions into runtime checks, but each wrapped call costs
about 2 ms, so checking is **off by default**. Set `WAREHOUSE_CONTRACTS=1` to turn it on; the
tests do this in `tests/conftest.py`, which makes the suite take about 50 seconds instead of 1.5.

```bash
WAREHOUSE_CONTRACTS=1 python play.py
```

PythonTA silently skips a precondition it cannot evaluate. In methods it cannot see module-level
constants such as `MAX_ROBOTS`, and it leaves keyword and default arguments out of the names it
checks. The constructor therefore keeps its explicit `ValueError` checks, and its preconditions
document them.

## Comments

1. **Never put a comment on the same line as code.** Put it on the line above. This includes
   `# noqa`: fix the cause, or move the exception into the `ruff` configuration.
2. **Docstrings say what; comments say why.** Do not narrate what the next line does. Comment a
   non-obvious reason, a trade-off, or an invariant that spans several lines.
3. **Write full sentences**, ending with a period, within 100 columns. A comment covers the block
   below it, up to the next blank line.
4. **No commented-out code and no placeholders** for features that do not exist (YAGNI). Delete it;
   git remembers.
5. **Keep comments true.** Change the comment in the same commit as the code it explains.
6. The same rules apply to the tests.

```python
# Don't: explains what, and shares a line with code.
self.step_count += 1  # add one to the step count

# Do: explains why, on its own line.
# Rewards are measured against the targets held at the start of the step.
rewards = {agent: self._reward(agent, old, new, collided) for agent in self.agents}
```
