"""
Core Chomp game logic.

Two layers live here:

* ``ChompState`` -- an immutable, hashable snapshot of a position. This is what
  players are handed when it is their turn to move (see ``player.py``).

* ``ChompGame`` -- a thin mutable wrapper used to *drive* a game (by the UI in
  ``chompUI.py`` and by the match runner in ``arena.py``).

Rules: the board is a grid of cookies. Eating cell ``(row, col)`` also eats
every cell below-and-right of it (``r >= row and c >= col``). The top-left
cell ``(0, 0)`` is poisoned -- whoever is forced to eat it loses.

State encoding -- the staircase boundary
-----------------------------------------
Because eating a cell removes the whole below-and-right rectangle, the eaten
region always forms a staircase in the bottom-right and the remaining cookies a
staircase in the top-left. A position is therefore fully described by the
boundary between the two, traced as a lattice path from the board's bottom-left
to its top-right corner: a ``0`` is a step right (east, crossing a column) and a
``1`` is a step up (north, climbing the eaten height). Everything below-right of
the path is eaten; everything above-left remains.

The path is exactly ``cols`` zeros and ``rows`` ones, so a position costs
``rows + cols`` bits instead of ``rows * cols`` -- a big saving for an RL
Q-table. The full board has nothing eaten, so its path hugs the bottom then the
right edge: e.g. a 3-high x 5-wide board starts as ``"00000111"``. ``ChompState``
stores this string and uses it directly as the canonical ``key``.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

Move = Tuple[int, int]


def _path_to_widths(path: str, rows: int, cols: int) -> List[int]:
    """
    Decode a boundary path into per-row remaining-cookie counts ``widths[r]``.

    Walks the path tracking the eaten height; each east step (``0``) records the
    eaten height of the column it crosses, from which the remaining height per
    column -- and then the remaining width per row -- is recovered.
    """
    eaten_height_per_col: List[int] = []
    height = 0
    for step in path:
        if step == "1":           # north: climb the eaten boundary
            height += 1
        else:                     # east: finish a column at the current height
            eaten_height_per_col.append(height)
    remaining_per_col = [rows - e for e in eaten_height_per_col]
    return [sum(1 for t in remaining_per_col if t > r) for r in range(rows)]


def _widths_to_path(widths: List[int], rows: int, cols: int) -> str:
    """Encode per-row remaining-cookie counts into the boundary path string."""
    remaining_per_col = [sum(1 for w in widths if w > c) for c in range(cols)]
    eaten_height_per_col = [rows - t for t in remaining_per_col]
    parts: List[str] = []
    prev = 0
    for height in eaten_height_per_col:
        parts.append("1" * (height - prev))   # climb to this column's height
        parts.append("0")                      # cross the column
        prev = height
    parts.append("1" * (rows - prev))          # climb to the top-right corner
    return "".join(parts)


@dataclass(frozen=True)
class ChompState:
    """
    An immutable snapshot of a Chomp position.

    The position is stored as ``path`` -- the staircase-boundary bitstring
    described in the module docstring. ``current_player`` (0 or 1) is the side
    *to move*; players should always reason from the perspective of the side to
    move, since you only lose by eating the poison on *your own* turn.
    """

    rows: int
    cols: int
    path: str
    current_player: int

    @classmethod
    def initial(cls, rows: int, cols: int) -> "ChompState":
        """Returns the full starting board with player 0 to move."""
        return cls(rows, cols, "0" * cols + "1" * rows, 0)

    @property
    def widths(self) -> Tuple[int, ...]:
        """Number of remaining cookies in each row, top to bottom (a partition)."""
        return tuple(_path_to_widths(self.path, self.rows, self.cols))

    @property
    def board(self) -> Tuple[Tuple[int, ...], ...]:
        """
        The position as a grid (0 = remaining cookie, 1 = eaten), reconstructed
        from ``path`` on demand. Convenient for the UI; not how state is stored.
        """
        widths = self.widths
        return tuple(
            tuple(0 if c < widths[r] else 1 for c in range(self.cols))
            for r in range(self.rows)
        )

    @property
    def is_terminal(self) -> bool:
        """
        True once the poison cell has been eaten.

        The poison is gone exactly when column 0 is fully eaten, i.e. the path
        opens with ``rows`` north steps before any east step.
        """
        return self.path.startswith("1" * self.rows)

    @property
    def winner(self) -> Optional[int]:
        """
        The winning player index, or None if the game is not over.

        Whoever ate the poison is the previous mover; since ``current_player``
        flips after every move, the side to move at a terminal position is the
        winner.
        """
        if not self.is_terminal:
            return None
        return self.current_player

    @property
    def key(self) -> str:
        """
        A compact, canonical, hashable encoding of the position: the boundary
        path itself. Use it as a Q-table / transposition-table key. The game is
        impartial, so this key intentionally does NOT include ``current_player``.
        """
        return self.path

    def legal_moves(self) -> List[Move]:
        """All ``(row, col)`` cells that still hold a cookie."""
        widths = self.widths
        return [(r, c) for r in range(self.rows) for c in range(widths[r])]

    def play(self, move: Move) -> "ChompState":
        """
        Returns the new state after the side to move plays ``move``.

        Pure: the current state is left unchanged. Raises ValueError on an
        illegal move so bugs surface immediately during tree search.
        """
        row, col = move
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise ValueError(f"move {move} is off the board")
        widths = _path_to_widths(self.path, self.rows, self.cols)
        if col >= widths[row]:
            raise ValueError(f"move {move} targets an already-eaten cell")

        # Eating (row, col) caps every row at or below it to width `col`.
        for r in range(row, self.rows):
            widths[r] = min(widths[r], col)
        new_path = _widths_to_path(widths, self.rows, self.cols)
        return ChompState(self.rows, self.cols, new_path, 1 - self.current_player)


class ChompGame:
    """
    A mutable game driver wrapping a ``ChompState``.

    Keeps the small, friendly API the UI was written against
    (``make_move``, ``is_game_over``, ``board``, ``current_player``,
    ``rows``, ``cols``) while delegating all rules to ``ChompState``.
    """

    def __init__(self, rows: int, cols: int):
        self.state = ChompState.initial(rows, cols)

    # --- read-only views onto the underlying state ---
    @property
    def rows(self) -> int:
        return self.state.rows

    @property
    def cols(self) -> int:
        return self.state.cols

    @property
    def board(self) -> Tuple[Tuple[int, ...], ...]:
        return self.state.board

    @property
    def current_player(self) -> int:
        return self.state.current_player

    def make_move(self, row: int, col: int) -> bool:
        """
        Plays a move for the current player.

        :return: True if the move was legal and applied, False otherwise.
        """
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return False
        if col >= self.state.widths[row]:   # cell already eaten
            return False
        self.state = self.state.play((row, col))
        return True

    def is_game_over(self) -> bool:
        """True if the poison cell has been eaten."""
        return self.state.is_terminal

    def legal_moves(self) -> List[Move]:
        """All currently legal moves."""
        return self.state.legal_moves()

    def snapshot(self) -> ChompState:
        """The immutable position to hand to a player on its turn."""
        return self.state

    def copy(self) -> "ChompGame":
        """A detached copy that can be advanced independently."""
        g = ChompGame(self.rows, self.cols)
        g.state = self.state  # ChompState is immutable, so sharing is safe
        return g

    def print_board(self) -> None:
        """Prints the current board (0 = cookie, 1 = eaten)."""
        for row in self.state.board:
            print(' '.join(str(cell) for cell in row))
        print()
