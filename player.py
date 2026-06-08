"""
The common interface every Chomp bot implements.

Both teammates subclass ``Player``. The same interface serves two roles:

* the **agent** being evaluated, and
* the **opponent / environment player** another agent trains against
  (e.g. the random-then-self-play environment in the RL approach).

A player is asked to move only on its own turn, so it should always reason
from the perspective of the side to move (``state.current_player``).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import random
from typing import List, Optional

from chompgame import ChompState, Move


@dataclass
class MemoryReport:
    """
    A player's self-reported memory footprint, in a form comparable across
    very different algorithms.

    The whole point of the comparison: Q-learning stores a value per
    ``(state, action)`` pair, while minimax stores one win/lose bit per state.
    Both maintain a table keyed by state, so the asymmetry is captured by the
    ratio ``n_values / n_states``:

    * minimax  -> 1.0   (one bit per stored state)
    * Q-learning -> avg branching factor over the states it actually visited

    Counts are the conceptually meaningful numbers; ``est_bytes`` is a rough
    common axis (it does not try to be exact about Python object overhead).
    """

    n_states: int          # distinct positions stored in the table
    n_values: int          # scalar values stored (state-action pairs, or bits)
    est_bytes: int = 0     # rough footprint estimate, for a single shared axis
    detail: str = ""       # free-form note for the report

    @property
    def values_per_state(self) -> float:
        return self.n_values / self.n_states if self.n_states else 0.0


class Player(ABC):
    """Base class for all Chomp players."""

    #: Human-readable name used in match reports. Override in subclasses.
    name: str = "Player"

    @abstractmethod
    def select_move(self, state: ChompState, legal_moves: List[Move]) -> Move:
        """
        Choose a move for the side to move.

        :param state: an immutable snapshot of the current position. Use
            ``state.key`` as a Q-table / transposition-table key and
            ``state.play(move)`` to look ahead without mutating anything.
        :param legal_moves: the moves available now (also obtainable via
            ``state.legal_moves()``; passed in so you needn't recompute it).
        :return: one move ``(row, col)`` drawn from ``legal_moves``.
        """
        ...

    def train(
        self,
        *,
        rows: int,
        cols: int,
        opponent: Optional["Player"] = None,
        time_budget: Optional[float] = None,
        memory_budget: Optional[int] = None,
    ) -> None:
        """
        Prepare the player for games on a ``rows`` x ``cols`` board, under the
        given time and memory budgets. This is the single place both budgets
        are applied, so it is how the comparison stays fair across approaches.

        It is *not* RL-only. Minimax should use it too: pre-solve the target
        board into its transposition table within the budget (and cap that
        table at ``memory_budget``). A player that genuinely needs no
        preparation can leave the default no-op.

        :param opponent: the environment player to train against. If None, the
            implementation may default to a baseline such as ``RandomPlayer``.
        :param time_budget: soft wall-clock limit in seconds, or None for
            unlimited. The harness also measures actual time spent here.
        :param memory_budget: soft cap on stored states (see ``memory_report``),
            or None for unlimited. Advisory -- Python cannot enforce it
            strictly, so honor it by checking ``memory_report().n_states``.
        """
        return None

    def memory_report(self) -> MemoryReport:
        """
        Report the player's current memory footprint for the comparison.

        Overridden by table-based players (RL, minimax). Measuring this from
        outside the object is unreliable (``sys.getsizeof`` counts interpreter
        overhead, not stored states), so each player reports its own counts.
        The default suits stateless players like ``RandomPlayer``.
        """
        return MemoryReport(n_states=0, n_values=0, detail="stateless")


class RandomPlayer(Player):
    """
    Picks a uniformly random legal move.

    Serves as the baseline opponent and as the initial environment strategy
    for the reinforcement-learning approach. Seedable for reproducible runs.
    """

    name = "Random"

    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)

    def select_move(self, state: ChompState, legal_moves: List[Move]) -> Move:
        return self._rng.choice(legal_moves)
