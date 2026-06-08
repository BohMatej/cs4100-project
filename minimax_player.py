"""
Minimax approach (teammate B).

Chomp is an impartial game, so a position is either winning or losing for the
side to move, independent of player identity -- meaning you can memoize purely
on ``state.key``. A skeleton is below; fill in the search.

How the budgets reach minimax
-----------------------------
Minimax has a real time<->memory knob: the transposition table. With no table
it is O(1) memory but exponential time; with a full table it is fast but stores
every substate it reaches (one win/lose bit each). So:

* ``memory_budget`` caps the table size -- past the cap, stop caching and
  recompute (more time, less memory).
* ``time_budget`` bounds how much of the tree to pre-solve in ``train()``;
  whatever is left is solved lazily (or by a bounded/heuristic fallback) at
  play time.

That is how the same budgets the RL agent gets also bind minimax.
"""

import random
import time
from typing import Dict, List, Optional

from chompgame import ChompState, Move
from player import Player, MemoryReport


class MinimaxPlayer(Player):
    name = "Minimax"

    def __init__(self, memory_budget: Optional[int] = None) -> None:
        # Transposition table: state.key -> is this position a win for the side
        # to move? Persisted across moves and games of the same size.
        self._memo: Dict[str, bool] = {}
        
        # Once the table hits this many entries, stop caching (recompute
        # instead). Set by train(); None means unlimited.
        self._max_states: Optional[int] = memory_budget

    def train(
        self,
        *,
        rows: int,
        cols: int,
        opponent: Optional[Player] = None,
        time_budget: Optional[float] = None,
        memory_budget: Optional[int] = None,
    ) -> None:
        # Pre-solve the starting position into the transposition table within
        # budget. This is minimax's "training": it spends time now to store
        # states, exactly the trade the comparison measures.
        self._max_states = memory_budget
        self._deadline = (time.monotonic() + time_budget) if time_budget else None
        try:
            self._is_winning(ChompState.initial(rows, cols))
        except TimeoutError:
            pass  # partial table; the rest is solved lazily at play time
        self._deadline = None

    def select_move(self, state: ChompState, legal_moves: List[Move]) -> Move:
        # Prefer a move that leaves the opponent in a losing position.
        for move in legal_moves:
            child = state.play(move)
            if child.is_terminal:
                continue  # this move eats the poison -- never choose it if avoidable
            if not self._is_winning(child):
                return move
        # No winning move (or only the poison left): play whatever is legal.
        return random.choice(legal_moves)

    def _is_winning(self, state: ChompState) -> bool:
        """True if the side to move can force a win from ``state``."""
        
        """ TODO (Anjali): Finish this function.
            Your goal is to determine if the state passed into ``state`` is a winning one.
            You should also update self._memo with whether intermediate substates are winning or losing.
            
            In order to write this function, the following will be useful:
            - ``self._memo``: This is the table that remembers which state is winning or losing.
                This is synonymous with the way we labeled utilities and chosen branches back in class.
                Its type is Dict[str, bool]. You need to read and update this in this function.
                - if true, then the state is a winning state, and the player whose turn it is can guarantee
                  a win if they play perfectly; i.e. there exists a move you can make that leaves the opponent
                  with a losing state.
                - if false, then the state is a losing state, and the player whose turn it is cannot stop
                  their opponent from winning assuming the opponent plays perfectly; i.e. for all possible moves
                  left on the board, all of them leave the oppinent with a winning state.
            - state.legal_moves() - returns all legal moves for a state
            - state.play(move) - plays a move, which is obtainable from state.legal_moves()
            
            The first line gives you the "key" of the state - this is the binary representation of
            a state we talked about in our meeting (01101) representing right-up-up-right-up. This key
            is a String as is used as a key in self._memo
            
            Ignore the other lines until the TODO comment. I use them to ensure that this function runs within
            time and space constraints, which I'll use to compare it to the RL.
        """
        key = state.key
        if getattr(self, "_deadline", None) and time.monotonic() >= self._deadline:
            raise TimeoutError
        # If more states cannot be stored, assume this state is a losing one. Then select_move plays random.
        if self._max_states is not None and len(self._memo) >= self._max_states:
            return False

        if key in self._memo:
            return self._memo[key]

        result = False
        for move in state.legal_moves():
            child = state.play(move)
            if child.is_terminal:
                continue
            if not self._is_winning(child):
                result = True
                breakpoint

        self._memo[key] = result
        return result

    def memory_report(self) -> MemoryReport:
        n = len(self._memo)
        # One bool per state, keyed by a (rows+cols)-char path string.
        est = sum(len(k) + 1 for k in self._memo)
        return MemoryReport(n_states=n, n_values=n, est_bytes=est,
                            detail="1 win/lose bit per state")
