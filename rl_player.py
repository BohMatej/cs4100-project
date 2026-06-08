"""
Reinforcement-learning approach (teammate A).

Plan: Q-learning / SARSA where the environment is an opponent ``Player``. Start
with a RandomPlayer opponent; after training, fold the learned policy into the
opponent's strategy, reset the agent's Q-table, and repeat -- iteratively
strengthening the environment.

Use ``state.key`` as the Q-table key (the game is impartial, so the key omits
whose turn it is). Reward: +1 for a win, -1 for a loss, given at the terminal.
A skeleton is below; fill in the learning loop.
"""

import random
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from chompgame import ChompState, Move
from player import Player, RandomPlayer, MemoryReport

GAMMA = 1.0


def _copy_q(q: Dict[str, Dict[Move, float]]) -> Dict[str, Dict[Move, float]]:
    """A deep-enough copy of a Q-table (snapshots the inner action dicts too)."""
    return {state: dict(actions) for state, actions in q.items()}


class RLPlayer(Player):
    name = "RL"

    def __init__(self, alpha: float = 0.1, epsilon: float = 0.1,
                 episodes_per_generation: int = 5000,
                 num_generations: int = 50,
                 q_table: Optional[Dict[str, Dict[Move, float]]] = None):
        self.alpha = alpha       # learning rate

        """ Remember that gamma is 1 after the discussion we had on saturday;
            since it's a finite game we do not discount future rewards """
        self.epsilon = epsilon   # exploration rate during training

        # How many self-play episodes make up ONE generation against a fixed
        # opponent. After a full generation finishes, the opponent is upgraded
        # to the freshly trained table and Q is reset (see train). Sam: feel
        # free to swap this fixed count for a convergence test instead.
        self.episodes_per_generation = episodes_per_generation

        # How many generations train() runs before stopping on its own. This
        # bounds total training at episodes_per_generation * num_generations
        # episodes, so train() always terminates even with no time/memory
        # budget. A budget can still cut it short earlier.
        self.num_generations = num_generations

        """ Q[state.key][move] -> value. defaultdict keeps unseen states cheap.
            state.key is a property of ChompState that returns the state in the
            "011000101.." format we talked about on saturday. Its type is str
            and only consists of 0s and 1s."""
        self.Q: Dict[str, Dict[Move, float]] = defaultdict(dict)

        """Seed a ready-made strategy directly from a Q-table. This is how the
            "environment" opponent for a generation is built: a player that just
            plays greedily (via select_move) off a previously trained table and
            never trains itself."""
        if q_table is not None:
            self.Q.update(_copy_q(q_table))

    @classmethod
    def from_qtable(cls, q_table: Dict[str, Dict[Move, float]]) -> "RLPlayer":
        """Build a (non-training) player whose strategy is the given Q-table."""
        return cls(q_table=q_table)

    def select_move(self, state: ChompState, legal_moves: List[Move]) -> Move:
        """
        NOTE to Sam: This function is NOT to be called to pick a move
        for the agent we are currently training.

        This is simply what the player themselves will do when actually playing a
        game for real. The move selection when actually training should be done in
        the train function, or some private helper idgaf.

        You will probably use this function to pick the "environment" (opponent) player's move
        in order to update the state for the agent you're training.
        """
        q = self.Q.get(state.key, {})
        if not q:
            return legal_moves[0]
        return max(legal_moves, key=lambda m: q.get(m, 0.0))

    def train(
        self,
        *,
        rows: int,
        cols: int,
        opponent: Optional[Player] = None,
        time_budget: Optional[float] = None,
        memory_budget: Optional[int] = None,
    ) -> None:
        opponent = opponent or RandomPlayer()
        deadline = (time.monotonic() + time_budget) if time_budget else None

        # Last fully-trained table. This will be the strategy adopted by the RL player once training is done.
        committed: Optional[Dict[str, Dict[Move, float]]] = None
        try:
            for _ in range(self.num_generations):
                finished = self._train_one_generation(
                    rows, cols, opponent, deadline, memory_budget)
                if not finished:
                    # Budget hit mid-generation: throw away this partial table.
                    break
                # Generation complete: remember it, promote it to the opponent
                # so the next generation trains against a stronger player, and
                # reset the learner to start fresh against it.
                committed = _copy_q(self.Q)
                opponent = RLPlayer.from_qtable(committed)
                self.Q = defaultdict(dict)
        finally:
            # Fall back to the last fully-trained generation.
            if committed is not None:
                self.Q = defaultdict(dict, _copy_q(committed))

    def _train_one_generation(self, rows: int, cols: int, opponent: Player,
                              deadline: Optional[float],
                              memory_budget: Optional[int]) -> bool:
        """
        Run one generation of self-play against the (fixed) ``opponent``.
        A generation basically trains against one specific opponent: after 5000
        (or whatever you set `episodes_per_generation` to be) the current Q-table
        is used to construct a new opponent and the agent's Q-table is reset.

        :return: True if the whole generation completed, False if a budget was
            hit partway through (so train() can discard the partial table).
        """
        for _ in range(self.episodes_per_generation):
            # Budget checks (Matej uses these for the comparison; they bound how
            # long / how big training gets). Hitting either aborts the generation.
            if deadline and time.monotonic() >= deadline:
                return False
            if memory_budget and len(self.Q) >= memory_budget:
                return False

            """ TODO (Sam): play one self-play episode against `opponent`,
                applying the Q-learning update on the agent's own moves, with
                terminal reward +1 (win) / -1 (loss). rows/cols are the board
                dimensions; opponent is the environment player.
            """
            self._run_episode(rows, cols, opponent)
        return True

    def _run_episode(self, rows: int, cols: int, opponent: Player) -> None:

        if not hasattr(self, "_N") or len(self.Q) == 0:
            self._N = defaultdict(dict)

        def backup(s_key: str, move: Move, reward: float,
                   next_state: Optional[ChompState]) -> None:
            n_sa = self._N[s_key].get(move, 0.0)
            eta = 1.0 / (1.0 + n_sa)
            if next_state is None:
                target = reward
            else:
                nq = self.Q.get(next_state.key, {})
                target = reward + GAMMA * max(
                    (nq.get(m, 0.0) for m in next_state.legal_moves()), default=0.0)
            old = self.Q[s_key].get(move, 0.0)
            self.Q[s_key][move] = (1.0 - eta) * old + eta * target
            self._N[s_key][move] = n_sa + 1.0

        state = ChompState.initial(rows, cols)
        agent_seat = random.randint(0, 1)

        last_key: Optional[str] = None
        last_move: Optional[Move] = None

        while not state.is_terminal:
            if state.current_player == agent_seat:
                legal = state.legal_moves()
                if random.random() < self.epsilon:
                    move = random.choice(legal)
                else:
                    q = self.Q.get(state.key, {})
                    move = max(legal, key=lambda m: q.get(m, 0.0))

                if last_key is not None:
                    backup(last_key, last_move, 0.0, state)
                last_key, last_move = state.key, move
                state = state.play(move)
                if state.is_terminal:
                    backup(last_key, last_move, -1.0, None)
                    return
            else:
                move = opponent.select_move(state, state.legal_moves())
                state = state.play(move)
                if state.is_terminal:
                    if last_key is not None:
                        backup(last_key, last_move, +1.0, None)
                    return

    def memory_report(self) -> MemoryReport:
        n_states = len(self.Q)
        # One float per (state, action) pair actually stored.
        n_values = sum(len(actions) for actions in self.Q.values())
        # Rough: key string + a float (8B) per stored action value.
        est = sum(len(k) + 8 * len(a) for k, a in self.Q.items())
        return MemoryReport(n_states=n_states, n_values=n_values, est_bytes=est,
                            detail="1 value per (state, action) pair")
