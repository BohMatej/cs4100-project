"""
The match runner that pits two players against each other.

This is the harness to compare the two bots: it drives a
``ChompGame``, asking each ``Player`` for a move on its turn, and reports who
won. ``play_series`` alternates the starting player so neither bot gets a
first-move advantage in the aggregate.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

from chompgame import ChompGame, Move
from player import Player


@dataclass
class MatchResult:
    """Outcome of a single game."""

    winner: int            # index (0 or 1) of the winning player in `players`
    players: Tuple[Player, Player]
    num_moves: int
    moves: List[Tuple[int, Move]]  # (mover index, move) in order played

    @property
    def winner_player(self) -> Player:
        return self.players[self.winner]


def play_match(player_0: Player, player_1: Player, rows: int, cols: int) -> MatchResult:
    """
    Play one game; ``player_0`` moves first.

    Raises ValueError if a player returns an illegal move, so buggy agents fail
    loudly rather than silently corrupting the comparison.
    """
    game = ChompGame(rows, cols)
    players = (player_0, player_1)
    moves: List[Tuple[int, Move]] = []

    while not game.is_game_over():
        idx = game.current_player
        mover = players[idx]
        move = mover.select_move(game.snapshot(), game.legal_moves())
        if not game.make_move(*move):
            raise ValueError(f"{mover.name} returned illegal move {move}")
        moves.append((idx, move))

    # Whoever ate the poison lost; the side to move at the terminal wins.
    return MatchResult(game.current_player, players, len(moves), moves)


def play_series(
    player_a: Player,
    player_b: Player,
    rows: int,
    cols: int,
    games: int = 100,
) -> Dict[str, int]:
    """
    Play ``games`` games, alternating who starts, and tally wins by name.

    Returns a dict mapping player name -> win count. (If both players share a
    name, give them distinct ``name`` attributes for a readable tally.)
    """
    wins = {player_a.name: 0, player_b.name: 0}
    for i in range(games):
        first, second = (player_a, player_b) if i % 2 == 0 else (player_b, player_a)
        result = play_match(first, second, rows, cols)
        wins[result.winner_player.name] += 1
    return wins


if __name__ == "__main__":
    # Smoke test: Random vs Random on a 5x7 board.
    from player import RandomPlayer
    # importing agents
    from minimax_player import MinimaxPlayer
    from rl_player import RLPlayer

    #a = RandomPlayer(seed=1)
    #a.name = "Random-A"
    #b = RandomPlayer(seed=2)
    #b.name = "Random-B"
    
    a = MinimaxPlayer()
    a.name = "Minimax-A"
    b = RLPlayer()
    b.name = "RL-B"
    
    #rows, cols = 7, 7

    print("Training Minimax")
    minimax = MinimaxPlayer()
    minimax.train(rows=3, cols=3, time_budget=10.0)

    print("Training RL")
    rl = RLPlayer(episodes_per_generation=50000, num_generations=20)
    rl.train(rows=3, cols=3, time_budget=60.0)

    print("Playing")
    results = play_series(minimax, rl, rows=3, cols=3, games=100)
    #results = play_series(a, rl, rows=3, cols=3, games=100)
    print(results)
    
    #print(play_series(a, b, rows=5, cols=7, games=1000))
