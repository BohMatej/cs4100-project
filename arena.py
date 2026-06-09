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
    import matplotlib.pyplot as plt
    import numpy as np
    from player import RandomPlayer
    # importing agents
    from minimax_player import MinimaxPlayer
    from rl_player import RLPlayer


    rows, cols = 3, 5
    games = 100
    
    random_a = RandomPlayer(seed=1)
    random_a.name = "Random-A"
    random_b = RandomPlayer(seed=2)
    random_b.name = "Random-B"

    print("Training Minimax")
    minimax = MinimaxPlayer()
    minimax.train(rows=rows, cols=cols, time_budget=10.0)

    print("Training RL")
    rl = RLPlayer(episodes_per_generation=50000, num_generations=20)
    rl.train(rows=rows, cols=cols, time_budget=60.0)

    print("Playing")
    r_v_r = play_series(random_a, random_b, rows=rows, cols=cols, games=games)
    m_v_r = play_series(minimax, RandomPlayer(seed=1), rows=rows, cols=cols, games=games)
    rl_v_r = play_series(rl, RandomPlayer(seed=1), rows=rows, cols=cols, games=games)
    m_v_rl = play_series(minimax, rl, rows=rows, cols=cols, games=games)
    print(r_v_r, m_v_r, rl_v_r, m_v_rl)
    #results = play_series(minimax, rl, rows=rows, cols=cols, games=games)
    #results = play_series(a, rl, rows=3, cols=3, games=100)
    #print(results)
    #print(play_series(a, b, rows=5, cols=7, games=1000))
    
    # plotting
    matchups = ["Random\nvs Random", "Minimax\nvs Random", "RL\nvs Random", "Minimax\nvs RL"]
    p1_wins  = [
        r_v_r.get("Random-A", 0),
        m_v_r.get("Minimax", 0),
        rl_v_r.get("RL", 0),
        m_v_rl.get("Minimax", 0),
    ]
    p2_wins  = [games - w for w in p1_wins]
    
    x = np.arange(len(matchups))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width/2, p1_wins, width, label="Player 1 (left)", color="#4C72B0")
    bars2 = ax.bar(x + width/2, p2_wins, width, label="Player 2 (right)", color="#DD8452")

    ax.set_ylabel("Wins (out of 100)")
    ax.set_title(f"Chomp Agent Comparison — {rows}×{cols} board, {games} games each")
    ax.set_xticks(x)
    ax.set_xticklabels(matchups)
    ax.set_ylim(0, 110)
    ax.legend()
    ax.bar_label(bars1, padding=3)
    ax.bar_label(bars2, padding=3)

    plt.tight_layout()
    plt.show()
