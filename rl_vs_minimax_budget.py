"""
Testing something:

Minimax only plays perfectly if it can store enough of the game tree. We cap how
many states it is allowed to remember and see how often the
trained RL agent beats it as that cap shrinks. The RL agent is trained once and
reused for every Minimax budget, so only Minimax's memory changes.
"""

from chompgame import ChompState
from player import RandomPlayer
from minimax_player import MinimaxPlayer
from rl_player import RLPlayer
from arena import play_series
import matplotlib.pyplot as plt


def count_states(rows, cols):
    # count how many non-terminal positions are reachable on this board
    seen = set()
    stack = [ChompState.initial(rows, cols)]
    while stack:
        s = stack.pop()
        if s.key in seen or s.is_terminal:
            continue
        seen.add(s.key)
        for m in s.legal_moves():
            stack.append(s.play(m))
    return len(seen)


if __name__ == "__main__":
    rows, cols = 7, 7
    games = 200
    rl_train_seconds = 15.0
    budget_fractions = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.0]

    total = count_states(rows, cols)
    print("Board", rows, "x", cols, "has", total, "reachable states")

    # train the RL agent once against random, then reuse it
    print("Training RL")
    rl = RLPlayer(epsilon=0.3, episodes_per_generation=500000, num_generations=1)
    rl.train(rows=rows, cols=cols, time_budget=rl_train_seconds)
    rl.name = "RL"
    print("RL learned", rl.memory_report().n_states, "states")

    fracs = []
    winrates = []
    for frac in budget_fractions:
        budget = max(5, int(total * frac))

        # train a fresh minimax that can only remember `budget` states
        minimax = MinimaxPlayer(memory_budget=budget)
        minimax.train(rows=rows, cols=cols, memory_budget=budget)
        minimax.name = "Minimax"

        result = play_series(rl, minimax, rows=rows, cols=cols, games=games)
        rl_wins = result.get("RL", 0)
        print("budget", budget, "-> RL won", rl_wins, "of", games)

        fracs.append(frac * 100)
        winrates.append(rl_wins / games * 100)

    # plotting
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(fracs, winrates, marker="o", color="#4C72B0", linewidth=2)
    ax.axhline(50, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Minimax memory budget (% of all reachable states)")
    ax.set_ylabel("RL win rate vs Minimax (%)")
    ax.set_title(f"RL vs memory-limited Minimax on a {rows}x{cols} board")
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
