"""
Watch two agents play Chomp against each other.

Configure the two players in the settings block below.

Controls: R restarts the game, ESC or closing the window quits.

For a minimax player, BUDGET is a fraction of all reachable positions it is
allowed to remember  None means unlimited.
"""

from chompgame import ChompGame, ChompState
from minimax_player import MinimaxPlayer
from rl_player import RLPlayer
import pygame
import sys

# settings
ROWS, COLS = 7, 7
MOVE_DELAY_MS = 1600


P0_KIND = "minimax"
P0_BUDGET = None
P1_KIND = "minimax"
P1_BUDGET = None

CELL_SIZE = 80
PADDING = 40
INFO_HEIGHT = 80

COLOR_BG = (245, 245, 220)
COLOR_CELL = (210, 180, 140)
COLOR_EATEN = (80, 50, 30)
COLOR_POISON = (180, 0, 0)
COLOR_GRID = (100, 70, 40)
COLOR_LAST = (255, 170, 60)
COLOR_TEXT = (30, 30, 30)
COLOR_WIN = (0, 150, 0)


def count_states(rows, cols):
    # total reachable non-terminal positions on this board
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


def build_agent(slot, kind, budget, rows, cols):
    # build and train one agent; slot is "P0" or "P1" so names stay distinct
    if kind == "minimax":
        if budget is None:
            cap = None
            desc = "Minimax (full)"
        else:
            cap = int(count_states(rows, cols) * budget)   # fraction -> count
            desc = f"Minimax ({int(budget * 100)}% mem)"
        agent = MinimaxPlayer(memory_budget=cap)
        agent.train(rows=rows, cols=cols, time_budget=10.0, memory_budget=cap)
    elif kind == "rl":
        agent = RLPlayer(epsilon=0.3, episodes_per_generation=500000, num_generations=1)
        desc = "RL"
        agent.train(rows=rows, cols=cols, time_budget=15.0)
    else:
        raise ValueError(f"unknown agent kind: {kind}")
    agent.name = f"{slot}: {desc}"
    print("Trained", agent.name)
    return agent


def make_agents(rows, cols):
    p0 = build_agent("P0", P0_KIND, P0_BUDGET, rows, cols)
    p1 = build_agent("P1", P1_KIND, P1_BUDGET, rows, cols)
    print("Done training, starting the game.")
    return p0, p1


class AgentViewer:
    def __init__(self, player_0, player_1, rows, cols):
        pygame.init()
        self.players = (player_0, player_1)   # player_0 moves first
        self.rows = rows
        self.cols = cols
        self.game = ChompGame(rows, cols)
        self.last_move = None
        self.last_move_time = 0

        width = cols * CELL_SIZE + 2 * PADDING
        height = rows * CELL_SIZE + 2 * PADDING + INFO_HEIGHT
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Chomp - Agent vs Agent")
        self.font = pygame.font.SysFont("Arial", 20)
        self.big_font = pygame.font.SysFont("Arial", 13, bold=True)
        self.clock = pygame.time.Clock()

    def _cell_rect(self, row, col):
        x = PADDING + col * CELL_SIZE
        y = PADDING + row * CELL_SIZE
        return pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)

    def _draw_board(self):
        g = self.game
        for r in range(g.rows):
            for c in range(g.cols):
                rect = self._cell_rect(r, c)
                if self.last_move == (r, c):
                    color = COLOR_LAST           # highlight the most recent bite
                elif g.board[r][c] == 1:
                    color = COLOR_EATEN
                elif r == 0 and c == 0:
                    color = COLOR_POISON         # the poison cell
                else:
                    color = COLOR_CELL
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, COLOR_GRID, rect, 2)

    def _draw_info(self):
        g = self.game
        board_bottom = PADDING + g.rows * CELL_SIZE
        if g.is_game_over():
            winner = self.players[g.current_player].name
            loser = self.players[1 - g.current_player].name
            msg = f"{winner} wins!   {loser} ate the poison."
            surf = self.big_font.render(msg, True, COLOR_WIN)
        else:
            mover = self.players[g.current_player].name
            msg = f"{mover} to move..."
            surf = self.font.render(msg, True, COLOR_TEXT)
        self.screen.blit(surf, surf.get_rect(centerx=self.screen.get_width() // 2,
                                              top=board_bottom + 20))

    def _step(self):
        # let the current player's agent make one move
        g = self.game
        mover = self.players[g.current_player]
        move = mover.select_move(g.snapshot(), g.legal_moves())
        g.make_move(*move)
        self.last_move = move

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit()
                    if event.key == pygame.K_r:
                        self.game = ChompGame(self.rows, self.cols)
                        self.last_move = None

            now = pygame.time.get_ticks()
            if not self.game.is_game_over() and now - self.last_move_time >= MOVE_DELAY_MS:
                self._step()
                self.last_move_time = now

            self.screen.fill(COLOR_BG)
            self._draw_board()
            self._draw_info()
            pygame.display.flip()
            self.clock.tick(60)


if __name__ == "__main__":
    p0, p1 = make_agents(ROWS, COLS)
    AgentViewer(p0, p1, ROWS, COLS).run()   # p0 moves first