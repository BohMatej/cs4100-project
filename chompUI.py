"""
A UI to test the game.
"""

from chompgame import ChompGame
import pygame
import sys

CELL_SIZE = 80
PADDING = 40
INFO_HEIGHT = 60

COLOR_BG = (245, 245, 220)
COLOR_CELL = (210, 180, 140)
COLOR_EATEN = (80, 50, 30)
COLOR_POISON = (180, 0, 0)
COLOR_GRID = (100, 70, 40)
COLOR_HOVER = (255, 220, 100)
COLOR_TEXT = (30, 30, 30)
COLOR_WIN = (0, 150, 0)
COLOR_LOSE = (180, 0, 0)


class ChompUI:
    def __init__(self, rows=5, cols=7):
        pygame.init()
        pygame.mixer.music.load("munch-sound-effect.mp3")
        self.sound_offset = 0.2  # seconds to skip into the file
        self.game = ChompGame(rows, cols)
        width = cols * CELL_SIZE + 2 * PADDING
        height = rows * CELL_SIZE + 2 * PADDING + INFO_HEIGHT
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Chomp")
        self.font = pygame.font.SysFont("Arial", 20)
        self.big_font = pygame.font.SysFont("Arial", 32, bold=True)
        self.hover = None
        self.clock = pygame.time.Clock()

    def _cell_rect(self, row, col):
        x = PADDING + col * CELL_SIZE
        y = PADDING + row * CELL_SIZE
        return pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)

    def _draw_board(self):
        g = self.game
        mouse_pos = pygame.mouse.get_pos()
        self.hover = None

        for r in range(g.rows):
            for c in range(g.cols):
                rect = self._cell_rect(r, c)
                if g.board[r][c] == 1:
                    color = COLOR_EATEN
                else:
                    # Determine if mouse hovers over this cell or any cell
                    # that would be eaten by clicking here
                    mx, my = mouse_pos
                    hovered_col = (mx - PADDING) // CELL_SIZE
                    hovered_row = (my - PADDING) // CELL_SIZE
                    in_range = (0 <= hovered_row < g.rows and
                                0 <= hovered_col < g.cols and
                                g.board[hovered_row][hovered_col] == 0)
                    if in_range and r >= hovered_row and c >= hovered_col:
                        color = COLOR_HOVER
                        if r == hovered_row and c == hovered_col:
                            self.hover = (hovered_row, hovered_col)
                    else:
                        color = COLOR_CELL

                    # Poison square override
                    if r == 0 and c == 0:
                        color = COLOR_POISON if color != COLOR_HOVER else (255, 100, 100)

                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, COLOR_GRID, rect, 2)

                # Draw skull on poison cell
                if r == 0 and c == 0 and g.board[0][0] == 0:
                    skull = self.font.render("☠", True, (255, 255, 255))
                    self.screen.blit(skull, skull.get_rect(center=rect.center))

    def _draw_info(self):
        g = self.game
        board_bottom = PADDING + g.rows * CELL_SIZE
        player_names = ["Player 1", "Player 2"]

        if g.is_game_over():
            # The player who just moved (1 - current) lost (they ate poison)
            loser = 1 - g.current_player
            winner = g.current_player
            msg = f"{player_names[winner]} wins! {player_names[loser]} ate the poison."
            surf = self.big_font.render(msg, True, COLOR_WIN)
        else:
            msg = f"{player_names[g.current_player]}'s turn  —  click a square to chomp"
            surf = self.font.render(msg, True, COLOR_TEXT)

        self.screen.blit(surf, surf.get_rect(centerx=self.screen.get_width() // 2,
                                              top=board_bottom + 15))

    def run(self):
        while True:
            self.screen.fill(COLOR_BG)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if not self.game.is_game_over() and self.hover is not None:
                        self.game.make_move(*self.hover)
                        pygame.mixer.music.play(start=self.sound_offset)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        self.game = ChompGame(self.game.rows, self.game.cols)

            self._draw_board()
            self._draw_info()
            pygame.display.flip()
            self.clock.tick(60)


if __name__ == "__main__":
    rows = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    cols = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    ChompUI(rows, cols).run()
