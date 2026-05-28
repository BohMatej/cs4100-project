"""
This is the main file for the Chomp game. It contains the main function and the game loop.
"""

class ChompGame:
    """
    This class represents the Chomp game. It contains the game state and the game logic.
    """

    def __init__(self, rows, cols):
        """
        Initializes the game state.

        :param rows: The number of rows in the game board.
        :param cols: The number of columns in the game board.
        """
        self.rows = rows
        self.cols = cols
        self.board = [[0 for _ in range(cols)] for _ in range(rows)]
        self.current_player = 0 # 0 for first player, 1 for second player
    
    def make_move(self, row, col):
        """
        Makes a move for the current player.

        :param row: The row of the move.
        :param col: The column of the move.
        :return: True if the move is valid, False otherwise.
        """
        if row < 0 or row >= self.rows or col < 0 or col >= self.cols:
            return False
        if self.board[row][col] == 1:
            return False
        
        # Mark the move on the board
        for r in range(row, self.rows):
            for c in range(col, self.cols):
                self.board[r][c] = 1
        
        # Switch to the other player
        self.current_player = 1 - self.current_player
        return True
    
    def is_game_over(self):
        """
        Checks if the game is over.

        :return: True if the game is over, False otherwise.
        """
        return self.board[0][0] == 1
    
    def print_board(self):
        """
        Prints the current state of the game board.
        """
        for row in self.board:
            print(' '.join(str(cell) for cell in row))
        print()


    