"""
tictactoe.py

A module that implements a Tic-Tac-Toe game with a variable-sized board.
This implementation uses an immutable design pattern where game state changes
create new board instances rather than modifying existing ones.

Classes:
    Player: An enumeration representing players and empty cells
    Board: The main class representing the Tic-Tac-Toe board and game logic
"""

from __future__ import annotations

__version__ = "1.3.0"
__author__ = "Tin LEELAVIMOLSILP"

from typing import List, Tuple, Optional
from enum import Enum
import copy


class Player(Enum):
    """
    Represents a player and an empty cell on the tic-tac-toe board.

    This enumeration defines the three possible states for any cell on the board:
    - EMPTY: Represents an unoccupied position
    - X: Represents positions occupied by player X
    - O: Represents positions occupied by player O
    """

    EMPTY = 0  # No player (empty cell)
    X = 1  # Player X
    O = 2  # Player O

    def opposite(self) -> Player:
        """Return the opposite player."""
        if self == Player.X:
            return Player.O
        elif self == Player.O:
            return Player.X
        return Player.EMPTY  # Empty has no opposite

    def __str__(self) -> str:
        """Return a string representation of the player for board display."""
        if self == Player.X:
            return "X"
        elif self == Player.O:
            return "O"
        return "_"  # Empty cell


class Board:
    """
    Represents the tic-tac-toe board and contains all game logic.

    This class implements an immutable board design, where any modifications
    (like placing a marker) return a new board instance rather than modifying
    the current one.
    """

    def __init__(
        self,
        positions: List[List[Player]] = None,
        size: int = 3,
    ):
        """Initialize a tic-tac-toe board with optional custom configuration."""
        # Create an empty board if none provided
        if positions is None:
            self.positions: List[List[Player]] = [
                [Player.EMPTY for _ in range(size)] for _ in range(size)
            ]
            self.size = size
        else:
            self.positions = positions
            # Determine size from the provided positions
            self.size = len(positions)

    def __eq__(self, other: object) -> bool:
        """Check if two board instances represent the same game state."""
        if not isinstance(other, Board):
            return NotImplemented
        return self.positions == other.positions

    def __hash__(self) -> int:
        """Hash the board state for use in sets and dictionaries."""
        # Convert positions to a tuple of tuples (immutable) for hashing
        positions_tuple = tuple(tuple(row) for row in self.positions)
        return hash(positions_tuple)

    def __str__(self) -> str:
        """String representation of the board."""
        result = []
        size = self.size

        for row in range(size):
            result.append(
                " ".join(str(self.positions[row][col]) for col in range(size))
            )

        return "\n".join(result)

    def copy(self) -> Board:
        """Create a deep copy of the board"""
        return Board(copy.deepcopy(self.positions))

    def place_marker(self, row: int, col: int, player: Player) -> Board:
        """Place a marker on the board for the specified player."""
        if row < 0 or row >= self.size or col < 0 or col >= self.size:
            raise ValueError(f"Position ({row}, {col}) is out of bounds")

        if self.positions[row][col] != Player.EMPTY:
            raise ValueError(f"Position ({row}, {col}) is already occupied")

        # Create a new board with the move applied
        new_board = self.copy()
        new_board.positions[row][col] = player
        return new_board

    def get_cells(self, player: Player) -> List[Tuple[int, int]]:
        """Get all board positions occupied by a specific player."""
        moves: List[Tuple[int, int]] = []
        for row in range(self.size):
            for col in range(self.size):
                if self.positions[row][col] == player:
                    moves.append((row, col))
        return moves

    def evaluate_winner(self) -> Optional[Player]:
        """Check if there's a winner on the board."""
        size = self.size

        # Check rows and columns
        for i in range(size):
            # Check row i
            if self.positions[i][0] != Player.EMPTY and all(
                self.positions[i][j] == self.positions[i][0] for j in range(1, size)
            ):
                return self.positions[i][0]

            # Check column i
            if self.positions[0][i] != Player.EMPTY and all(
                self.positions[j][i] == self.positions[0][i] for j in range(1, size)
            ):
                return self.positions[0][i]

        # Check main diagonal (top-left to bottom-right)
        if self.positions[0][0] != Player.EMPTY and all(
            self.positions[i][i] == self.positions[0][0] for i in range(1, size)
        ):
            return self.positions[0][0]

        # Check other diagonal (top-right to bottom-left)
        if self.positions[0][size - 1] != Player.EMPTY and all(
            self.positions[i][size - 1 - i] == self.positions[0][size - 1]
            for i in range(1, size)
        ):
            return self.positions[0][size - 1]

        return None  # No winner

    def is_full(self) -> bool:
        """Check if the board is full (no empty cells remaining)."""
        return all(
            self.positions[row][col] != Player.EMPTY
            for row in range(self.size)
            for col in range(self.size)
        )

    def is_game_over(self) -> bool:
        """Check if the game is over (winner or draw)."""
        return self.evaluate_winner() is not None or self.is_full()


if __name__ == "__main__":
    # Example 1
    board = Board()
    print("Empty board:")
    print(board)
    print()

    try:  # Make some moves
        # Player X makes a move
        board = board.place_marker(0, 0, Player.X)
        print("After X plays at (0,0):")
        print(board)
        print()

        # Player O makes a move
        board = board.place_marker(1, 1, Player.O)
        print("After O plays at (1,1):")
        print(board)
        print()

        # Player X makes another move
        board = board.place_marker(0, 1, Player.X)
        print("After X plays at (0,1):")
        print(board)
        print()

        # Try making an invalid move
        board.place_marker(0, 0, Player.O)  # This should raise an error
    except ValueError as e:
        print(f"Invalid move: {e}")

    # Example 2
    print("\nCreating a game with multiple moves:")
    board = Board()
    moves = [
        (0, 0, Player.X),
        (0, 1, Player.O),
        (1, 1, Player.X),
        (0, 2, Player.O),
        (2, 2, Player.X),
    ]

    for row, col, player in moves:
        board = board.place_marker(row, col, player)
        print(f"After {player} plays at ({row},{col}):")
        print(board)

        # Check for winner
        winner = board.evaluate_winner()
        if winner:
            print(f"Player {winner} wins!")
            break

        # Check if game is over (draw)
        if board.is_full():
            print("The game is a draw!")
            break
