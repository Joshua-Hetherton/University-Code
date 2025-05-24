"""
Test suite for the tictactoe module.

This module contains unit tests for the Player enumeration and Board class
from the tictactoe module. It verifies correctness of all game logic, including
board initialization, marker placement, winner evaluation, and game state checking.

Testing Strategy:
- Uses unittest framework for test execution and assertions
- Tests immutability of the Board class design
- Tests standard and edge case scenarios
- Tests various board sizes and game configurations
- Validates all game mechanics and rules enforcement
"""

__version__ = "1.0.1"
__author__ = "Tin LEELAVIMOLSILP"

import unittest
from tictactoe import Player, Board


class TestPlayer(unittest.TestCase):
    """
    Test cases for the Player enumeration.

    These tests verify the correct implementation of the Player enumeration,
    particularly the opposite() method which returns the opposing player.
    """

    def test_player_opposite(self):
        """
        Test the opposite method returns the correct player.

        Verifies:
        - Player.X.opposite() returns Player.O
        - Player.O.opposite() returns Player.X
        - Player.EMPTY.opposite() returns Player.EMPTY (stays empty)
        """
        self.assertEqual(Player.X.opposite(), Player.O)
        self.assertEqual(Player.O.opposite(), Player.X)
        self.assertEqual(Player.EMPTY.opposite(), Player.EMPTY)


class TestBoard(unittest.TestCase):
    """
    Test cases for the Board class.

    These tests verify the immutable design of the Board class and all aspects
    of game logic including initialization, marker placement, win condition
    detection, and game state evaluation.
    """

    def setUp(self):
        """
        Set up test fixtures used across multiple test methods.

        Creates several board configurations:
        - empty_board: A standard 3x3 empty board
        - custom_board: A 4x4 empty board
        - game_board: A partially played board with some markers placed
          (X at (0,0), O at (1,1), X at (0,1))
        - winning_board: A board where player X has won diagonally
          (X at (0,0), (1,1), and (2,2))
        """
        self.empty_board = Board()
        self.custom_board = Board(size=4)  # 4x4

        # Create a board with some moves
        self.game_board = Board()
        self.game_board = self.game_board.place_marker(0, 0, Player.X)
        self.game_board = self.game_board.place_marker(1, 1, Player.O)
        self.game_board = self.game_board.place_marker(0, 1, Player.X)

        # Create a winning board for X (diagonal)
        self.winning_board = Board()
        self.winning_board = self.winning_board.place_marker(0, 0, Player.X)
        self.winning_board = self.winning_board.place_marker(0, 1, Player.O)
        self.winning_board = self.winning_board.place_marker(1, 1, Player.X)
        self.winning_board = self.winning_board.place_marker(0, 2, Player.O)
        self.winning_board = self.winning_board.place_marker(2, 2, Player.X)

    def test_board_initialization(self):
        """Test that boards initialize with correct size and empty cells."""
        self.assertEqual(self.empty_board.size, 3)
        self.assertEqual(self.custom_board.size, 4)

        # Check all positions are empty on new board
        for row in range(3):
            for col in range(3):
                self.assertEqual(self.empty_board.positions[row][col], Player.EMPTY)

        # Check all positions are empty on custom board
        for row in range(4):
            for col in range(4):
                self.assertEqual(self.custom_board.positions[row][col], Player.EMPTY)

    def test_board_equality(self):
        """
        Test board equality comparison.

        Verifies:
        - Two newly initialized boards are considered equal
        - Two boards with the same moves in the same sequence are equal
        - Two boards with different configurations are not equal

        This tests the implementation of the __eq__ method in the Board class.
        """
        board1 = Board()
        board2 = Board()
        self.assertEqual(board1, board2)

        # Make same moves on both boards
        board1 = board1.place_marker(0, 0, Player.X)
        board2 = board2.place_marker(0, 0, Player.X)
        self.assertEqual(board1, board2)

        # Make different moves
        board1 = board1.place_marker(0, 1, Player.O)
        board2 = board2.place_marker(1, 0, Player.O)
        self.assertNotEqual(board1, board2)

    def test_board_hash(self):
        """
        Test that board hashing works for dictionary/set usage.

        Verifies:
        - Different board configurations have different hash values
        - Identical board configurations have the same hash value
        - Board objects work correctly as keys in dictionaries and sets

        This tests the implementation of the __hash__ method in the Board class.
        """
        board1 = Board()
        board2 = Board()
        board1 = board1.place_marker(0, 0, Player.X)

        # Test set functionality
        board_set = {board1, board2}
        self.assertEqual(len(board_set), 2)  # Two different boards

        # Add a duplicate board
        board3 = Board()
        board_set.add(board3)
        self.assertEqual(len(board_set), 2)  # board3 is same as board2

    def test_place_marker(self):
        """
        Test placing markers on the board.

        This test verifies:
        1. Markers are correctly placed at specified positions
        2. The returned board contains the markers at expected positions
        3. The original board is not modified (immutability)
        """
        board = self.empty_board

        # Place X in top left
        board = board.place_marker(0, 0, Player.X)
        self.assertEqual(board.positions[0][0], Player.X)

        # Place O in center
        board = board.place_marker(1, 1, Player.O)
        self.assertEqual(board.positions[1][1], Player.O)

        # Original board should be unchanged (immutability test)
        self.assertEqual(self.empty_board.positions[0][0], Player.EMPTY)

    def test_place_marker_invalid(self):
        """
        Test invalid marker placements.

        Verifies proper error handling in these invalid scenarios:
        - Placing a marker at negative coordinates (out of bounds)
        - Placing a marker at coordinates greater than board size (out of bounds)
        - Placing a marker at a position that is already occupied

        Each of these should raise a ValueError with an appropriate message.
        """
        board = self.empty_board.place_marker(0, 0, Player.X)

        # Out of bounds
        with self.assertRaises(ValueError):
            board.place_marker(-1, 0, Player.O)  # Negative row
        with self.assertRaises(ValueError):
            board.place_marker(0, 3, Player.O)  # Column too large
        with self.assertRaises(ValueError):
            board.place_marker(3, 0, Player.O)  # Row too large

        # Already occupied
        with self.assertRaises(ValueError):
            board.place_marker(0, 0, Player.O)  # Position already has Player.X

    def test_get_cells(self):
        """
        Test getting cells occupied by a player.

        Verifies:
        - get_cells(Player.X) returns the correct positions occupied by X
        - get_cells(Player.O) returns the correct positions occupied by O
        - get_cells(Player.EMPTY) returns all empty positions
        - The sum of cells for all players equals the total number of cells on the board
        """
        x_cells = self.game_board.get_cells(Player.X)
        o_cells = self.game_board.get_cells(Player.O)
        empty_cells = self.game_board.get_cells(Player.EMPTY)

        self.assertEqual(len(x_cells), 2)
        self.assertEqual(len(o_cells), 1)
        self.assertEqual(len(empty_cells), 6)  # 3x3 board - 3 occupied = 6 empty

        # Check specific positions
        self.assertIn((0, 0), x_cells)
        self.assertIn((0, 1), x_cells)
        self.assertIn((1, 1), o_cells)

    def test_evaluate_winner_none(self):
        """
        Test winner evaluation when there's no winner.

        Verifies that evaluate_winner() returns None when the game is still ongoing
        or when there's a draw (no winning combination on the board).
        """
        self.assertIsNone(self.game_board.evaluate_winner())

    def test_evaluate_winner_row(self):
        """
        Test winner evaluation for row win.

        Creates a board where Player X has three consecutive markers in the top row
        (positions (0,0), (0,1), and (0,2)) and verifies that evaluate_winner()
        correctly identifies Player X as the winner.
        """
        # Create board with X winning in top row
        board = Board()
        board = board.place_marker(0, 0, Player.X)
        board = board.place_marker(1, 0, Player.O)
        board = board.place_marker(0, 1, Player.X)
        board = board.place_marker(1, 1, Player.O)
        board = board.place_marker(0, 2, Player.X)  # X wins with top row

        self.assertEqual(board.evaluate_winner(), Player.X)

    def test_evaluate_winner_column(self):
        """
        Test winner evaluation for column win.

        Creates a board where Player O has three consecutive markers in the middle column
        (positions (0,1), (1,1), and (2,1)) and verifies that evaluate_winner()
        correctly identifies Player O as the winner.
        """
        # Create board with O winning in middle column
        board = Board()
        board = board.place_marker(0, 0, Player.X)
        board = board.place_marker(0, 1, Player.O)
        board = board.place_marker(2, 2, Player.X)
        board = board.place_marker(1, 1, Player.O)
        board = board.place_marker(2, 0, Player.X)
        board = board.place_marker(2, 1, Player.O)  # O wins with middle column

        self.assertEqual(board.evaluate_winner(), Player.O)

    def test_evaluate_winner_diagonal(self):
        """
        Test winner evaluation for diagonal win.

        Verifies two different diagonal win scenarios:
        1. Player X winning with markers at (0,0), (1,1), and (2,2) (main diagonal)
        2. Player O winning with markers at (0,2), (1,1), and (2,0) (anti-diagonal)

        This tests that evaluate_winner() correctly identifies diagonal wins in both directions.
        """
        # The winning_board from setUp has X winning diagonally
        self.assertEqual(self.winning_board.evaluate_winner(), Player.X)

        # Create board with O winning in other diagonal
        board = Board()
        board = board.place_marker(0, 1, Player.X)
        board = board.place_marker(0, 2, Player.O)
        board = board.place_marker(1, 0, Player.X)
        board = board.place_marker(1, 1, Player.O)
        board = board.place_marker(2, 1, Player.X)
        board = board.place_marker(2, 0, Player.O)  # O wins with diagonal

        self.assertEqual(board.evaluate_winner(), Player.O)

    def test_is_full(self):
        """
        Test checking if board is full.

        Verifies:
        - An empty board reports is_full() = False
        - A partially filled board reports is_full() = False
        - A completely filled board reports is_full() = True

        This tests the board's ability to accurately report when all positions
        have been filled with markers.
        """
        # Empty board
        self.assertFalse(self.empty_board.is_full())

        # Partially filled board
        self.assertFalse(self.game_board.is_full())

        # Create a full board
        board = Board()
        positions = [
            (0, 0, Player.X),
            (0, 1, Player.X),
            (0, 2, Player.O),
            (1, 0, Player.O),
            (1, 1, Player.O),
            (1, 2, Player.X),
            (2, 0, Player.X),
            (2, 1, Player.O),
            (2, 2, Player.X),
        ]
        for row, col, player in positions:
            board = board.place_marker(row, col, player)

        self.assertTrue(board.is_full())

    def test_is_game_over(self):
        """
        Test checking if game is over.

        Verifies game over conditions:
        - Empty board: game is not over (is_game_over() = False)
        - Partially filled board with no winner: game is not over
        - Board with a winner: game is over (is_game_over() = True)
        - Full board with no winner (draw): game is over

        This tests that the game correctly identifies when play should stop,
        either due to a win or a draw.
        """
        # Empty board - not over
        self.assertFalse(self.empty_board.is_game_over())

        # Partially filled board without winner - not over
        self.assertFalse(self.game_board.is_game_over())

        # Board with winner - game over
        self.assertTrue(self.winning_board.is_game_over())

        # Full board without winner (draw) - game over
        board = Board()
        positions = [
            (0, 0, Player.X),
            (0, 1, Player.X),
            (0, 2, Player.O),
            (1, 0, Player.O),
            (1, 1, Player.O),
            (1, 2, Player.X),
            (2, 0, Player.X),
            (2, 1, Player.O),
            (2, 2, Player.X),
        ]
        for row, col, player in positions:
            board = board.place_marker(row, col, player)

        self.assertTrue(board.is_full())
        self.assertTrue(board.is_game_over())

    def test_custom_size_win_conditions(self):
        """
        Test win conditions on a custom sized board (4x4).

        Verifies that win conditions work properly on boards of non-standard sizes:
        - Row win: Player X placing markers across an entire row
        - Column win: Player O placing markers down an entire column
        - Diagonal win: Player X placing markers along the main diagonal

        This ensures the Board class can handle different size configurations
        while maintaining proper game logic.
        """
        board = Board(size=4)

        # Create a row win
        for col in range(4):
            board = board.place_marker(1, col, Player.X)

        self.assertEqual(board.evaluate_winner(), Player.X)

        # Create a column win
        board = Board(size=4)
        for row in range(4):
            board = board.place_marker(row, 2, Player.O)

        self.assertEqual(board.evaluate_winner(), Player.O)

        # Create a diagonal win
        board = Board(size=4)
        for i in range(4):
            board = board.place_marker(i, i, Player.X)

        self.assertEqual(board.evaluate_winner(), Player.X)


if __name__ == "__main__":
    unittest.main()
