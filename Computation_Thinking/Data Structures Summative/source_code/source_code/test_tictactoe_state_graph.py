"""
Tic-Tac-Toe State Graph Unit Tests

This module contains unit tests for the TicTacToeState and TicTacToeStateGraph classes.
It verifies the correctness of:
1. Game state representation (board configuration and active player)
2. State transitions and move application
3. Game state evaluation (winners, draws, game over conditions)
4. Graph construction and traversal algorithms
5. Graph metrics and properties (node counts, degrees, paths, etc.)

The tests are organized into two main test case classes:
- TestTicTacToeState: Tests individual game state functionality
- TestTicTacToeStateGraph: Tests the directed graph representation of game states
"""

__version__ = "1.0.0"
__author__ = "Tin LEELAVIMOLSILP"

import unittest
from tictactoe_state_graph import TicTacToeState, TicTacToeStateGraph
from tictactoe import Board, Player


class TestTicTacToeState(unittest.TestCase):
    """
    Tests for the TicTacToeState class.

    This test suite verifies the functionality of the TicTacToeState class, which
    represents a complete state of a Tic-Tac-Toe game including board configuration
    and active player. The tests cover initialization, equality, immutability,
    state transitions, game state evaluation, and successor state generation.
    """

    def test_init_empty_board(self):
        """
        Test initializing a TicTacToeState with an empty board.

        Verifies that a new state is correctly created with an empty board
        and X as the starting player, with all 9 possible moves available.
        """
        state = TicTacToeState()
        self.assertEqual(state.turn, Player.X)
        self.assertEqual(
            len(state.get_moves()), 9
        )  # 3x3 empty board has 9 possible moves

    def test_init_with_board(self):
        """
        Test initializing with an existing board.

        Verifies that a state can be created with a pre-configured board
        and specified player turn, correctly maintaining the board state
        and available moves.
        """
        board = Board()
        board = board.place_marker(0, 0, Player.X)
        state = TicTacToeState(board, Player.O)
        self.assertEqual(state.turn, Player.O)
        self.assertEqual(len(state.get_moves()), 8)  # One cell is occupied

    def test_equality(self):
        """
        Test that states with the same board and turn are equal.

        Verifies that two states with identical board configurations and player turns
        are considered equal, while states that differ in either board or turn are not equal.
        """
        board1 = Board().place_marker(0, 0, Player.X)
        board2 = Board().place_marker(0, 0, Player.X)

        state1 = TicTacToeState(board1, Player.O)
        state2 = TicTacToeState(board2, Player.O)
        state3 = TicTacToeState(board1, Player.X)  # Different turn

        self.assertEqual(state1, state2)
        self.assertNotEqual(state1, state3)
        self.assertNotEqual(state2, state3)

    def test_hash(self):
        """
        Test that equal states have the same hash.

        Verifies that two equal states generate identical hash values,
        allowing them to be used properly as dictionary keys and in sets.
        """
        board1 = Board().place_marker(0, 0, Player.X)
        board2 = Board().place_marker(0, 0, Player.X)

        state1 = TicTacToeState(board1, Player.O)
        state2 = TicTacToeState(board2, Player.O)

        self.assertEqual(hash(state1), hash(state2))
        # States can be used as dictionary keys
        test_dict = {state1: "value"}
        self.assertEqual(test_dict[state2], "value")

    def test_copy(self):
        """
        Test copying a state.

        Verifies that the copy method creates a new state object that is equal to
        but distinct from the original, with its own separate board object.
        """
        original = TicTacToeState()
        copy = original.copy()

        self.assertEqual(original, copy)
        self.assertIsNot(original, copy)
        self.assertIsNot(original.board, copy.board)

    def test_apply_move(self):
        """
        Test applying a move to create a new state.

        Verifies that:
        1. After a move, a new state is created with the right marker placement
        2. The player turn switches correctly (from X to O or O to X)
        3. The original state remains unchanged (immutability principle)
        4. Invalid moves (on occupied cells) raise appropriate exceptions

        This test confirms the state transition logic and immutability of states.
        """
        state = TicTacToeState()  # Empty board, X's turn
        new_state = state.apply_move(1, 1)  # X plays in the center

        self.assertEqual(new_state.turn, Player.O)  # Turn switches to O
        self.assertEqual(
            new_state.board.positions[1][1], Player.X
        )  # X marker is placed

        # Original state is unchanged
        self.assertEqual(state.turn, Player.X)
        self.assertEqual(state.board.positions[1][1], Player.EMPTY)

        # Test invalid move
        with self.assertRaises(ValueError):
            new_state.apply_move(1, 1)  # Cell already occupied

    def test_get_moves(self):
        """
        Test getting available moves.

        Verifies that the get_moves method correctly identifies all and only
        the empty positions on the board as available moves, excluding
        positions that are already occupied.
        """
        board = Board()
        board = board.place_marker(0, 0, Player.X)  # Top-left
        board = board.place_marker(1, 1, Player.O)  # Center

        state = TicTacToeState(board, Player.X)
        moves = state.get_moves()

        self.assertEqual(len(moves), 7)  # 9 cells - 2 occupied = 7 available
        self.assertNotIn((0, 0), moves)  # Occupied by X
        self.assertNotIn((1, 1), moves)  # Occupied by O

    def test_evaluate_winner(self):
        """
        Test detecting a winner.

        Verifies that the evaluate_winner method correctly identifies:
        1. When a player has won (by creating a winning row)
        2. When there is no winner yet (game in progress)
        """
        # Create a winning board for X (horizontal top row)
        board = Board()
        board = board.place_marker(0, 0, Player.X)
        board = board.place_marker(0, 1, Player.X)
        board = board.place_marker(0, 2, Player.X)

        state = TicTacToeState(board, Player.O)
        self.assertEqual(state.evaluate_winner(), Player.X)

        # No winner
        board = Board()
        board = board.place_marker(0, 0, Player.X)
        board = board.place_marker(0, 1, Player.O)
        state = TicTacToeState(board, Player.X)
        self.assertIsNone(state.evaluate_winner())

    def test_is_full(self):
        """Test detecting a full board.

        Verifies that the is_full method correctly identifies:
        1. When the board is completely filled (no empty cells)
        2. When the board still has empty cells"""
        # Create a full board (draw)
        board = Board()
        # Fill the board in a way that results in no winner
        board = board.place_marker(0, 0, Player.X)
        board = board.place_marker(0, 1, Player.O)
        board = board.place_marker(0, 2, Player.X)
        board = board.place_marker(1, 0, Player.X)
        board = board.place_marker(1, 1, Player.O)
        board = board.place_marker(1, 2, Player.X)
        board = board.place_marker(2, 0, Player.O)
        board = board.place_marker(2, 1, Player.X)
        board = board.place_marker(2, 2, Player.O)

        state = TicTacToeState(board, Player.X)
        self.assertTrue(state.is_full())

        # Not full
        board = Board()
        board = board.place_marker(0, 0, Player.X)
        state = TicTacToeState(board, Player.O)
        self.assertFalse(state.is_full())

    def test_is_game_over(self):
        """Test detecting if the game is over"""
        # Game over by win
        board = Board()
        board = board.place_marker(0, 0, Player.X)
        board = board.place_marker(0, 1, Player.X)
        board = board.place_marker(0, 2, Player.X)

        state = TicTacToeState(board, Player.O)
        self.assertTrue(state.is_game_over())

        # Game over by draw (full board)
        board = Board()
        # Fill the board in a way that results in no winner
        board = board.place_marker(0, 0, Player.X)
        board = board.place_marker(0, 1, Player.O)
        board = board.place_marker(0, 2, Player.X)
        board = board.place_marker(1, 0, Player.X)
        board = board.place_marker(1, 1, Player.O)
        board = board.place_marker(1, 2, Player.X)
        board = board.place_marker(2, 0, Player.O)
        board = board.place_marker(2, 1, Player.X)
        board = board.place_marker(2, 2, Player.O)

        state = TicTacToeState(board, Player.X)
        self.assertTrue(state.is_game_over())

        # Game not over
        board = Board()
        board = board.place_marker(0, 0, Player.X)
        state = TicTacToeState(board, Player.O)
        self.assertFalse(state.is_game_over())

    def test_generate_successor_states(self):
        """Test generating successor states"""
        # From an empty board
        state = TicTacToeState()  # Empty board, X's turn
        successors = state.generate_successor_states()

        self.assertEqual(len(successors), 9)  # 9 possible moves on an empty board
        self.assertTrue(
            all(s.turn == Player.O for s in successors)
        )  # All turn to O after X moves

        # Game over state has no successors
        board = Board()
        board = board.place_marker(0, 0, Player.X)
        board = board.place_marker(0, 1, Player.X)
        board = board.place_marker(0, 2, Player.X)

        state = TicTacToeState(board, Player.O)
        successors = state.generate_successor_states()
        self.assertEqual(
            len(successors), 0
        )  # No successor states from a terminal state


class TestTicTacToeStateGraph(unittest.TestCase):
    """
    Tests for the TicTacToeStateGraph class.

    This test suite verifies the functionality of the TicTacToeStateGraph class, which
    represents the complete state space of a Tic-Tac-Toe game as a directed graph.
    The tests cover graph construction, node/edge management, path finding, game outcome
    analysis, and various graph metrics like node degrees and depths.

    Each test focuses on a specific aspect of the graph functionality, often using
    simplified board configurations to keep test complexity manageable.
    """

    def test_empty_graph(self):
        """Test an empty graph"""
        graph = TicTacToeStateGraph()
        self.assertEqual(graph.node_count, 0)
        self.assertEqual(graph.edge_count, 0)
        self.assertIsNone(graph.root_node)

    def test_add_node(self):
        """Test adding nodes to the graph"""
        graph = TicTacToeStateGraph()
        state = TicTacToeState()

        graph.add_node(state)
        self.assertEqual(graph.node_count, 1)
        self.assertTrue(graph.has_node(state))

    def test_add_edge(self):
        """Test adding edges to the graph"""
        graph = TicTacToeStateGraph()
        state1 = TicTacToeState()
        state2 = state1.apply_move(0, 0)  # X plays top-left

        graph.add_node(state1)
        graph.add_node(state2)
        graph.add_edge(state1, state2)

        self.assertEqual(graph.edge_count, 1)
        self.assertTrue(graph.has_edge(state1, state2))
        self.assertEqual(graph.get_next_states(state1), [state2])

    def test_construct_from_initial_state(self):
        """Test constructing a graph from an initial state"""
        # Create a nearly complete game to limit the graph size for testing
        board = Board()
        # Create the following position:
        # X O X
        # O X O
        # _ _ _
        board = board.place_marker(0, 0, Player.X)
        board = board.place_marker(0, 1, Player.O)
        board = board.place_marker(0, 2, Player.X)
        board = board.place_marker(1, 0, Player.O)
        board = board.place_marker(1, 1, Player.X)
        board = board.place_marker(1, 2, Player.O)

        initial_state = TicTacToeState(
            board, Player.X
        )  # X's turn to move in bottom row

        graph = TicTacToeStateGraph()
        graph.construct_from_initial_state(initial_state)

        self.assertEqual(graph.root_node, initial_state)
        self.assertTrue(graph.node_count > 1)  # Should have multiple states
        self.assertTrue(graph.edge_count > 0)  # Should have edges

        # Verify that terminal states are leaf nodes
        winning_states = graph.get_winning_states(Player.X)
        for state in winning_states:
            self.assertEqual(graph.get_out_degree(state), 0)

    def test_construct_from_terminal_state(self):
        """Test constructing a graph from an initial state that is already terminal.

        This test verifies that when constructing a graph from an already-won game position:
        1. The graph contains exactly one node (the initial/terminal state)
        2. The node has no outgoing edges (out-degree = 0)
        3. The graph correctly identifies it as a winning state
        """
        # Create a winning board for X (horizontal top row)
        board = Board()
        board = board.place_marker(0, 0, Player.X)
        board = board.place_marker(1, 0, Player.O)
        board = board.place_marker(0, 1, Player.X)
        board = board.place_marker(1, 1, Player.O)
        board = board.place_marker(0, 2, Player.X)  # X wins with top row

        # Create terminal state with X as winner
        initial_state = TicTacToeState(board, Player.O)

        # Verify it's indeed a terminal state
        self.assertTrue(initial_state.is_game_over())
        self.assertEqual(initial_state.evaluate_winner(), Player.X)

        # Create graph from this terminal state
        graph = TicTacToeStateGraph()
        graph.construct_from_initial_state(initial_state)

        # Check graph properties
        self.assertEqual(graph.node_count, 1, "Graph should contain exactly one node")
        self.assertEqual(graph.edge_count, 0, "Graph should have no edges")
        self.assertEqual(
            graph.root_node, initial_state, "The root node should be the initial state"
        )

        # Check that it's correctly identified as a winning state
        winning_states = graph.get_winning_states(Player.X)
        self.assertEqual(
            len(winning_states), 1, "Should identify one winning state for X"
        )
        self.assertIn(
            initial_state,
            winning_states,
            "The winning state should be in the winning states",
        )
        self.assertEqual(
            graph.get_out_degree(initial_state),
            0,
            "Terminal state should have no outgoing edges",
        )

    def test_get_winning_states(self):
        """Test getting winning states for a player"""
        # Create a simple graph with a winning position for X
        graph = TicTacToeStateGraph()

        # Initial state
        initial_state = TicTacToeState()

        # X's winning state - top row
        board = Board()
        board = board.place_marker(0, 0, Player.X)
        board = board.place_marker(0, 1, Player.X)
        board = board.place_marker(0, 2, Player.X)
        x_wins = TicTacToeState(board, Player.O)

        graph.add_node(initial_state)
        graph.add_node(x_wins)

        self.assertEqual(len(graph.get_winning_states(Player.X)), 1)
        self.assertEqual(len(graph.get_winning_states(Player.O)), 0)

    def test_get_draw_states(self):
        """Test getting draw states"""
        graph = TicTacToeStateGraph()

        # Create a draw state
        board = Board()
        # Fill the board in a way that results in no winner
        board = board.place_marker(0, 0, Player.X)
        board = board.place_marker(0, 1, Player.O)
        board = board.place_marker(0, 2, Player.X)
        board = board.place_marker(1, 0, Player.X)
        board = board.place_marker(1, 1, Player.O)
        board = board.place_marker(1, 2, Player.X)
        board = board.place_marker(2, 0, Player.O)
        board = board.place_marker(2, 1, Player.X)
        board = board.place_marker(2, 2, Player.O)

        draw_state = TicTacToeState(board, Player.X)
        graph.add_node(draw_state)

        self.assertEqual(len(graph.get_draw_states()), 1)

    def test_find_path_from_root(self):
        """Test finding a path from the root state to a target state"""
        # Create a small graph with a simple path
        graph = TicTacToeStateGraph()

        # Initial state (empty board, X's turn)
        initial_state = TicTacToeState()

        # X plays center
        board1 = Board()
        board1 = board1.place_marker(1, 1, Player.X)
        state1 = TicTacToeState(board1, Player.O)

        # O plays top-left
        board2 = board1.place_marker(0, 0, Player.O)
        state2 = TicTacToeState(board2, Player.X)

        graph.add_node(initial_state)
        graph.add_node(state1)
        graph.add_node(state2)
        graph.add_edge(initial_state, state1)
        graph.add_edge(state1, state2)

        graph._root_state = initial_state  # Set root state

        path = graph.find_path_from_root(state2)
        self.assertIsNotNone(path)
        self.assertEqual(len(path), 3)
        self.assertEqual(path[0], initial_state)
        self.assertEqual(path[1], state1)
        self.assertEqual(path[2], state2)

    def test_get_nodes_at_depth(self):
        """Test getting nodes at a specific depth"""
        graph = TicTacToeStateGraph()

        # Initial state (empty board, X's turn)
        initial_state = TicTacToeState()

        # X plays center
        board1 = Board()
        board1 = board1.place_marker(1, 1, Player.X)
        state1 = TicTacToeState(board1, Player.O)

        # O plays top-left
        board2 = board1.place_marker(0, 0, Player.O)
        state2 = TicTacToeState(board2, Player.X)

        graph.add_node(initial_state)
        graph.add_node(state1)
        graph.add_node(state2)
        graph.add_edge(initial_state, state1)
        graph.add_edge(state1, state2)

        graph._root_state = initial_state  # Set root state

        nodes_depth_0 = graph.get_nodes_at_depth(0)
        self.assertEqual(len(nodes_depth_0), 1)
        self.assertIn(initial_state, nodes_depth_0)

        nodes_depth_1 = graph.get_nodes_at_depth(1)
        self.assertEqual(len(nodes_depth_1), 1)
        self.assertIn(state1, nodes_depth_1)

    def test_get_max_depth_from_root(self):
        """Test finding the maximum depth from the root state"""
        graph = TicTacToeStateGraph()

        # Initial state (empty board, X's turn)
        initial_state = TicTacToeState()

        # X plays center
        board1 = Board()
        board1 = board1.place_marker(1, 1, Player.X)
        state1 = TicTacToeState(board1, Player.O)

        # O plays top-left
        board2 = board1.place_marker(0, 0, Player.O)
        state2 = TicTacToeState(board2, Player.X)

        graph.add_node(initial_state)
        graph.add_node(state1)
        graph.add_node(state2)
        graph.add_edge(initial_state, state1)
        graph.add_edge(state1, state2)

        graph._root_state = initial_state  # Set root state

        max_depth = graph.get_max_depth_from_root()
        self.assertEqual(max_depth, 2)

    def test_get_leaf_states(self):
        """Test getting leaf states"""
        graph = TicTacToeStateGraph()

        # Initial state (empty board, X's turn)
        initial_state = TicTacToeState()

        # X plays center
        board1 = Board()
        board1 = board1.place_marker(1, 1, Player.X)
        state1 = TicTacToeState(board1, Player.O)

        # O plays top-left
        board2 = board1.place_marker(0, 0, Player.O)
        state2 = TicTacToeState(board2, Player.X)

        graph.add_node(initial_state)
        graph.add_node(state1)
        graph.add_node(state2)
        graph.add_edge(initial_state, state1)
        graph.add_edge(state1, state2)

        leaf_states = graph.get_leaf_states()
        self.assertEqual(len(leaf_states), 1)
        self.assertIn(state2, leaf_states)

    def test_get_in_degree(self):
        """Test calculating in-degree of a state"""
        graph = TicTacToeStateGraph()

        state1 = TicTacToeState()
        state2 = state1.apply_move(0, 0)  # X plays top-left
        state3 = state1.apply_move(1, 1)  # X plays center

        graph.add_node(state1)
        graph.add_node(state2)
        graph.add_node(state3)

        graph.add_edge(state1, state2)
        graph.add_edge(state1, state3)

        self.assertEqual(graph.get_in_degree(state1), 0)
        self.assertEqual(graph.get_in_degree(state2), 1)
        self.assertEqual(graph.get_in_degree(state3), 1)

    def test_get_out_degree(self):
        """Test calculating out-degree of a state"""
        graph = TicTacToeStateGraph()

        state1 = TicTacToeState()
        state2 = state1.apply_move(0, 0)  # X plays top-left
        state3 = state1.apply_move(1, 1)  # X plays center

        graph.add_node(state1)
        graph.add_node(state2)
        graph.add_node(state3)

        graph.add_edge(state1, state2)
        graph.add_edge(state1, state3)

        self.assertEqual(graph.get_out_degree(state1), 2)
        self.assertEqual(graph.get_out_degree(state2), 0)
        self.assertEqual(graph.get_out_degree(state3), 0)

    def test_count_nodes_by_degree(self):
        """Test counting nodes by degree"""
        graph = TicTacToeStateGraph()

        state1 = TicTacToeState()
        state2 = state1.apply_move(0, 0)  # X plays top-left
        state3 = state1.apply_move(1, 1)  # X plays center

        graph.add_node(state1)
        graph.add_node(state2)
        graph.add_node(state3)

        graph.add_edge(state1, state2)
        graph.add_edge(state1, state3)

        degree_counts = graph.count_nodes_by_degree()

        self.assertIn("in_degree", degree_counts)
        self.assertIn("out_degree", degree_counts)

        self.assertEqual(degree_counts["in_degree"][0], 1)  # 1 node with in-degree 0
        self.assertEqual(degree_counts["in_degree"][1], 2)  # 2 nodes with in-degree 1

        self.assertEqual(degree_counts["out_degree"][0], 2)  # 2 nodes with out-degree 0
        self.assertEqual(degree_counts["out_degree"][2], 1)  # 1 node with out-degree 2


if __name__ == "__main__":
    unittest.main()
