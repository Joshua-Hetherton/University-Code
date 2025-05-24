"""
tictactoe_state_graph.py - Game State Graph Generator for Tic-Tac-Toe

This module provides classes for representing Tic-Tac-Toe game states and analyzing
the complete state space as a directed graph.

The module implements two main classes:
- TicTacToeState: Represents individual states of a Tic-Tac-Toe game
- TicTacToeStateGraph: Represents the directed graph of all reachable game states

This implementation uses immutable game states and builds a directed graph where:
- Nodes represent game board configurations and player turns
- Edges represent legal moves from one state to another
"""

from __future__ import annotations

__version__ = "1.0.0"
__author__ = "Tin LEELAVIMOLSILP"

from typing import Dict, List, Set, Optional, Iterator, Tuple
from tictactoe import Board, Player

class TicTacToeState:
    """
    Represents a complete state of a Tic-Tac-Toe game including the board configuration and active player.

    This class encapsulates all information needed to represent a single point in the game's state space.
    States are immutable - methods that change the state return a new TicTacToeState object rather than
    modifying the existing one.

    States can be terminal (game over) or non-terminal (game in progress). Terminal states occur when
    one player has won or the board is full (draw).

    NOTE: The generate_successor_states method is incomplete and needs to be implemented.
    """

    def __init__(self, board: Board = None, turn: Player = Player.X, size: int = 3):
        """
        Initialize a Tic-Tac-Toe game state.

        Args:
            board: The current board configuration
            turn: Which player's turn it is
            size: Size of the board (used when board is None)
        """
        if board is None:
            self.board = Board(size=size)
        else:
            self.board = board

        self.turn = turn

    def __eq__(self, other: object) -> bool:
        """Check if two game states are equal"""
        if not isinstance(other, TicTacToeState):
            return NotImplemented
        return self.board == other.board and self.turn == other.turn

    def __hash__(self) -> int:
        """Hash the game state for use in sets and dictionaries"""
        board_hash = hash(self.board)
        turn_hash = hash(self.turn)
        return hash((board_hash, turn_hash))

    def __str__(self) -> str:
        """String representation of the game state"""
        result = str(self.board)
        result += f"\nTurn: {self.turn}"

        # Add winner information if game is over
        winner = self.evaluate_winner()
        if winner:
            result += f"\nWinner: {winner}"
        elif self.is_full():
            result += "\nGame ended in a draw"

        return result

    def copy(self) -> TicTacToeState:
        """Create a deep copy of the game state"""
        return TicTacToeState(self.board.copy(), self.turn)

    def get_moves(self) -> List[Tuple[int, int]]:
        """
        Return a list of legal moves (row, col) in this state.

        Returns:
            List of (row, col) tuples for empty positions
        """
        return self.board.get_cells(Player.EMPTY)

    def evaluate_winner(self) -> Optional[Player]:
        """
        Check if there's a winner.

        Returns:
            The player who won, or None if there's no winner
        """
        return self.board.evaluate_winner()

    def is_full(self) -> bool:
        """Check if the board is full"""
        return self.board.is_full()

    def is_game_over(self) -> bool:
        """Check if the game is over (winner or draw)"""
        return self.board.is_game_over()

    def apply_move(self, row: int, col: int) -> TicTacToeState:
        """
        Apply a move on the board and create a new game state.

        Args:
            row: Row index (0 to size-1)
            col: Column index (0 to size-1)

        Returns:
            A new game state with the move applied and the turn switched to the opposite player

        Raises:
            ValueError: If the move is invalid
        """
        # Place marker for the current player
        new_board = self.board.place_marker(row, col, self.turn)

        # Switch turns and return a new state
        return TicTacToeState(new_board, self.turn.opposite())

    def generate_successor_states(self) -> List[TicTacToeState]:
        """
        Generate all possible successor states from this state.

        This method produces all valid game states that can be reached by making a single
        move from the current state. Terminal states (where the game is over) have no
        valid successors.

        Returns:
            A list of all possible next game states after legal moves

        Example:
            >>> state = TicTacToeState()  # Empty board, X's turn
            >>> successors = state.generate_successor_states()
            >>> len(successors)  # Should be 9 for an empty 3x3 board
            9
        """
        if self.is_game_over():
            return []
        successors=[]
        for row, col in self.get_moves():
            successors.append(self.apply_move(row, col))
        return successors


        ##raise NotImplementedError("TODO: Implement generate_successor_states method")


class TicTacToeStateGraph:
    """
    Represents the complete state space of a Tic-Tac-Toe game as a directed graph with one root node.

    This class builds and manages a directed graph where:
    - Nodes are TicTacToeState objects representing distinct game configurations
    - Edges represent legal moves that transition from one state to another

    NOTE: All methods, except for __repr__, are not implemented and need to be completed.
    """

    def __init__(self):
        """
        Initialize an empty graph.
        """
        self._graph: Dict[TicTacToeState, Set[TicTacToeState]] = {}
        self._in_degrees: Dict[TicTacToeState, int] = {}
        self._root: Optional[TicTacToeState] = None
        self._adjacency_list = self._graph

        #raise NotImplementedError("TODO: Implement __init__ method")

    def add_node(self, state: TicTacToeState) -> None:
        """
        Add a node (game state) to the graph.

        Args:
            state: The game state to add as a node
        """
        if state not in self._graph:
            self._graph[state] = set()
            self._in_degrees[state] = 0
        #raise NotImplementedError("TODO: Implement add_node method")

    def add_edge(self, from_state: TicTacToeState, to_state: TicTacToeState) -> None:
        """
        Add a directed edge from one game state to another.

        Args:
            from_state: The starting game state
            to_state: The game state after a legal move

        Raises:
            ValueError: If either game state is not already a node in the graph
        """
        if from_state not in self._graph or to_state not in self._graph:
            raise ValueError("States must be added before creating an edge.")
        if to_state not in self._graph[from_state]:
            self._graph[from_state].add(to_state)
            self._in_degrees[to_state] += 1
        #raise NotImplementedError("TODO: Implement add_edge method")

    def get_next_states(self, state: TicTacToeState) -> List[TicTacToeState]:
        """
        Retrieve next states from the given game state in this graph.

        Args:
            state: The current game state

        Returns:
            List of all possible next game states

        Raises error if the state is not a node in the graph
        """
        if state not in self._graph:
            raise ValueError("State not found in graph")
        return list(self._graph[state])
        #raise NotImplementedError("TODO: Implement get_next_states method")

    def has_node(self, state: TicTacToeState) -> bool:
        """
        Check if a game state exists in the graph.

        Args:
            state: The game state to check

        Returns:
            True if the game state is in the graph, False otherwise
        """
        return state in self._graph
        #raise NotImplementedError("TODO: Implement has_node method")

    def has_edge(self, from_state: TicTacToeState, to_state: TicTacToeState) -> bool:
        """
        Check if a directed edge exists between two game states.

        Args:
            from_state: The starting game state
            to_state: The target game state

        Returns:
            True if there is an edge from from_state to to_state, False otherwise
        """
        return to_state in self._graph.get(from_state, set())
        #raise NotImplementedError("TODO: Implement has_edge method")

    def construct_from_initial_state(self, initial_state: TicTacToeState) -> None:
        """
        Construct the entire state graph starting from the given initial state.

        Args:
            initial_state: The starting game state (typically an empty board with X's turn)
                           but can be any valid mid-game position
        """
        self._root = initial_state
        queue = [initial_state]
        seen = {initial_state}
        self.add_node(initial_state)

        while queue:
            current_state = queue.pop(0)
            for successor in current_state.generate_successor_states():
                if successor not in self._graph:
                    self.add_node(successor)
                if successor not in seen:
                    queue.append(successor)
                    seen.add(successor)
                self.add_edge(current_state, successor)

        #raise NotImplementedError("TODO: Implement construct_from_initial_state method")

    @property
    def root_node(self) -> Optional[TicTacToeState]:
        """Get the root state of the graph."""
        return self._root
        #raise NotImplementedError("TODO: Implement root_node method")

    @property
    def node_count(self) -> int:
        """Get the total number of nodes (game states) in the graph."""
        return len(self._graph)
        #raise NotImplementedError("TODO: Implement node_count method")

    @property
    def edge_count(self) -> int:
        """Get the total number of edges (moves) in the graph."""
        return sum(len(neighbors) for neighbors in self._graph.values())
        #raise NotImplementedError("TODO: Implement edge_count method")

    @property
    def terminal_state_count(self) -> int:
        """Get the number of terminal states (game over states) in the graph."""
        return sum(1 for state in self._graph if state.is_game_over())
        
        #raise NotImplementedError("TODO: Implement terminal_state_count method")

    @property
    def non_terminal_state_count(self) -> int:
        """Get the number of non-terminal states in the graph."""
        return sum(1 for state in self._graph if not state.is_game_over())
        #raise NotImplementedError("TODO: Implement non_terminal_state_count method")

    def get_winning_states(self, player: Player) -> Set[TicTacToeState]:
        """
        Get all terminal states where the specified player wins.

        Args:
            player: The player (X or O) to check for wins

        Returns:
            Set of game states where the specified player wins
        """
        return {state for state in self._graph if state.evaluate_winner() == player}
        #raise NotImplementedError("TODO: Implement get_winning_states method")

    def get_draw_states(self) -> Set[TicTacToeState]:
        """
        Get all terminal states that end in a draw.

        Returns:
            Set of game states that end in a draw
        """
        return {state for state in self._graph if state.is_full() and state.evaluate_winner() is None}

        #raise NotImplementedError("TODO: Implement get_draw_states method")

    def find_path_from_root(
        self, target_state: TicTacToeState
    ) -> Optional[List[TicTacToeState]]:
        """
        Find a path from the root state to the target state.
        """
        if self._root is None:
            return None
        queue = [(self._root, [self._root])]
        visited = set()
        while queue:
            current, path = queue.pop(0)
            if current == target_state:
                return path
            if current in visited:
                continue
            visited.add(current)
            for neighbor in self._graph.get(current, []):
                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor]))
        return None
        #raise NotImplementedError("TODO: Implement find_path_from_root method")

    def get_nodes_at_depth(self, depth: int) -> Set[TicTacToeState]:
        """
        Get all states at a specific depth from the root state.

        Args:
            depth: The depth level to retrieve states from

        Returns:
            Set of states at the specified depth

        Raises:
            ValueError: If root state is not set or depth is negative
        """
        if self._root is None or depth < 0:
            raise ValueError("Invalid root or depth")
        level = 0
        current = {self._root}
        while level < depth:
            next_level = set()
            for node in current:
                next_level.update(self._graph.get(node, []))
            current = next_level
            level += 1
        return current
        #raise NotImplementedError("TODO: Implement get_nodes_at_depth method")

    def get_max_depth_from_root(self) -> int:
        """
        Calculate the maximum depth (longest path) from the root state.

        This method finds the longest possible sequence of moves from the root state
        to any reachable terminal state.

        Returns:
            The maximum depth of the graph (longest possible game from root)

        Raises:
            ValueError: If root state is not set
        """
        if self._root is None:
            raise ValueError("Root is not set")
        queue = [(self._root, 0)]
        visited = set()
        max_depth = 0
        while queue:
            current, depth = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            max_depth = max(max_depth, depth)
            for neighbor in self._graph.get(current, []):
                queue.append((neighbor, depth + 1))
        return max_depth
        #raise NotImplementedError("TODO: Implement get_max_depth_from_root method")

    def get_leaf_states(self) -> Set[TicTacToeState]:
        """
        Get all leaf states (states with no outgoing edges / terminal states).

        Returns:
            Set of leaf states
        """
        return {state for state in self._graph if not self._graph[state]}
        #raise NotImplementedError("TODO: Implement get_leaf_states method")

    def get_in_degree(self, state: TicTacToeState) -> int:
        """
        Calculate the in-degree of a game state (number of parent states).

        Args:
            state: The game state to calculate in-degree for

        Returns:
            The number of parent states that can lead to this game state

        Raises error if the state is not a node in the graph
        """
        if state not in self._in_degrees:
            raise ValueError("State not found in graph")
        return self._in_degrees[state]

        #raise NotImplementedError("TODO: Implement get_in_degree method")

    def get_out_degree(self, state: TicTacToeState) -> int:
        """
        Calculate the out-degree of a game state (number of child states).

        Args:
            state: The game state to calculate out-degree for

        Returns:
            The number of states that can be reached from this game state

        Raises error if the state is not a node in the graph
        """
        if state not in self._graph:
            raise ValueError("State not found in graph")
        return len(self._graph[state])
        #raise NotImplementedError("TODO: Implement get_out_degree method")

    def count_nodes_by_degree(self) -> Dict[str, Dict[int, int]]:
        """
        Count nodes (game states) by their in-degree and out-degree.

        This method analyzes the graph structure by counting how many states have each
        possible in-degree and out-degree value. This provides insights into the game's
        branching factor and state space connectivity.

        Returns:
            A dictionary with two keys:
                - 'in_degree': Dict mapping in-degree values to counts
                - 'out_degree': Dict mapping out-degree values to counts
        """
        in_deg_count: Dict[int, int] = {}
        out_deg_count: Dict[int, int] = {}
        for state, children in self._graph.items():
            out_deg = len(children)
            out_deg_count[out_deg] = out_deg_count.get(out_deg, 0) + 1
        for state, degree in self._in_degrees.items():
            in_deg_count[degree] = in_deg_count.get(degree, 0) + 1
        return {"in_degree": in_deg_count, "out_degree": out_deg_count}
        #raise NotImplementedError("TODO: Implement count_nodes_by_degree method")

    def __iter__(self) -> Iterator[TicTacToeState]:
        """Iterate over all nodes (game states) in the graph."""
        #raise NotImplementedError("TODO: Implement __iter__ method")
        return iter(self._graph)  # Placeholder for actual implementation which should replace None

    def __contains__(self, state: TicTacToeState) -> bool:
        """Check if a game state is in the graph."""
        return state in self._graph
        #raise NotImplementedError("TODO: Implement __contains__ method")

    def __len__(self) -> int:
        """Get the number of nodes in the graph."""
        return len(self._graph)
        #raise NotImplementedError("TODO: Implement __len__ method")

    def __repr__(self) -> str:
        """String representation of the graph."""
        root_info = f", root={self.root_node is not None}" if self.root_node else ""
        return f"TicTacToeStateGraph(nodes={self.node_count}, edges={self.edge_count}{root_info})"


# Example usage
if __name__ == "__main__":
    # This example demonstrates how to create a game state from a specific board position,
    # build the complete state graph from that position, and analyze various properties
    # of the resulting graph.

    # Start with an empty board
    board = Board()

    # Create a specific game position by placing markers
    # This creates the following board:
    # +-+-+-+
    # |X|O| |
    # +-+-+-+
    # | |X| |
    # +-+-+-+
    # |O|X|O|
    # +-+-+-+
    board = board.place_marker(0, 1, Player.O)  # O at top-middle
    board = board.place_marker(0, 0, Player.X)  # X at top-left
    board = board.place_marker(2, 0, Player.O)  # O at bottom-left
    board = board.place_marker(1, 1, Player.X)  # X at center
    board = board.place_marker(2, 2, Player.O)  # O at bottom-right
    board = board.place_marker(2, 1, Player.X)  # X at bottom-middle

    # Create a state with this board, assuming it's O's turn next
    initial_state = TicTacToeState(board, Player.O)
    print("Initial state:")
    print(initial_state)

    # Create the state graph
    graph = TicTacToeStateGraph()
    print("\nConstructing state graph...")
    graph.construct_from_initial_state(initial_state)

    # Print graph statistics
    # Display statistics about the graph structure
    # This information helps us understand the complexity of the game tree
    print("\nGraph statistics:")
    print(f"Total nodes: {graph.node_count}")
    print(f"Total edges: {graph.edge_count}")
    print(f"Terminal states: {graph.terminal_state_count}")
    print(f"Non-terminal states: {graph.non_terminal_state_count}")

    # Find and analyze the terminal states (game outcomes)
    # This shows all possible ways the game can end from the current position
    x_winning_states = graph.get_winning_states(Player.X)
    o_winning_states = graph.get_winning_states(Player.O)
    draw_states = graph.get_draw_states()

    print(f"\nX can win in {len(x_winning_states)} different ways")
    print(f"O can win in {len(o_winning_states)} different ways")
    print(f"Games can end in a draw in {len(draw_states)} different ways")

    # Check degree distribution
    degree_counts = graph.count_nodes_by_degree()
    print("\nIn-degree distribution:")
    for degree, count in sorted(degree_counts["in_degree"].items()):
        print(f"  {degree}: {count} nodes")

    print("\nOut-degree distribution:")
    for degree, count in sorted(degree_counts["out_degree"].items()):
        print(f"  {degree}: {count} nodes")
